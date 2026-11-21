#!/usr/bin/env python3
"""MAPLE one-click Kaggle session restore.

Run this once in a new Kaggle session (see maple_oneclick_cell.py). It restores what a session snapshot loses, starts the
heartbeat supervisor (which starts the MAPLE API on 127.0.0.1), waits for readiness and prints a SUCCESS / FAILED line for
every stage.

What it does NOT do (by design):
  * never compiles or downloads anything, never touches model files (it only reads their size / GGUF header);
  * never starts the MAPLE API directly (only the heartbeat does, through scripts/start_maple.sh);
  * never starts, stops or changes the authentication proxy, Cloudflare, sshd or any SSH access process;
  * never prints, reads out or stores an API key; never overwrites existing project files (no --overwrite-source);
  * never restarts a healthy service and never starts a second heartbeat or API.

Usage:  python3 maple_oneclick_restore.py [--check-only] [--force-restore] [--sha] [--startup-timeout 600]
                                          [--artifacts DIR ...] [--maple-home DIR] [--state-dir DIR]
Exit codes: 0 ready | 2 something required is missing | 3 GPU/disk problem | 4 restore failed | 5 start/health failed | 6 other
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

__version__ = "0.1.0"
GIB = 1024 ** 3
HERE = Path(__file__).resolve().parent
DEFAULT_HOME = "/kaggle/working/MAPLE-llama.cpp"
API_HOST = "127.0.0.1"                       # fixed on purpose: this script can never bind MAPLE elsewhere
ARCHIVES = {"llamacpp": "MAPLE-llamacpp-cuda.tar.gz", "runtime": "MAPLE-runtime.tar.gz",
            "deps": "MAPLE-python-deps.tar.gz", "complete": "MAPLE-complete.tar.gz"}
FALLBACK_ROSTER = {   # same paths as MODEL_ROSTER in src/maple/config.py (used only if the project cannot be imported)
    "deepseek": "/kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf",
    "gemma": "/kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf",
    "qwen3": "/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf",
    "fast": "/kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
}
SEARCH_ROOTS = ["/kaggle/working/maple_artifacts_flat", "/kaggle/working/artifacts", "/kaggle/input", "/kaggle/working"]
INPUT_ROOT = "/kaggle/input"                 # attached Datasets; only folders named maple* are ever entered (see safe_walk)
RESTORE_EXIT = {2: "archives missing/corrupt (checksum)", 3: "model path missing", 4: "CUDA/binary incompatibility or build mismatch",
                5: "smoke/API test failed", 6: "other error (e.g. not enough disk)"}


# ------------------------------------------------------------------ small helpers
def run(cmd, timeout=60, env=None, cwd=None, inp=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd, input=inp)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timed out after %ss" % timeout
    except OSError as e:
        return 127, str(e)


def procs():
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                out[int(d)] = Path("/proc/%s/cmdline" % d).read_bytes().decode("utf-8", "replace").replace("\0", " ").strip()
            except OSError:
                pass
    return out


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def listeners():
    """{port: sorted bind addresses} of TCP listeners (IPv4 + IPv6)."""
    res = {}
    for f, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            p = ln.split()
            if p[3] != "0A":
                continue
            a, port = p[1].rsplit(":", 1)
            if v6:
                ip = "::1" if a == "0" * 31 + "1" else ("::" if set(a) == {"0"} else "ipv6")
            else:
                ip = ".".join(str(int(a[i:i + 2], 16)) for i in (6, 4, 2, 0))
            res.setdefault(int(port, 16), set()).add(ip)
    return {k: sorted(v) for k, v in res.items()}


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa
        return None, "%s: %s" % (type(e).__name__, e)


def tail(path, n=12, width=200):
    try:
        return "\n".join("      | " + l[:width] for l in Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "      | (no log)"


def gib(n):
    return "%.2f GiB" % (n / GIB)


# ------------------------------------------------------------------ run context
class Ctx:
    def __init__(self, a):
        self.a = a
        self.home = Path(a.maple_home)
        self.compat = HERE
        self.state = Path(a.state_dir) if a.state_dir else HERE / "state"
        self.api_port, self.llama_port = a.api_port, a.llama_port
        self.hb_script = self.compat / "maple_heartbeat.py"
        self.bin_dir = self.home / "vendor/llamacpp/llama.cpp/build/bin"
        self.roster = dict(FALLBACK_ROSTER)
        self.expected = {}
        self.artifact_dir = None
        self.notes = {}

    def py(self):
        return shutil.which("python3") or sys.executable

    def llama_env(self):
        env = dict(os.environ)
        env.pop("API_KEY", None)
        libs = ["/usr/local/nvidia/lib64"]
        envf = self.home / "maple.env"
        if envf.is_file():
            for ln in envf.read_text(errors="replace").splitlines():
                m = re.match(r"\s*export\s+LD_LIBRARY_PATH=(.*)", ln)
                if m:
                    libs = [x for x in m.group(1).strip().strip("'\"").split(":") if x and not x.startswith("$")] + libs
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(libs + [env.get("LD_LIBRARY_PATH", "")]))
        return env


class Skip(Exception):
    pass


# ------------------------------------------------------------------ stages (each returns (status, detail))
def s_session(c):
    up = float(Path("/proc/uptime").read_text().split()[0])
    rc, kern = run(["uname", "-r"])
    me = pid_alive(os.getpid())
    pr = procs()
    info = [(p, cmd[:60]) for p, cmd in pr.items() if re.search(r"cloudflared|sshd: /usr/sbin/sshd", cmd)]
    lines = ["hostname %s | %s UTC | kernel %s | python %s | uptime %.0f min" % (socket.gethostname(), time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), kern.strip(), sys.version.split()[0], up / 60),
             "/kaggle present: %s | user: %s | cwd: %s" % (Path("/kaggle").is_dir(), os.environ.get("USER") or os.getuid(), os.getcwd()),
             "SSH access processes (informational, never touched by this script): %s" % (", ".join("%d %s" % x for x in info[:4]) or "none found")]
    c.notes["hostname"] = socket.gethostname()
    return ("SUCCESS" if Path("/kaggle").is_dir() else "WARN"), "\n      ".join(lines)


def s_gpu(c):
    smi = shutil.which("nvidia-smi") or ("/opt/bin/nvidia-smi" if Path("/opt/bin/nvidia-smi").exists() else None)
    if not smi:
        return "FAILED", "nvidia-smi not found. Turn the GPU on: notebook Settings > Accelerator > GPU T4 x2, then run this cell again."
    env = c.llama_env()
    rc, out = run([smi, "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"], env=env, timeout=30)
    gpus = [l.strip() for l in out.splitlines() if re.match(r"\d+,", l.strip())]
    if rc != 0 or not gpus:
        return "FAILED", "nvidia-smi reported no GPU (rc %s): %s. Enable the GPU accelerator (T4 x2) and restart the session." % (rc, out.strip()[:150])
    c.notes["gpus"] = len(gpus)
    names = {g.split(",")[1].strip() for g in gpus}
    note = "" if len(gpus) >= 2 else "  [only 1 GPU: the large models' two-GPU layer split cannot be used; two-T4 operation is not claimed]"
    used = " | ".join("GPU%s %s used of %s" % (g.split(",")[0], g.split(",")[2].strip(), g.split(",")[3].strip()) for g in gpus)
    return "SUCCESS", "%d GPU(s): %s | %s%s" % (len(gpus), ", ".join(sorted(names)), used, note)


def s_disk(c):
    d = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else ".")
    free = d.free
    detail = "/kaggle/working: %s free of %s" % (gib(free), gib(d.total))
    if free < 0.3 * GIB:
        return "FAILED", detail + " - below 0.3 GiB; free space (delete your own old outputs) before continuing"
    if free < 1 * GIB:
        return "WARN", detail + " - low (restore needs about 0.2 GiB)"
    return "SUCCESS", detail


def safe_walk(root, maxdepth=3):
    """os.walk with a depth limit. Under /kaggle/input it ONLY enters dataset folders whose name starts with 'maple' (and the
    'datasets/<owner>' container levels), so unrelated Datasets - for example one that holds SSH keys - are never opened or listed.
    To use a differently named Dataset, pass its folder explicitly (--artifacts / --models-source): explicit folders are always searched."""
    root = root.rstrip("/")
    guarded = root == INPUT_ROOT
    base = root.count("/")
    for dp, dn, fn in os.walk(root):
        depth = dp.count("/") - base
        if depth >= maxdepth:
            dn[:] = []
        elif guarded:
            rel = [x for x in dp[len(root):].split("/") if x]
            inside = (len(rel) >= 1 and rel[0].lower().startswith("maple")) or (len(rel) >= 3 and rel[0] == "datasets" and rel[2].lower().startswith("maple"))
            if not inside:      # entered a maple* Dataset: search it freely (depth-limited); elsewhere only maple* folders and the container levels
                dn[:] = [d for d in dn if (rel == [] and (d == "datasets" or d.lower().startswith("maple"))) or rel == ["datasets"]
                         or (len(rel) == 2 and rel[0] == "datasets" and d.lower().startswith("maple"))]
        yield dp, dn, fn


def find_gguf_sources(extra):
    """Folders (explicit ones, plus maple* Datasets under /kaggle/input) that contain *.gguf files."""
    out = []
    for root in list(extra or []) + [INPUT_ROOT]:
        if os.path.isdir(root):
            for dp, dn, fn in safe_walk(root, 4):
                if any(f.lower().endswith(".gguf") for f in fn):
                    out.append(dp)
    return list(dict.fromkeys(out))


def find_artifacts(extra):
    roots = list(extra or []) + SEARCH_ROOTS
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in safe_walk(root, 3):
            if dp.startswith("/kaggle/working/MAPLE-llama.cpp") or "/models" in dp:
                dn[:] = []
                continue
            names = [n for n in fn if n in ARCHIVES.values()]
            if names:
                found.setdefault(dp, set()).update(names)
    for dp, names in found.items():
        has_sum = (Path(dp) / "checksums.sha256").is_file()
        parts = ARCHIVES["llamacpp"] in names and ARCHIVES["runtime"] in names
        if has_sum and (parts or ARCHIVES["complete"] in names):
            return dp, sorted(names), None
    if found:
        dp, names = next(iter(found.items()))
        miss = [n for k, n in ARCHIVES.items() if k in ("llamacpp", "runtime") and n not in names]
        if not (Path(dp) / "checksums.sha256").is_file():
            miss.append("checksums.sha256")
        return None, sorted(names), miss
    return None, [], [ARCHIVES["llamacpp"], ARCHIVES["runtime"], "checksums.sha256"]


def s_layout(c):
    rows, bad = [], []

    def chk(label, path, required=False, kind="any"):
        p = Path(path)
        ok = p.is_dir() if kind == "dir" else p.is_file() if kind == "file" else p.exists()
        rows.append("%-34s %s  %s" % (label, "OK     " if ok else ("MISSING" if required else "absent "), path))
        if required and not ok:
            bad.append(path)
        return ok

    chk("heartbeat script (REQUIRED)", c.hb_script, True, "file")
    chk("tunnel script (not started here)", c.compat / "maple_tunnel.py", False, "file")
    try:
        c.state.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        bad.append("%s (%s)" % (c.state, e))
    chk("state directory", c.state, True, "dir")
    chk("MAPLE project", c.home, False, "dir")
    chk("models directory", "/kaggle/working/models", False, "dir")
    adir, names, miss = find_artifacts(c.a.artifacts)
    c.artifact_dir = adir
    rows.append("%-34s %s  %s" % ("restore artifacts", "OK     " if adir else "absent ", adir or "(none complete; found: %s)" % (names or "nothing")))
    if bad:
        return "FAILED", "\n      ".join(rows) + "\n      MISSING: %s\n      The supervisor scripts are not part of the MAPLE archives. Copy supervisor/maple_heartbeat.py and maple_tunnel.py from the repository " \
               "(https://github.com/devbbsr23/MAPLE-llama.cpp, folder supervisor/) into %s, then run again." % (", ".join(bad), c.compat)
    return "SUCCESS", "\n      ".join(rows)


def project_state(c):
    proj = (c.home / "src/maple/__init__.py").is_file() and (c.home / "scripts/start_maple.sh").is_file()
    srv = c.bin_dir / "llama-server"
    exe = srv.is_file() and os.access(srv, os.X_OK)
    cuda = bool(list(c.bin_dir.glob("libggml-cuda.so*"))) if c.bin_dir.is_dir() else False
    nonexec = [p.name for p in c.bin_dir.iterdir() if p.is_file() and not p.name.endswith(".so") and ".so." not in p.name and not os.access(p, os.X_OK)] if c.bin_dir.is_dir() else []
    return proj, exe, cuda, nonexec


def s_restore(c):
    proj, exe, cuda, nonexec = project_state(c)
    need = (not proj) or (not exe) or (not cuda) or bool(nonexec) or c.a.force_restore
    why = []
    if not proj:
        why.append("project files missing")
    if not exe:
        why.append("llama-server missing or not executable (permission bits stripped by the snapshot)")
    if not cuda:
        why.append("libggml-cuda missing")
    if nonexec:
        why.append("%d tool(s) without execute bit" % len(nonexec))
    if c.a.force_restore:
        why.append("--force-restore")
    if not need:
        raise Skip("not needed: project present, llama-server executable (%s), CUDA backend present" % (c.bin_dir / "llama-server"))
    if not c.artifact_dir:
        _, names, miss = find_artifacts(c.a.artifacts)
        return "FAILED", ("restore is needed (%s) but the archives are not available.\n      MISSING: %s\n      Looked in: %s\n"
                          "      Attach/upload the MAPLE archives together with checksums.sha256 (see docs/RESTORATION.md). Nothing was changed." %
                          ("; ".join(why), ", ".join(miss or ["archives"]), ", ".join(c.a.artifacts or []) + (", " if c.a.artifacts else "") + ", ".join(SEARCH_ROOTS)))
    if c.a.check_only:
        raise Skip("CHECK-ONLY: would run restore_kaggle.py (%s) with archives from %s" % ("; ".join(why), c.artifact_dir))
    script = c.home / "scripts/restore_kaggle.py"
    tmp = None
    if not script.is_file():
        rt = Path(c.artifact_dir) / ARCHIVES["runtime"]
        if not rt.is_file():
            return "FAILED", "scripts/restore_kaggle.py is missing and %s is not available to extract it from" % ARCHIVES["runtime"]
        tmp = tempfile.mkdtemp(prefix="maple_restore_")
        with tarfile.open(rt, "r:gz") as t:
            m = next((x for x in t if x.name.endswith("scripts/restore_kaggle.py") and x.isfile()), None)
            if not m:
                return "FAILED", "restore_kaggle.py not found inside %s" % rt
            m.name = "restore_kaggle.py"
            t.extract(m, tmp)
        script = Path(tmp) / "restore_kaggle.py"
    cmd = [c.py(), "-I", str(script), "--artifacts", str(c.artifact_dir), "--dest", str(c.home), "--no-sha", "--skip-smoke", "--skip-api"]
    print("      running: " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc, out = run(cmd, timeout=900)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    for l in out.strip().splitlines()[-14:]:
        print("      | " + l[:190])
    proj, exe, cuda, nonexec = project_state(c)
    if rc != 0:
        return "FAILED", "restore_kaggle.py exit %d: %s" % (rc, RESTORE_EXIT.get(rc, "see output above"))
    if not (proj and exe and cuda):
        return "FAILED", "restore finished but the result is incomplete (project=%s llama-server executable=%s cuda=%s)" % (proj, exe, cuda)
    return "SUCCESS", "restored in %.0f s (%s); no compilation, no download, no overwrite of existing project files" % (time.time() - t0, "; ".join(why))


def s_models(c):
    env = dict(os.environ, PYTHONPATH=str(c.home / "src"))
    rc, out = run([c.py(), "-s", "-c", "import json;from maple.config import MODEL_ROSTER as R;print(json.dumps({k:v['model_path'] for k,v in R.items()}))"], env=env, timeout=30)
    src = "project config"
    try:
        c.roster = json.loads(out.strip().splitlines()[-1])
    except Exception:
        src = "built-in fallback list (project config not importable: %s)" % out.strip()[-80:]
    try:
        man = json.loads((c.home / "manifests/models_manifest.json").read_text())
        man = man.get("models") or man
        c.expected = {k: v for k, v in man.items() if isinstance(v, dict)}
    except Exception:
        c.expected = {}

    def state():
        res = {}
        for mid, path in c.roster.items():
            p = Path(path)
            want = (c.expected.get(mid) or {}).get("size_bytes")
            if not p.exists():
                res[mid] = ("MISSING", path, want)
                continue
            size = p.stat().st_size
            try:
                magic = p.open("rb").read(4)
            except OSError:
                magic = b""
            if magic != b"GGUF":
                res[mid] = ("BAD-HEADER", path, want)
            elif want and size != want:
                res[mid] = ("SIZE-MISMATCH", "%s (size %d, expected %d)" % (path, size, want), want)
            else:
                res[mid] = ("OK", path, size)
        return res

    st = state()
    missing = [m for m, v in st.items() if v[0] == "MISSING"]
    linked = ""
    gsrc = find_gguf_sources(c.a.models_source) if missing and (c.home / "scripts/link_models.py").is_file() else []
    if gsrc:
        if c.a.check_only:
            linked = " [CHECK-ONLY: would run scripts/link_models.py --apply with --source %s (symlinks only)]" % ", ".join(gsrc)
        else:
            cmd = [c.py(), "-I", str(c.home / "scripts/link_models.py"), "--apply"]
            for s in gsrc:
                cmd += ["--source", s]
            rc, lo = run(cmd, timeout=120)
            linked = " [ran link_models.py --apply (symlinks only, never copies/overwrites): rc %d]" % rc
            st = state()
    rows = ["%-9s %-13s %s%s" % (m, v[0], v[1], (" (%s)" % gib(v[2])) if v[0] == "OK" else "") for m, v in st.items()]
    bad = {m: v for m, v in st.items() if v[0] != "OK"}
    if c.a.sha and not bad:
        import hashlib
        for mid, v in st.items():
            h = hashlib.sha256()
            with open(v[1], "rb") as f:
                for blk in iter(lambda: f.read(8 << 20), b""):
                    h.update(blk)
            want = (c.expected.get(mid) or {}).get("sha256")
            if want and h.hexdigest() != want:
                bad[mid] = ("SHA-MISMATCH", v[1], None)
            rows.append("%-9s sha256 %s" % (mid, "match" if want and h.hexdigest() == want else ("MISMATCH" if want else "no recorded hash")))
    if bad:
        miss_txt = "\n      ".join("%s: %s -> %s" % (m, v[0], v[1]) for m, v in bad.items())
        return "FAILED", "roster from %s\n      %s\n      PROBLEM:\n      %s\n      Models are never downloaded or replaced by MAPLE. Attach the Dataset that holds them (then run again; " \
                         "scripts/link_models.py links them), or place the exact files at the paths above." % (src, "\n      ".join(rows), miss_txt)
    return "SUCCESS", "%d models present, GGUF headers valid, sizes match the manifest%s%s\n      %s" % (len(st), " (sha256 verified)" if c.a.sha else " (sha256 not recomputed; use --sha)", linked, "\n      ".join(rows))


def s_deps(c):
    if sys.version_info < (3, 9):
        return "FAILED", "Python >= 3.9 required (found %s)" % sys.version.split()[0]
    env = dict(os.environ, PYTHONPATH="%s:%s" % (c.home / "vendor/pydeps", c.home / "src"))
    rc, out = run([c.py(), "-s", "-c", "import yaml,maple,sys;print('pyyaml',yaml.__version__,yaml.__file__);print('maple',maple.__file__)"], env=env, timeout=30)
    bash = shutil.which("bash")
    if rc != 0:
        return "FAILED", "import failed: %s\n      Run again with --force-restore (re-installs the bundled wheels offline into vendor/pydeps; no internet needed)." % out.strip()[-200:]
    if not bash:
        return "FAILED", "bash not found (scripts/start_maple.sh needs it)"
    return "SUCCESS", "python %s | %s | bash %s | stdlib-only runtime + PyYAML from the offline bundle" % (sys.version.split()[0], " | ".join(out.strip().splitlines()[-2:]), bash)


SECRETISH = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL", re.I)


def s_env(c):
    envf = c.home / "maple.env"
    if not envf.is_file():
        return "FAILED", "%s missing. Run again with --force-restore (restore_kaggle.py regenerates it; it contains no secrets)." % envf
    names, warn = {}, []
    for ln in envf.read_text(errors="replace").splitlines():
        m = re.match(r"\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)", ln)
        if m:
            names[m.group(1)] = m.group(2).strip().strip("'\"")
            if SECRETISH.search(m.group(1)):
                warn.append(m.group(1))
    rows = ["variables defined (values not printed): %s" % ", ".join(sorted(names))]
    if "MAPLE_HOME" in names:
        rows.append("MAPLE_HOME points to this project: %s" % (os.path.realpath(names["MAPLE_HOME"]) == os.path.realpath(str(c.home))))
    if "MAPLE_LLAMA_BIN" in names:
        b = Path(names["MAPLE_LLAMA_BIN"])
        rows.append("MAPLE_LLAMA_BIN exists: %s" % (b.exists()))
    if "LD_LIBRARY_PATH" in names:
        dirs = [d for d in names["LD_LIBRARY_PATH"].split(":") if d and "$" not in d]
        rows.append("LD_LIBRARY_PATH dirs: %d, all exist: %s" % (len(dirs), all(Path(d).is_dir() for d in dirs)))
    rows.append("API_KEY set in this process's environment: %s (the heartbeat strips it before starting MAPLE)" % ("yes" if os.environ.get("API_KEY") else "no"))
    if warn:
        return "WARN", "\n      ".join(rows) + "\n      secret-looking variable name(s) in maple.env: %s (values not shown) - maple.env should contain no secrets" % ", ".join(warn)
    return "SUCCESS", "\n      ".join(rows)


def s_llama(c):
    srv = c.bin_dir / "llama-server"
    if not (srv.is_file() and os.access(srv, os.X_OK)):
        return "FAILED", "%s is not executable. Run again with --force-restore." % srv
    rc, out = run([str(srv), "--version"], env=c.llama_env(), timeout=40)
    ver = next((l.strip() for l in out.splitlines() if "version" in l.lower()), out.strip()[-120:])
    if rc != 0:
        return "FAILED", "llama-server --version failed (rc %s): %s" % (rc, out.strip()[-200:])
    return "SUCCESS", "%s executable | %s (no model loaded by this check)" % (srv, ver[:120])


def heartbeat_pids(c):
    me = os.getpid()
    return [p for p, cmd in procs().items() if p != me and "maple_heartbeat.py" in cmd and re.search(r"\brun\b", cmd)]


def api_pids(c):
    return [p for p, cmd in procs().items() if re.search(r"maple\s+serve", cmd) and ("--port %d" % c.api_port) in cmd]


def s_heartbeat(c):
    lock = {}
    try:
        lock = json.loads((c.state / "supervisor.lock").read_text())
    except Exception:
        pass
    alive = heartbeat_pids(c)
    lp = int(lock.get("pid") or 0)
    if alive:
        extra = "" if lp in alive else " (lock file says pid %s: stale or foreign; the heartbeat repairs its own lock)" % (lp or "none")
        if len(alive) > 1:
            return "WARN", "MORE THAN ONE heartbeat running: pids %s. Not starting another. Stop the extras with maple_heartbeat.py stop (see guide section 11)." % alive
        raise Skip("already running (pid %d); not starting a duplicate%s" % (alive[0], extra))
    if lp:
        print("      note: stale supervisor.lock (pid %d not running): the heartbeat replaces it" % lp)
    port_open = c.api_port in listeners()
    if port_open and not api_pids(c):
        return "FAILED", "port %d is held by a process that is not the MAPLE API. Not starting anything. Identify it with: ss -ltnp | grep :%d" % (c.api_port, c.api_port)
    if c.a.check_only:
        raise Skip("CHECK-ONLY: would start the heartbeat detached (it then starts/adopts the MAPLE API on %s:%d)" % (API_HOST, c.api_port))
    cmd = [c.py(), "-B", str(c.hb_script), "run", "--maple-home", str(c.home), "--state-dir", str(c.state)]
    if c.api_port != 8000:
        cmd += ["--api-port", str(c.api_port)]
    if c.llama_port != 8081:
        cmd += ["--llama-port", str(c.llama_port)]
    env = {k: v for k, v in os.environ.items() if k != "API_KEY"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    log = c.state / "heartbeat_console.log"
    with open(log, "ab") as console:
        sup = subprocess.Popen(cmd, cwd=str(c.compat), stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    c.notes["hb_proc"] = sup
    time.sleep(3)
    rc = sup.poll()
    if rc is not None:
        return "FAILED", "heartbeat exited immediately with code %s (3 = another heartbeat holds the lock):\n%s" % (rc, tail(log))
    return "SUCCESS", "heartbeat started detached, pid %d | command: %s\n      The heartbeat starts the MAPLE API itself (scripts/start_maple.sh --host %s --port %d); this script never launches the API." % (sup.pid, " ".join(cmd), API_HOST, c.api_port)


def s_ready(c):
    url = "http://%s:%d/health" % (API_HOST, c.api_port)
    st, _ = http_get(url, 5)
    if st == 200 and not (c.a.check_only and False):
        return "SUCCESS", "MAPLE API already answering on %s:%d (HTTP 200); nothing restarted" % (API_HOST, c.api_port)
    if c.a.check_only:
        return "FAILED", "CHECK-ONLY: the API is not answering (%s) and nothing was started" % st
    t0, last = time.time(), 0
    sup = c.notes.get("hb_proc")
    while time.time() - t0 < c.a.startup_timeout:
        if sup is not None and sup.poll() is not None:
            return "FAILED", "heartbeat exited with code %s while waiting:\n%s" % (sup.poll(), tail(c.state / "heartbeat_console.log"))
        st, _ = http_get(url, 5)
        if st == 200:
            return "SUCCESS", "MAPLE API healthy after %.0f s (HTTP 200)" % (time.time() - t0)
        if time.time() - last > 20:
            print("      ... waiting for the API (%.0f s, last status %s)" % (time.time() - t0, st), flush=True)
            last = time.time()
        time.sleep(2)
    return "FAILED", "API not healthy after %d s. The heartbeat keeps running and retrying. Evidence:\n      heartbeat.log\n%s\n      maple_api.log\n%s" % (
        c.a.startup_timeout, tail(c.state / "heartbeat.log"), tail(c.state / "maple_api.log"))


def s_endpoints(c):
    base = "http://%s:%d" % (API_HOST, c.api_port)
    st, body = http_get(base + "/health", 15)
    if st != 200:
        return "FAILED", "/health -> HTTP %s" % st
    try:
        h = json.loads(body)
    except Exception:
        h = {}
    status = h.get("status")
    st2, body2 = http_get(base + "/v1/models", 15)
    if st2 != 200:
        return "FAILED", "/v1/models -> HTTP %s" % st2
    try:
        ids = [m["id"] for m in json.loads(body2)["data"]]
    except Exception:
        return "FAILED", "/v1/models returned an unexpected body"
    want = set(c.roster) | {"auto"}
    missing = sorted(want - set(ids))
    rt = (h.get("runtime") or {})
    d = "/health HTTP 200 status=%r | /v1/models HTTP 200 ids=%s | model loaded now: %s (models load on the first request)" % (status, ids, rt.get("loaded") or "none")
    if missing:
        return "FAILED", d + " | MISSING ids: %s" % missing
    if status != "ok":
        return "WARN", d + " | health is not 'ok': %s" % str(h.get("missing") or h)[:200]
    return "SUCCESS", d


def s_processes(c):
    pr, ls = procs(), listeners()
    me = os.getpid()
    rows = []
    hb = heartbeat_pids(c)
    api = api_pids(c)
    llama = [p for p, cmd in pr.items() if "llama-server" in cmd.split(" ")[0] or re.search(r"/llama-server\s", cmd)]
    ssh = [p for p, cmd in pr.items() if re.search(r"cloudflared|sshd: /usr/sbin/sshd", cmd) and p != me]
    proxy = [p for p, cmd in pr.items() if "maple_tunnel.py" in cmd and "run" in cmd and p != me]
    rows.append("heartbeat pid(s): %s | MAPLE API pid(s): %s | llama-server pid(s): %s (none until a model is requested)" % (hb or "none", api or "none", llama or "none"))
    rows.append("SSH access pid(s), not managed here: %s | auth proxy pid(s): %s (this script does not start it)" % (ssh or "none", proxy or "none"))
    bad = []
    for port, label in ((c.api_port, "MAPLE API"), (c.llama_port, "llama-server (internal)"), (8088, "auth proxy"), (2222, "sshd"), (8888, "Jupyter (Kaggle's own)")):
        a = ls.get(port)
        if a is None:
            rows.append("port %-5d %-24s not listening" % (port, label))
            continue
        loop = all(x in ("127.0.0.1", "::1") for x in a)
        rows.append("port %-5d %-24s listening on %s%s" % (port, label, ",".join(a), "" if loop else "  <-- NOT loopback"))
        if not loop and port in (c.api_port, c.llama_port, 8088):
            bad.append("%s on %s" % (label, a))
    st, _ = run([shutil.which("nvidia-smi") or "/opt/bin/nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"], env=c.llama_env(), timeout=20)
    rows.append("GPU memory now: %s" % " | ".join(l.strip() for l in _.strip().splitlines()))
    c.notes.update(hb=hb, api=api, llama=llama)
    if bad:
        return "FAILED", "\n      ".join(rows) + "\n      EXPOSED: " + "; ".join(bad) + " - MAPLE must be bound to 127.0.0.1 only. Stop it with maple_heartbeat.py stop --all and report this."
    if not hb or not api:
        return "FAILED", "\n      ".join(rows) + "\n      expected a heartbeat and a MAPLE API process"
    return "SUCCESS", "\n      ".join(rows)


STAGES = [
    ("Session information", s_session, "env"),
    ("GPU availability (nvidia-smi)", s_gpu, "env"),
    ("Disk space", s_disk, "env"),
    ("Required directories and artifacts", s_layout, "miss"),
    ("Project and llama.cpp binaries (restore_kaggle.py if needed)", s_restore, "restore"),
    ("Model files (presence, GGUF header, size)", s_models, "miss"),
    ("Python and dependencies", s_deps, "miss"),
    ("maple.env (no values printed)", s_env, "miss"),
    ("llama-server runtime check", s_llama, "restore"),
    ("Heartbeat supervisor", s_heartbeat, "start"),
    ("MAPLE API readiness", s_ready, "start"),
    ("Endpoint tests (/health, /v1/models)", s_endpoints, "start"),
    ("Processes and listening ports", s_processes, "start"),
]
EXIT = {"env": 3, "miss": 2, "restore": 4, "start": 5}


def main(argv=None):
    ap = argparse.ArgumentParser(description="MAPLE one-click Kaggle session restore (see the module docstring)")
    ap.add_argument("--check-only", action="store_true", help="run every check; change nothing and start nothing")
    ap.add_argument("--force-restore", action="store_true", help="run restore_kaggle.py even if the project looks complete (never overwrites existing files)")
    ap.add_argument("--sha", action="store_true", help="also recompute model SHA256 (reads ~16 GB; several minutes)")
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--artifacts", action="append", help="extra folder to search for MAPLE-*.tar.gz + checksums.sha256 (repeatable)")
    ap.add_argument("--models-source", action="append", help="extra folder that holds the model .gguf files, to be symlinked if missing (repeatable); /kaggle/input is searched only inside maple* Datasets")
    ap.add_argument("--maple-home", default=DEFAULT_HOME)
    ap.add_argument("--state-dir")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--llama-port", type=int, default=8081)
    a = ap.parse_args(argv)
    c = Ctx(a)
    print("=" * 78)
    print("MAPLE ONE-CLICK RESTORE v%s%s" % (__version__, "  [CHECK-ONLY: nothing will be changed or started]" if a.check_only else ""))
    print("=" * 78, flush=True)
    results, code, t_all = [], 0, time.time()
    for i, (name, fn, kind) in enumerate(STAGES, 1):
        t0 = time.time()
        try:
            status, detail = fn(c)
        except Skip as s:
            status, detail = "SKIPPED", str(s)
        except Exception as e:  # noqa
            status, detail = "FAILED", "unexpected error: %s: %s" % (type(e).__name__, e)
        results.append((i, name, status, detail, time.time() - t0))
        print("\n[%s] Stage %d/%d: %s  (%.1f s)\n      %s" % (status, i, len(STAGES), name, time.time() - t0, detail), flush=True)
        if status == "FAILED":
            code = EXIT[kind]
            for j, (n2, _, _) in enumerate(STAGES[i:], i + 1):
                results.append((j, n2, "NOT RUN", "", 0))
            break
    print("\n" + "=" * 78 + "\nFINAL REPORT (%.0f s)\n" % (time.time() - t_all) + "=" * 78)
    for i, name, status, _, dt in results:
        print("  %2d. %-9s %s" % (i, status, name))
    ok = code == 0
    if ok:
        print("\nRESULT: SUCCESS - MAPLE API is running and healthy on %s:%d under heartbeat supervision." % (API_HOST, a.api_port))
        print("Container hostname: %s (this is NOT the Cloudflare SSH host name; take that from your existing SSH cell output)" % c.notes.get("hostname"))
        print("NEXT STEP: READY FOR AUTHENTICATION SETUP (guide section 7.3). The authentication proxy was NOT started.")
    else:
        print("\nRESULT: FAILED (exit code %d). Fix the item in the FAILED stage above and run the cell again; it is safe to repeat." % code)
    try:
        (c.state / "oneclick_last_run.json").write_text(json.dumps({"version": __version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hostname": c.notes.get("hostname"),
                                                                 "ok": ok, "exit": code, "check_only": a.check_only, "stages": [{"n": i, "name": n, "status": s} for i, n, s, _, _ in results]}, indent=1))
    except OSError:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
