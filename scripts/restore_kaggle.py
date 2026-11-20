#!/usr/bin/env python3
"""Restore MAPLE in a fresh Kaggle session from the TAR.GZ artifacts.

Never compiles llama.cpp, never downloads or touches model files, never installs
or replaces CUDA libraries, never extracts a *.gguf. Incompatible binaries are
reported with the exact reason; nothing is built or substituted.

Exit codes: 0 ok | 2 artifacts missing/corrupt | 3 model path missing |
            4 CUDA/binary incompatibility | 5 smoke/API test failed | 6 other
Usage: restore_kaggle.py [--artifacts DIR ...] [--dest DIR] [--no-sha] [--skip-smoke] [--skip-api]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

NAMES = {"llamacpp": "MAPLE-llamacpp-cuda.tar.gz", "runtime": "MAPLE-runtime.tar.gz",
         "deps": "MAPLE-python-deps.tar.gz", "complete": "MAPLE-complete.tar.gz"}
DEFAULT_SEARCH = ["/kaggle/input", "/kaggle/working/artifacts", "/kaggle/working", os.getcwd()]
DEST_DEFAULT = "/kaggle/working/MAPLE-llama.cpp"
SYSTEM_LIBS = ["libssl.so.3", "libcrypto.so.3", "libgomp.so.1", "libstdc++.so.6"]
CUDA_LIBS = ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "libnccl.so.2", "libcuda.so.1"]
GIB = 1024 ** 3
report = {"steps": [], "ok": False}


def say(step, ok, detail=""):
    report["steps"].append({"step": step, "ok": ok, "detail": detail})
    print("[%s] %s%s" % ("ok" if ok else "FAIL", step, (" - " + detail) if detail else ""))
    sys.stdout.flush()


def restore_deps(archive, dest):
    """Install the bundled wheels OFFLINE into <dest>/vendor/pydeps (never system site-packages). Non-fatal: the runtime is stdlib-only."""
    work = Path(dest) / "vendor" / "python-deps-bundle"
    target = Path(dest) / "vendor" / "pydeps"
    try:
        with tarfile.open(archive, "r:gz") as t:
            for m in t:
                if m.name.startswith("/") or ".." in Path(m.name).parts or m.name.lower().endswith(".gguf"):
                    raise RuntimeError("unsafe member %s" % m.name)
            work.mkdir(parents=True, exist_ok=True)
            t.extractall(work, filter="data") if sys.version_info >= (3, 12) else t.extractall(work)
        root = work / "MAPLE-python-deps"
        lock, wheels = root / "requirements.lock", root / "wheels"
        if not lock.is_file() or not any(wheels.glob("*.whl")):
            say("python dependencies", True, "bundle has no wheels; runtime uses the standard library only")
            return
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "--target", str(target),
                            "--no-deps", "--disable-pip-version-check", "-q", "-r", str(lock)], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            say("python dependencies", False, "offline wheel install failed (non-fatal, runtime is stdlib-only): %s" % (r.stderr.strip().splitlines() or ["?"])[-1][:200])
            return
        c = subprocess.run([sys.executable, "-s", "-c", "import yaml;print(yaml.__version__, yaml.__file__)"], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=str(target)), timeout=60)
        say("python dependencies", c.returncode == 0 and str(target) in c.stdout,
            "offline wheels installed into %s; import check: %s" % (target, c.stdout.strip() or c.stderr.strip()[-120:]))
    except Exception as e:  # noqa
        say("python dependencies", False, "%s: %s (non-fatal, runtime is stdlib-only)" % (type(e).__name__, e))


def die(code, msg):
    print("\nSTOP: " + msg, file=sys.stderr)
    report["stopped"] = msg
    save_report()
    sys.exit(code)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_report():
    try:
        d = Path(DEST) / "manifests"
        d.mkdir(parents=True, exist_ok=True)
        (d / "restore_report.json").write_text(json.dumps(report, indent=2, default=str))
    except OSError:
        pass


def free(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def run(cmd, env=None, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


# ---------------------------------------------------------------- stage: locate + verify
def locate(search, deep_roots=("/kaggle/input",), deep_depth=6, shallow_depth=2, max_dirs=3000):
    """Find the deployment archives. Breadth-first, shallowest first, so earlier behaviour is unchanged:
    for each search root the root itself is checked, then its children, then deeper levels.

    Kaggle mounts Datasets at differing depths (e.g. /kaggle/input/<slug>/ or /kaggle/input/datasets/<owner>/<slug>/),
    so roots under /kaggle/input and any folder given with --artifacts are searched up to ``deep_depth`` levels;
    other roots keep the old two-level limit. Symlinked directories are not followed. Only directory names are listed
    (never file contents); at most ``max_dirs`` directories are visited per root.
    """
    found, dupes = {}, {}
    for d in search:
        dp = Path(d)
        if not dp.is_dir():
            continue
        is_deep = str(dp).startswith(tuple(deep_roots)) or d in explicit_roots
        limit = deep_depth if is_deep else shallow_depth
        queue, seen = [(dp, 0)], 0
        while queue and seen < max_dirs:
            c, depth = queue.pop(0)
            seen += 1
            for k, n in NAMES.items():
                if (c / n).is_file():
                    if k not in found:
                        found[k] = c / n
                    elif found[k] != c / n:
                        dupes.setdefault(k, []).append(str(c / n))
            if depth < limit:
                try:
                    queue.extend((x, depth + 1) for x in sorted(c.iterdir()) if x.is_dir() and not x.is_symlink())
                except OSError:
                    pass
    for k, extra in dupes.items():
        print("note: %s also found at %s; using %s" % (NAMES[k], ", ".join(extra), found[k]))
    return found


explicit_roots = set()


def expected_checksums(found):
    exp = {}
    for p in {v.parent for v in found.values()}:
        cs = p / "checksums.sha256"
        if cs.is_file():
            for line in cs.read_text().splitlines():
                m = re.match(r"^([0-9a-f]{64})\s+\*?(\S+)$", line.strip())
                if m:
                    exp[m.group(2)] = m.group(1)
    return exp


def safe_members(t: tarfile.TarFile, label: str):
    for m in t.getmembers():
        n = m.name
        if n.startswith("/") or ".." in Path(n).parts:
            raise ValueError("%s: unsafe path %s" % (label, n))
        if n.endswith(".gguf") or "/models/" in "/" + n and n.split("/")[-1].endswith(".gguf"):
            raise ValueError("%s: refusing to extract GGUF member %s" % (label, n))
        if m.issym() or m.islnk():
            tgt = os.path.normpath(os.path.join(os.path.dirname(n), m.linkname))
            if os.path.isabs(m.linkname) or tgt.startswith(".."):
                raise ValueError("%s: link escapes archive: %s -> %s" % (label, n, m.linkname))
        yield m


def extract(archive, into, label, overwrite=False):
    """Extract with safety checks. Existing differing files are kept unless overwrite."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    kept, wrote = [], 0
    global MODES_FIXED
    with tarfile.open(archive, "r:gz") as t:
        members = list(safe_members(t, label))
        need = sum(m.size for m in members if m.isfile())
        if free(into) < need * 1.1:
            die(6, "not enough disk to extract %s: need %.2f GiB, free %.2f GiB" % (label, need * 1.1 / GIB, free(into) / GIB))
        for m in members:
            dst = into / m.name
            if dst.exists() and m.isfile() and not overwrite:
                f = t.extractfile(m)
                h = hashlib.sha256(f.read()).hexdigest()
                if sha256(dst) != h:
                    kept.append(m.name)
                elif (dst.stat().st_mode & 0o777) != (m.mode & 0o777):
                    # Kaggle session snapshots strip the execute bit; contents are identical, so only
                    # re-apply the mode recorded in the archive (never touches file contents).
                    os.chmod(dst, m.mode & 0o777)
                    MODES_FIXED += 1
                continue
            try:
                t.extract(m, into, filter="tar")   # keeps modes, rejects absolute/escaping paths
            except TypeError:               # Python < 3.9.17 has no filter argument (members are pre-validated)
                t.extract(m, into)
            wrote += 1
    return wrote, kept


DEST = DEST_DEFAULT
MODES_FIXED = 0


def main(argv=None):
    global DEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", action="append", help="directory to search (repeatable)")
    ap.add_argument("--dest", default=DEST_DEFAULT)
    ap.add_argument("--no-sha", action="store_true", help="skip model SHA256 verification (size + GGUF magic only)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-api", action="store_true")
    ap.add_argument("--overwrite-source", action="store_true", help="overwrite differing project files (default: keep and report)")
    ap.add_argument("--smoke-model", default="fast")
    ap.add_argument("--api-port", type=int, default=8000)
    a = ap.parse_args(argv)
    DEST = a.dest
    dest = Path(DEST)
    report.update(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), python=sys.version.split()[0], platform=platform.platform())

    # 1. environment
    on_kaggle = Path("/kaggle").is_dir()
    say("detect environment", True, "kaggle=%s python=%s fresh=%s" % (on_kaggle, sys.version.split()[0], not (dest / "src" / "maple").is_dir()))
    if sys.version_info < (3, 9):
        die(6, "Python >= 3.9 required")

    # 2. locate + verify archives
    if a.artifacts:
        explicit_roots.update(a.artifacts)
    found = locate(a.artifacts or DEFAULT_SEARCH)
    report["archives_found"] = {k: str(v) for k, v in found.items()}
    have_parts = all(k in found for k in ("llamacpp", "runtime"))
    if not have_parts and "complete" not in found:
        die(2, "no MAPLE archives found in %s. Attach/upload MAPLE-*.tar.gz (+ checksums.sha256)." % (a.artifacts or DEFAULT_SEARCH))
    exp = expected_checksums(found)
    for k, p in found.items():
        want = exp.get(p.name)
        got = sha256(p)
        if want is None:
            say("checksum %s" % p.name, False, "no recorded checksum (checksums.sha256 missing) - computed %s" % got)
            die(2, "cannot verify %s: checksums.sha256 not found next to it" % p.name)
        if got != want:
            die(2, "CHECKSUM MISMATCH for %s: expected %s got %s. Archive is corrupt or modified; not extracting." % (p, want, got))
        say("checksum %s" % p.name, True, got[:16] + "...")
        with tarfile.open(p, "r:gz") as t:
            for _ in safe_members(t, p.name):
                pass
        say("integrity+safety %s" % p.name, True, "no GGUF, no unsafe paths")

    # 3. extract (complete -> staging -> parts)
    stage = None
    parts = dict(found)
    if not have_parts and "complete" in found:
        stage = Path(tempfile.mkdtemp(prefix="maple_stage_", dir=str(dest.parent if dest.parent.exists() else "/tmp")))
        with tarfile.open(found["complete"], "r:gz") as t:
            mem = [m for m in safe_members(t, "complete") if m.isfile() and m.name.endswith(".tar.gz")]
            if free(stage) < sum(m.size for m in mem) * 1.1:
                die(6, "not enough disk to unpack MAPLE-complete.tar.gz")
            for m in mem:
                try:
                    t.extract(m, stage, filter="tar")
                except TypeError:
                    t.extract(m, stage)
        for k, n in NAMES.items():
            for p in stage.rglob(n):
                parts.setdefault(k, p)
    wrote, kept = extract(parts["runtime"], dest.parent, "runtime", overwrite=a.overwrite_source)
    say("restore MAPLE application", True, "%d files written; %d differing files kept (use --overwrite-source): %s" % (wrote, len(kept), kept[:5]))
    report["kept_differing_files"] = kept
    bin_dir = dest / "vendor" / "llamacpp" / "llama.cpp" / "build" / "bin"
    # Always run the extractor: it writes missing files and re-applies archive modes on identical ones
    # (an existing build whose execute bits were stripped by a session snapshot is repaired, not recompiled).
    before = MODES_FIXED
    w, kept_l = extract(parts["llamacpp"], dest / "vendor" / "llamacpp", "llamacpp")
    if kept_l:
        say("restore llama.cpp", False, "existing files differ from the archive (kept, not overwritten): %s" % kept_l[:5])
        die(4, "vendored llama.cpp differs from MAPLE-llamacpp-cuda.tar.gz; refusing to mix builds. Remove %s manually or use --overwrite-source." % (dest / "vendor"))
    exe_ok = (bin_dir / "llama-server").is_file() and os.access(bin_dir / "llama-server", os.X_OK)
    say("restore llama.cpp", exe_ok, "%d files written, %d permission modes re-applied from the archive; no compilation" % (w, MODES_FIXED - before))
    if not exe_ok:
        die(4, "llama-server missing or not executable at %s" % bin_dir)
    if "deps" in parts:
        restore_deps(parts["deps"], dest)
    if stage:
        shutil.rmtree(stage, ignore_errors=True)

    # 4. environment file + sys.path
    sys.path.insert(0, str(dest / "src"))
    os.environ["MAPLE_HOME"] = str(dest)
    from maple.config import MODEL_ROSTER, build_env, load_runtime_config  # noqa: E402
    cfg = load_runtime_config()
    env = build_env(cfg)
    (dest / "maple.env").write_text("# generated by restore_kaggle.py (no secrets)\nexport MAPLE_HOME=%s\nexport MAPLE_LLAMA_BIN=%s\nexport LD_LIBRARY_PATH=%s\n" % (dest, bin_dir, env["LD_LIBRARY_PATH"]))
    say("write maple.env", True, "LD_LIBRARY_PATH set per-process only; system environment untouched")

    # 5. dependency + CUDA compatibility
    problems = []
    rc, out = run(["ldd", "--version"])
    m = re.search(r"(\d+)\.(\d+)\s*$", out.splitlines()[0]) if out else None
    glibc = (int(m.group(1)), int(m.group(2))) if m else None
    if glibc and glibc < (2, 38):
        problems.append("glibc %d.%d < 2.38 required by the compiled libraries" % glibc)
    for exe in ("llama-server", "llama-cli", "llama-quantize"):
        if not (bin_dir / exe).is_file():
            problems.append("missing executable %s in %s" % (exe, bin_dir))
    rc, out = run(["ldd", str(bin_dir / "libggml-cuda.so.1")], env=env)
    missing = sorted(set(re.findall(r"(\S+)\s+=>\s+not found", out)))
    if missing:
        problems.append("unresolved shared libraries: %s" % ", ".join(missing))
    rc2, out2 = run(["ldd", str(bin_dir / "llama-server")], env=env)
    missing2 = sorted(set(re.findall(r"(\S+)\s+=>\s+not found", out2)) - set(missing))
    if missing2:
        problems.append("unresolved shared libraries (llama-server): %s" % ", ".join(missing2))
    cuda_state = {lib: bool(re.search(re.escape(lib) + r"\s+=>\s+/", out)) for lib in CUDA_LIBS if lib != "libcuda.so.1"}
    rcv, outv = run([str(bin_dir / "llama-server"), "--version"], env=env)
    verline = next((l for l in outv.splitlines() if "version" in l.lower()), (outv.strip().splitlines() or ["(no output)"])[-1])
    say("llama-server --version", rcv == 0, verline.strip()[:160])
    if rcv != 0:
        problems.append("llama-server --version failed: %s" % outv.strip()[-200:])
    rcd, outd = run([str(bin_dir / "llama-server"), "--list-devices"], env=env)
    devs = re.findall(r"(CUDA\d+): (.+?) \((\d+) MiB", outd)
    report["cuda"] = {"glibc": glibc, "libs": cuda_state, "devices": devs, "list_devices_output": outd[-500:]}
    if not devs:
        problems.append("llama.cpp CUDA backend sees no GPU: %s" % outd.strip().splitlines()[-1][:200] if outd.strip() else "no output")
    else:
        bad = [d for d in devs if "T4" not in d[1] and not re.search(r"RTX 20|Quadro RTX|TITAN RTX|A100|A10|L4|RTX 30|RTX 40|H100", d[1])]
        if bad:
            problems.append("GPU(s) %s may not be covered by the compiled CUDA arch (sm_75 only: Turing + PTX). Verify before use." % bad)
    if problems:
        say("CUDA/dependency compatibility", False, "; ".join(problems))
        print("\nThe restored binaries are NOT usable here for the reason(s) above.\nNothing was installed, replaced or compiled. MAPLE never builds llama.cpp.\n"
              "Use a host that provides the CUDA 13 runtime libraries and NCCL listed above (see docs/HARDWARE_REQUIREMENTS.md).", file=sys.stderr)
        report["compatible"] = False
        die(4, "incompatible environment: " + "; ".join(problems))
    report["compatible"] = True
    say("CUDA/dependency compatibility", True, "%d GPU(s): %s; glibc %s" % (len(devs), [d[1] for d in devs], glibc))

    # 6. models (external; never touched)
    from maple.registry import Registry
    reg = Registry()
    missing_models = []
    for mid, m in MODEL_ROSTER.items():
        c = reg.check_path(mid)
        if not c["exists"]:
            missing_models.append((mid, m["model_path"]))
            continue
        rec = reg.recorded(mid)
        shaok = None
        if rec.get("sha256") and not a.no_sha:
            shaok = sha256(m["model_path"]) == rec["sha256"]
        good = c["status"] == "ok" and shaok is not False
        say("model %s" % mid, good, "%s size=%s sha256=%s" % (m["model_path"], c["size"], {True: "match", False: "MISMATCH", None: "not checked"}[shaok]))
        if not good:
            missing_models.append((mid, m["model_path"] + " (invalid or checksum mismatch)"))
    if missing_models:
        for mid, p in missing_models:
            print("MISSING/INVALID model '%s': %s" % (mid, p), file=sys.stderr)
        die(3, "model files unavailable; inference startup stopped. Models are never downloaded or restored by MAPLE - attach them at the exact paths above.")

    # 7. smoke test (real inference through ModelManager)
    if not a.skip_smoke:
        from maple.runtime import ModelManager
        mgr = ModelManager(reg, cfg)
        try:
            with mgr.use(a.smoke_model, 2048) as base:
                req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Reply with the word OK."}], "max_tokens": 16}).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    txt = json.loads(r.read())["choices"][0]["message"]
            say("llama.cpp smoke test (%s)" % a.smoke_model, True, "loaded in %ss; reply=%r" % (mgr.last_load.get("load_time_s"), (txt.get("content") or txt.get("reasoning_content") or "")[:40]))
        except Exception as e:  # noqa
            say("llama.cpp smoke test", False, str(e)[-400:])
            mgr.unload()
            die(5, "smoke test failed: %s" % str(e)[-300:])
        finally:
            mgr.unload()

    # 8. API health
    if not a.skip_api:
        api_env = dict(env, MAPLE_HOME=str(dest), PYTHONPATH=str(dest / "src"))
        p = subprocess.Popen([sys.executable, "-s", "-m", "maple", "serve", "--port", str(a.api_port)], env={**api_env, "PYTHONPATH": str(dest / "src")}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        ok, detail = False, ""
        try:
            for _ in range(60):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen("http://127.0.0.1:%d/health" % a.api_port, timeout=3) as r:
                        h = json.loads(r.read())
                        ok, detail = h.get("status") == "ok", json.dumps(h.get("models"))
                        break
                except Exception as e:  # noqa
                    detail = str(e)[:100]
        finally:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        say("MAPLE API health", ok, detail)
        if not ok:
            die(5, "API health check failed: " + detail)
    report["ok"] = True
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_report()
    print("\nMAPLE restored. Start with:  source %s/maple.env && %s/scripts/start_maple.sh" % (dest, dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
