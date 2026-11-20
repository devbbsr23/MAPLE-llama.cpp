#!/usr/bin/env python3
"""Rapid recovery: take the LOCAL recovery directory to a working MAPLE inference server on a (new) Kaggle session.

Run on Windows after `ssh kaggle` works:

    python scripts/push_to_kaggle.py --start

Steps (each is reported; the first failure stops the run with a clear message):
 1. verify the local artifacts against checksums.sha256
 2. read-only remote preflight: free disk, GPU device nodes, the four EXTERNAL model paths
 3. upload archives + bootstrap scripts to --incoming (resumable; SHA256 verified on the remote side)
 4. run restore_kaggle.py on the server (no compilation, no model download, no CUDA install)
 5. optionally start the API in the background and run smoke_api.py against all four specialists

Models are never uploaded, copied or modified. If they are missing, the run stops after step 3 and says exactly
which paths are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["MAPLE-llamacpp-cuda.tar.gz", "MAPLE-runtime.tar.gz", "MAPLE-python-deps.tar.gz", "checksums.sha256"]
BOOTSTRAP = ["restore_kaggle.py", "link_models.py", "smoke_api.py"]


def ssh_env():
    env = dict(os.environ)
    cf = r"C:\cloudflared"
    if os.path.isdir(cf) and cf.lower() not in env.get("PATH", "").lower():
        env["PATH"] = cf + os.pathsep + env.get("PATH", "")
    return env


def ssh_cmd(host, remote):
    return ["ssh", "-o", "UpdateHostKeys=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=20", host, remote]


def rsh(host, remote, timeout=600, stream=False):
    if stream:
        p = subprocess.Popen(ssh_cmd(host, remote), env=ssh_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print("   | " + line.rstrip())
        return p.wait(), ""
    r = subprocess.run(ssh_cmd(host, remote), capture_output=True, text=True, timeout=timeout, env=ssh_env())
    return r.returncode, r.stdout + r.stderr


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def step(n, ok, msg):
    print("[%s] %s - %s" % ("ok" if ok else "FAIL", n, msg))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--host", default="kaggle")
    ap.add_argument("--incoming", default="/kaggle/working/maple_incoming", help="remote directory that receives the archives")
    ap.add_argument("--dest", default="/kaggle/working/MAPLE-llama.cpp", help="remote MAPLE project directory to restore into")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--start", action="store_true", help="after a successful restore: start the API and run smoke_api.py")
    ap.add_argument("--skip-restore", action="store_true", help="upload only")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)

    # 1. local verification
    try:
        want = {l.split()[1].lstrip("*"): l.split()[0] for l in (art / "checksums.sha256").read_text().splitlines() if len(l.split()) == 2}
        bad = [n for n in PARTS[:3] if sha256(art / n) != want.get(n)]
    except OSError as e:
        return 2 if not step("local artifacts", False, "cannot read %s (%s). Run scripts/pull_artifacts.py first." % (art, e)) else 0
    if not step("local artifacts", not bad, "checksums match" if not bad else "MISMATCH %s - refusing to upload" % bad):
        return 2

    # 2. preflight (read-only)
    models = runpy.run_path(str(ROOT / "src" / "maple" / "config.py"))["MODEL_ROSTER"]
    probe = "df -B1 --output=avail /kaggle/working | tail -1; ls /dev/nvidia[0-9]* 2>/dev/null | wc -l; " + \
            "; ".join("test -f '%s' && echo HAVE %s || echo MISSING %s '%s'" % (m["model_path"], k, k, m["model_path"]) for k, m in models.items())
    rc, out = rsh(a.host, probe)
    if rc != 0 and not out:
        return 2 if not step("preflight", False, "server unreachable (rc=%d). Check the tunnel/hostname." % rc) else 0
    lines = out.strip().splitlines()
    free, gpus = int(lines[0]), int(lines[1])
    missing = [l for l in lines[2:] if l.startswith("MISSING")]
    need = sum((art / n).stat().st_size for n in PARTS) * 3
    if not step("remote disk", free > need, "%.2f GiB free, need about %.2f GiB (archives + extraction)" % (free / 2**30, need / 2**30)):
        return 6
    step("GPU device nodes", gpus > 0, "%d /dev/nvidia* device(s)%s" % (gpus, "" if gpus else " - this is not a GPU session; enable GPU T4 x2 in Kaggle settings"))
    if missing:
        step("external models", False, "%d of %d missing; first: %s" % (len(missing), len(models), missing[0]))
    else:
        step("external models", True, "all %d model files present at their exact roster paths (not touched)" % len(models))

    # 3. upload (resumable, verified on the remote side)
    rsh(a.host, "mkdir -p '%s'" % a.incoming)
    sends = [(art / n, n) for n in PARTS] + [(ROOT / "scripts" / n, n) for n in BOOTSTRAP]
    for lp, n in sends:
        h = sha256(lp)
        rc, out = rsh(a.host, "sha256sum '%s/%s' 2>/dev/null | cut -c1-64" % (a.incoming, n))
        if out.strip() == h:
            print("   unchanged on server: %s" % n)
            continue
        p = subprocess.Popen(ssh_cmd(a.host, "cat > '%s/%s.partial'" % (a.incoming, n)), stdin=subprocess.PIPE, env=ssh_env())
        with open(lp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                p.stdin.write(chunk)
        p.stdin.close()
        p.wait()
        rc, out = rsh(a.host, "sha256sum '%s/%s.partial' | cut -c1-64" % (a.incoming, n))
        if out.strip() != h:
            rsh(a.host, "rm -f '%s/%s.partial'" % (a.incoming, n))      # only the partial; nothing else touched
            step("upload %s" % n, False, "checksum mismatch after transfer (partial removed); re-run to resume")
            return 3
        rsh(a.host, "mv '%s/%s.partial' '%s/%s'" % (a.incoming, n, a.incoming, n))
        print("   uploaded + verified: %s" % n)
    step("upload", True, "%d files in %s" % (len(sends), a.incoming))
    if a.skip_restore:
        return 0
    if missing:
        print("\nSTOP: model files are missing, so inference cannot start. Make them available at the exact paths, e.g.\n"
              "  python3 %s/link_models.py --source /kaggle/input --apply\n"
              "then re-run this script. Nothing was downloaded." % a.incoming)
        return 3

    # 4. restore
    print("\n== restore_kaggle.py (no compilation, no downloads) ==")
    rc, _ = rsh(a.host, "python3 -I '%s/restore_kaggle.py' --artifacts '%s' --dest '%s' --api-port %d" % (a.incoming, a.incoming, a.dest, a.api_port + 1), stream=True)
    if not step("restore", rc == 0, "rc=%d" % rc):
        return rc or 5

    # 5. start + smoke
    if a.start:
        cmd = ("cd '%s' && mkdir -p logs && (setsid nohup bash -c '. ./maple.env && exec scripts/start_maple.sh --port %d' > logs/api.log 2>&1 < /dev/null & echo $! > maple_api.pid); sleep 1; cat maple_api.pid"
               % (a.dest, a.api_port))
        rc, out = rsh(a.host, cmd)
        step("start API", rc == 0, "pid %s, log %s/logs/api.log" % (out.strip(), a.dest))
        up = False
        for _ in range(30):
            rc, out = rsh(a.host, "python3 -I -c \"import urllib.request as u;print(u.urlopen('http://127.0.0.1:%d/health',timeout=5).status)\"" % a.api_port)
            if out.strip() == "200":
                up = True
                break
            time.sleep(2)
        if not step("API health", up, "http://127.0.0.1:%d/health" % a.api_port):
            return 5
        print("\n== smoke_api.py (all four specialists; models load one at a time) ==")
        rc, _ = rsh(a.host, "python3 -I '%s/smoke_api.py' --base http://127.0.0.1:%d" % (a.incoming, a.api_port), stream=True)
        if not step("specialists", rc == 0, "rc=%d" % rc):
            return 5
    print("\nMAPLE recovery complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
