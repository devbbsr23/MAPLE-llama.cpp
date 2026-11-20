#!/usr/bin/env python3
"""Pull the small deployment artifacts from Kaggle into the local recovery directory (run on Windows).

Streams over `ssh`, verifies SHA256 against the remote copy, is resumable (files already identical are
skipped), never deletes anything local, and refuses anything that looks like a model (*.gguf / models/).
Records the result in LOCAL_BACKUP_MANIFEST.json ("artifacts") and SYNC_STATUS.md.

    python scripts/pull_artifacts.py                       # llama.cpp + runtime + python-deps + manifests
    python scripts/pull_artifacts.py --include-complete    # also the (duplicate) MAPLE-complete.tar.gz
    python scripts/pull_artifacts.py --include-original    # also downloaded-llama-cpp-full-cuda-t4.tar.gz (235 MB)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["MAPLE-llamacpp-cuda.tar.gz", "MAPLE-runtime.tar.gz", "MAPLE-python-deps.tar.gz"]
SMALL = ["MANIFEST.json", "checksums.sha256", "packaging-report.txt"]
ORIGINAL = "/kaggle/working/downloaded-llama-cpp-full-cuda-t4.tar.gz"
BEGIN, END = "<!-- ARTIFACTS:BEGIN -->", "<!-- ARTIFACTS:END -->"


def ssh_env():
    env = dict(os.environ)
    cf = r"C:\cloudflared"
    if os.path.isdir(cf) and cf.lower() not in env.get("PATH", "").lower():
        env["PATH"] = cf + os.pathsep + env.get("PATH", "")      # session-only, this process tree
    return env


def ssh_cmd(host, remote):
    return ["ssh", "-o", "UpdateHostKeys=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
            "-o", "ServerAliveInterval=20", host, remote]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def block(art, not_stored):
    L = [BEGIN, "", "## Recovery artifacts (stored locally; GGUF models are NEVER stored)", ""]
    for n, m in sorted(art.items()):
        L.append("- `%s` %d bytes sha256 `%s` (pulled %s)" % (n, m["size"], m["sha256"], m["pulled_utc"]))
    for n, m in sorted(not_stored.items()):
        L.append("- not stored: `%s` sha256 `%s` - %s" % (n, m.get("sha256"), m["reason"]))
    return "\n".join(L + ["", END, ""])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT))
    ap.add_argument("--host", default="kaggle")
    ap.add_argument("--remote-dir", default="/kaggle/working/artifacts")
    ap.add_argument("--include-complete", action="store_true")
    ap.add_argument("--include-original", action="store_true")
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    adir = dest / "artifacts"
    adir.mkdir(parents=True, exist_ok=True)
    wanted = [(n, "%s/%s" % (a.remote_dir, n)) for n in PARTS + SMALL]
    if a.include_complete:
        wanted.append(("MAPLE-complete.tar.gz", "%s/MAPLE-complete.tar.gz" % a.remote_dir))
    if a.include_original:
        wanted.append((os.path.basename(ORIGINAL), ORIGINAL))
    for n, rp in wanted:
        if n.lower().endswith(".gguf") or "/models/" in rp:
            print("REFUSED: %s looks like a model file" % n)
            return 2

    q = " ".join("'%s'" % rp for _, rp in wanted)
    r = subprocess.run(ssh_cmd(a.host, "sha256sum %s && stat -c '%%s %%n' %s" % (q, q)), capture_output=True, text=True, env=ssh_env(), timeout=300)
    if r.returncode != 0:
        print("remote unreachable or file missing (local copy untouched): rc=%d %s" % (r.returncode, r.stderr[-300:]))
        return 2
    rsha, rsize = {}, {}
    for line in r.stdout.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(\S.*)$", line)
        if m:
            rsha[m.group(2).strip()] = m.group(1)
            continue
        m = re.match(r"^(\d+)\s+(\S.*)$", line)
        if m:
            rsize[m.group(2).strip()] = int(m.group(1))

    need = sum(rsize[rp] for n, rp in wanted if not (adir / n).is_file() or sha256(adir / n) != rsha[rp])
    free = shutil.disk_usage(dest).free
    print("to download: %.1f MiB; local free: %.1f GiB" % (need / 2**20, free / 2**30))
    if free < need * 1.1 + 50 * 2**20:
        print("not enough local disk space; nothing copied")
        return 6

    got, pulled = {}, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for n, rp in wanted:
        lp = adir / n
        if lp.is_file() and sha256(lp) == rsha[rp]:
            print("unchanged  %s" % n)
            got[n] = {"sha256": rsha[rp], "size": rsize[rp], "pulled_utc": pulled, "source": "%s:%s" % (a.host, rp)}
            continue
        part = adir / (n + ".partial")
        errf = tempfile.TemporaryFile()
        p = subprocess.Popen(ssh_cmd(a.host, "cat -- '%s'" % rp), stdout=subprocess.PIPE, stderr=errf, env=ssh_env())
        h, size = hashlib.sha256(), 0
        with open(part, "wb") as f:
            for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
                f.write(chunk)
                h.update(chunk)
                size += len(chunk)
        rc = p.wait()
        if rc != 0 or h.hexdigest() != rsha[rp] or size != rsize[rp]:
            errf.seek(0)
            print("FAILED %s: rc=%d size=%d/%d sha_ok=%s %s" % (n, rc, size, rsize[rp], h.hexdigest() == rsha[rp], errf.read()[-200:]))
            part.unlink(missing_ok=True)       # partial only; the previous good local file is untouched
            return 3
        os.replace(part, lp)
        print("downloaded %s (%d bytes, sha256 verified against the remote copy)" % (n, size))
        got[n] = {"sha256": rsha[rp], "size": size, "pulled_utc": pulled, "source": "%s:%s" % (a.host, rp)}

    not_stored = {}
    if not a.include_complete:
        cs = (adir / "checksums.sha256").read_text().splitlines()
        h = next((l.split()[0] for l in cs if "MAPLE-complete.tar.gz" in l), None)
        not_stored["MAPLE-complete.tar.gz"] = {"sha256": h, "reason": "duplicate wrapper of the three stored archives; rebuild with package_artifacts.py complete or use --include-complete"}
    if not a.include_original:
        not_stored[os.path.basename(ORIGINAL)] = {"sha256": rsha.get(ORIGINAL) or "541483fc880595fe20b02f04a0280eb0024d57c9bd4ac2082e7b04fd619eb5f0",
                                                  "reason": "the owner's shared source distribution (kept elsewhere); MAPLE-llamacpp-cuda.tar.gz is the stripped copy of its build/bin"}
    mp, sp = dest / "LOCAL_BACKUP_MANIFEST.json", dest / "SYNC_STATUS.md"
    man = json.loads(mp.read_text()) if mp.exists() else {"schema": 1, "project": "MAPLE-llama.cpp", "files": {}}
    man["artifacts"], man["artifacts_not_stored"], man["artifacts_last_pull_utc"] = got, not_stored, pulled
    tmp = mp.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=2))
    os.replace(tmp, mp)
    st = sp.read_text() if sp.exists() else "# MAPLE-llama.cpp local backup: sync status\n"
    st = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", st, flags=re.S).rstrip() + "\n\n" + block(got, not_stored)
    sp.write_text(st)
    print("artifacts recorded in LOCAL_BACKUP_MANIFEST.json; stored: %d files, %.1f MiB" % (len(got), sum(m["size"] for m in got.values()) / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
