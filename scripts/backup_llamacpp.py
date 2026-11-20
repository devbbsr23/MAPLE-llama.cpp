#!/usr/bin/env python3
"""Download the ORIGINAL compiled llama.cpp CUDA archive from Kaggle to Windows storage (run on Windows).

    python scripts/backup_llamacpp.py                      # -> <parent of this repo>/backups/llamacpp/
    python scripts/backup_llamacpp.py --verify-only        # re-verify the local copy (and the remote hash if reachable)

Guarantees
  * read-only on Kaggle (sha256sum, stat, and `tail -c` reads only); the original is never modified or deleted
  * original filename preserved; the final file appears only after its SHA256 equals the Kaggle source's SHA256
  * resumable: an interrupted transfer leaves <name>.partial; the next run continues from its size (tail -c +N),
    and the whole file is hashed at the end, so a corrupt resume can never be accepted
  * an existing final file is never overwritten unless it is verified identical (then it is simply kept);
    a differing existing file is reported and left alone
  * refuses anything that looks like a model (*.gguf / /models/)
  * records size, SHA256, source path, destination and timestamps in LLAMACPP_BACKUP_MANIFEST.json (kept next to the archive,
    separate from the MAPLE source backup)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SRC = "/kaggle/working/downloaded-llama-cpp-full-cuda-t4.tar.gz"
DEST = str(Path(__file__).resolve().parents[2] / "backups" / "llamacpp")   # sibling of the repo folder; override with --dest
MANIFEST = "LLAMACPP_BACKUP_MANIFEST.json"


def ssh_env():
    env = dict(os.environ)
    cf = r"C:\cloudflared"
    if os.path.isdir(cf) and cf.lower() not in env.get("PATH", "").lower():
        env["PATH"] = cf + os.pathsep + env.get("PATH", "")      # this process tree only
    return env


def ssh_cmd(host, remote):
    return ["ssh", "-o", "UpdateHostKeys=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
            "-o", "ServerAliveInterval=20", host, remote]


def sha256(p, upto=None):
    h, n = hashlib.sha256(), 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def integrity(p):
    """Full gzip CRC read + tar header walk. Returns (members, gguf_members)."""
    n = g = 0
    with tarfile.open(p, "r:gz") as t:
        for m in t:
            n += 1
            g += m.name.lower().endswith(".gguf")
            if m.isfile():
                f = t.extractfile(m)
                while f.read(1 << 20):
                    pass
    return n, g


def remote_facts(host, src):
    r = subprocess.run(ssh_cmd(host, "sha256sum -- '%s' && stat -c '%%s' -- '%s'" % (src, src)),
                       capture_output=True, text=True, env=ssh_env(), timeout=300)
    if r.returncode != 0:
        return None, None, r.stderr[-300:]
    lines = r.stdout.split()
    m = re.fullmatch(r"[0-9a-f]{64}", lines[0]) if lines else None
    if not m or len(lines) < 3:
        return None, None, "unexpected output: %r" % r.stdout[:200]
    return lines[0], int(lines[-1]), ""


def write_manifest(d, rec):
    mp = d / MANIFEST
    tmp = d / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, mp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="kaggle")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args(argv)
    name = os.path.basename(a.src)
    if name.lower().endswith(".gguf") or "/models/" in a.src:
        print("REFUSED: looks like a model file")
        return 2
    d = Path(a.dest)
    d.mkdir(parents=True, exist_ok=True)
    final, part = d / name, d / (name + ".partial")

    rsha, rsize, err = remote_facts(a.host, a.src)
    if rsha is None:
        print("Kaggle source unreachable (%s)" % err.strip())
        if not a.verify_only:
            print("nothing changed locally; %s kept for resume" % (part.name if part.exists() else "no partial"))
            return 2
        mp = d / MANIFEST
        if not (final.is_file() and mp.exists()):
            return 2
        want = json.loads(mp.read_text())["sha256"]
        print("verifying against the SHA256 recorded in the manifest (remote not reachable)")
    else:
        want = rsha
        print("remote: %s  %d bytes  sha256 %s" % (a.src, rsize, rsha))

    status = "verified"
    if final.is_file():
        got = sha256(final)
        if got == want:
            print("local file already present and identical - kept, not rewritten")
            status = "already-present"
        else:
            print("CONFLICT: %s exists locally with a different SHA256 (%s). Left untouched; nothing downloaded." % (final, got))
            return 4
    elif a.verify_only:
        print("no local archive to verify")
        return 2
    else:
        free = shutil.disk_usage(d).free
        have = part.stat().st_size if part.exists() else 0
        need = rsize - have
        print("local free %.1f GiB, still to transfer %.1f MiB (resuming from byte %d)" % (free / 2**30, need / 2**20, have))
        if free < need * 1.1 + 100 * 2**20:
            print("not enough local disk space; nothing transferred")
            return 6
        if have > rsize:
            print("partial is larger than the source; discarding the partial only")
            part.unlink()
            have = 0
        t0 = time.time()
        if have < rsize:
            p = subprocess.Popen(ssh_cmd(a.host, "tail -c +%d -- '%s'" % (have + 1, a.src)), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=ssh_env())
            with open(part, "ab") as f:
                for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            rc = p.wait()
            if rc != 0:
                print("transfer interrupted (rc=%d); partial kept at %d bytes, re-run to resume. %s" % (
                    rc, part.stat().st_size, p.stderr.read()[-200:].decode(errors="replace")))
                return 3
        size = part.stat().st_size
        got = sha256(part)
        if size != rsize or got != want:
            print("VERIFY FAILED: size %d/%d sha_ok=%s - the partial is not trustworthy and was discarded; re-run" % (size, rsize, got == want))
            part.unlink(missing_ok=True)
            return 5
        os.replace(part, final)
        print("downloaded %d bytes in %.0fs; sha256 matches the Kaggle source" % (size, time.time() - t0))
        status = "downloaded"

    try:
        members, gguf = integrity(final)
    except (tarfile.TarError, OSError, EOFError, gzip.BadGzipFile) as e:
        print("INTEGRITY FAILED: %s" % e)
        return 5
    print("integrity: gzip CRC + %d tar members readable; GGUF members in archive: %d (informational: vocab test files)" % (members, gguf))

    # re-verify the Kaggle original is unchanged after our read
    rsha2, _, _ = remote_facts(a.host, a.src) if rsha is not None else (None, None, "")
    unchanged = (rsha2 == rsha) if rsha is not None else None
    rec = {
        "schema": 1,
        "file": name,
        "source": "%s:%s" % (a.host, a.src),
        "local_path": str(final),
        "size_bytes": final.stat().st_size,
        "sha256": want,
        "sha256_matches_kaggle_source": (got == rsha) if rsha is not None else None,
        "kaggle_original_unchanged_after_transfer": unchanged,
        "tar_members": members,
        "gguf_members_in_archive": gguf,
        "integrity": "gzip CRC and tar structure fully read OK",
        "status": status,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": "Original archive, byte-identical to the Kaggle source. No models transferred. Kaggle original not modified or deleted.",
    }
    write_manifest(d, rec)
    ok = rec["sha256_matches_kaggle_source"] is not False and unchanged is not False
    print("manifest: %s\nLOCAL LLAMA.CPP BACKUP: %s" % (d / MANIFEST, "VERIFIED" if ok else "NOT VERIFIED"))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
