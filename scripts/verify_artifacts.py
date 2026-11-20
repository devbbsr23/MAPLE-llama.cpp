#!/usr/bin/env python3
"""Verify MAPLE artifacts: SHA256 vs checksums.sha256, gzip/tar integrity, and that
no member is a GGUF/model file. Exit 0 only if all checks pass.
A component is only marked verified in MANIFEST.json by --mark after these pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="/kaggle/working/artifacts")
    ap.add_argument("--allow-missing", action="append", default=[], metavar="NAME",
                    help="archive intentionally not stored here (e.g. the duplicate MAPLE-complete.tar.gz in the local recovery dir); reported as ABSENT, not failed")
    ap.add_argument("--mark", action="store_true", help="write verified=true into MANIFEST.json for passing artifacts")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    cs = d / "checksums.sha256"
    if not cs.is_file():
        print("FAIL: checksums.sha256 not found in", d)
        return 2
    results = {}
    for line in cs.read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?(\S+)$", line.strip())
        if not m:
            continue
        want, name = m.groups()
        p = d / name
        r = {"exists": p.is_file(), "checksum": False, "integrity": False, "no_gguf": False, "members": 0}
        if r["exists"]:
            r["checksum"] = sha(p) == want
            try:
                with tarfile.open(p, "r:gz") as t:
                    names = t.getnames()
                    r["members"] = len(names)
                    r["integrity"] = True
                    r["no_gguf"] = not any(n.endswith(".gguf") for n in names) and not any("/models/" in "/" + n for n in names)
            except (tarfile.TarError, OSError, EOFError) as e:
                r["error"] = str(e)[:200]
        r["pass"] = all(r[k] for k in ("exists", "checksum", "integrity", "no_gguf"))
        if not r["exists"] and name in a.allow_missing:
            print("%-32s ABSENT (allowed: not stored in this directory)" % name)
            continue
        results[name] = r
        print("%-32s %s  members=%d checksum=%s integrity=%s no_gguf=%s" % (name, "PASS" if r["pass"] else "FAIL", r["members"], r["checksum"], r["integrity"], r["no_gguf"]))
    if not any(r["exists"] for r in results.values()):
        print("FAIL: no entries in checksums.sha256")
        return 2
    if a.mark and (d / "MANIFEST.json").is_file():
        man = json.loads((d / "MANIFEST.json").read_text())
        for art in man["artifacts"]:
            r = results.get(art["name"])
            art["verified"] = bool(r and r["pass"])
            art["verified_note"] = "checksum + gzip/tar integrity + no-GGUF checked" if art["verified"] else "verification failed"
        (d / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    return 0 if all(r["pass"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
