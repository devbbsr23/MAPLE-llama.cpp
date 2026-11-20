#!/usr/bin/env python3
"""Pull MAPLE project SOURCE files from the Kaggle host to a local backup directory.

Run on Windows (or any machine with `ssh kaggle` configured):
    python sync_from_kaggle.py --dest <LOCAL_PROJECT_DIR>

Guarantees
  * Only small project source/config/doc/manifest files are copied (never archives,
    binaries, shared libraries, GGUF/model files, caches, logs, vendor/ build dirs).
  * Files are fetched into a staging dir, SHA256-verified against the remote listing,
    and only then moved into place -> an interrupted run leaves the last good copy.
  * Never deletes a local file. Never overwrites a local file that changed since the
    last sync, or an unrelated pre-existing local file: those are reported as CONFLICT.
  * Resumable: re-running copies only new/changed files.
  * Writes LOCAL_BACKUP_MANIFEST.json and SYNC_STATUS.md in the destination.
Read-only on the remote host.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REMOTE_ROOT = "/kaggle/working/MAPLE-llama.cpp"
MAX_FILE = 5 * 1024 * 1024
EXCLUDE_DIRS = {"vendor", "logs", "__pycache__", ".git", ".cache", "models", "build", "node_modules", ".ipynb_checkpoints"}
EXCLUDE_SUFFIX = (".tar.gz", ".tgz", ".zip", ".zst", ".gguf", ".bin", ".so", ".o", ".a", ".pyc", ".partial", ".whl", ".safetensors", ".pt", ".log")
EXCLUDE_NAMES = {"benchmark_results.json.big", "maple.env", "maple_api.pid", "restore_report.json"}   # generated per session
# explicitly allowed even though a suffix rule could catch them
ALWAYS = {"checksums.sha256"}

REMOTE_LIST = r'''
import hashlib, json, os, sys
root, maxf = sys.argv[1], int(sys.argv[2])
exd = set(json.loads(sys.argv[3])); exs = tuple(json.loads(sys.argv[4])); always = set(json.loads(sys.argv[5]))
out, skipped = {}, []
for d, dirs, files in os.walk(root):
    dirs[:] = sorted(x for x in dirs if x not in exd)
    for f in sorted(files):
        p = os.path.join(d, f); rel = os.path.relpath(p, root).replace(os.sep, "/")
        if os.path.islink(p): continue
        if f not in always and f.endswith(exs): skipped.append([rel, "excluded-suffix"]); continue
        sz = os.path.getsize(p)
        if sz > maxf: skipped.append([rel, "too-large:%d" % sz]); continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
        out[rel] = {"size": sz, "sha256": h.hexdigest()}
print(json.dumps({"files": out, "skipped": skipped}))
'''


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ssh_env():
    env = dict(os.environ)
    cf = r"C:\cloudflared"
    if os.path.isdir(cf) and cf.lower() not in env.get("PATH", "").lower():
        env["PATH"] = cf + os.pathsep + env.get("PATH", "")  # session-only, this process tree
    return env


def ssh(host, cmd, stdin=None, timeout=300):
    return subprocess.run(["ssh", "-o", "UpdateHostKeys=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", host, cmd],
                          input=stdin, capture_output=True, timeout=timeout, env=ssh_env())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--host", default="kaggle")
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    mpath, spath = dest / "LOCAL_BACKUP_MANIFEST.json", dest / "SYNC_STATUS.md"
    prev = json.loads(mpath.read_text()) if mpath.exists() else {"files": {}}
    pf = prev.get("files", {})
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = {"started_utc": now, "result": "unknown", "copied": [], "unchanged": 0, "conflicts": [], "local_newer": [], "skipped_remote": [], "error": None}

    def finish(code):
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        files = {}
        for rel, meta in pf.items():
            lp = dest / rel
            if lp.is_file():
                files[rel] = dict(meta, local_size=lp.stat().st_size, local_sha256=sha256(lp))
        man = {"schema": 1, "project": "MAPLE-llama.cpp", "source": "%s:%s" % (a.host, a.remote_root),
               "last_sync_utc": status["finished_utc"], "last_sync_result": status["result"], "file_count": len(files),
               "excluded": {"dirs": sorted(EXCLUDE_DIRS), "suffixes": list(EXCLUDE_SUFFIX), "max_file_bytes": MAX_FILE},
               "files": files}
        for k in ("artifacts", "artifacts_not_stored", "artifacts_last_pull_utc"):   # written by pull_artifacts.py; keep them
            if k in prev:
                man[k] = prev[k]
        if not a.dry_run:
            mpath.write_text(json.dumps(man, indent=2))
            L = ["# MAPLE-llama.cpp local backup: sync status", "",
                 "- Last sync (UTC): %s" % status["finished_utc"], "- Result: **%s**" % status["result"],
                 "- Source: `%s:%s`" % (a.host, a.remote_root), "- Files tracked: %d" % len(files),
                 "- Copied this run: %d | unchanged: %d | conflicts: %d | local-newer (kept): %d" % (
                     len(status["copied"]), status["unchanged"], len(status["conflicts"]), len(status["local_newer"]))]
            if status["error"]:
                L += ["- Error: `%s`" % status["error"]]
            for title, key in (("Copied", "copied"), ("CONFLICTS (nothing overwritten; resolve manually)", "conflicts"), ("Local newer than last sync (kept, not overwritten)", "local_newer")):
                if status[key]:
                    L += ["", "## " + title] + ["- `%s`" % (x if isinstance(x, str) else x["path"] + " - " + x["why"]) for x in status[key]]
            if status["skipped_remote"]:
                L += ["", "## Not copied (by policy)", "- %d files skipped (archives, binaries, logs, oversize). First few:" % len(status["skipped_remote"])]
                L += ["  - `%s` (%s)" % tuple(x) for x in status["skipped_remote"][:10]]
            if prev.get("artifacts"):
                L += ["", "<!-- ARTIFACTS:BEGIN -->", "", "## Recovery artifacts (stored locally; GGUF models are NEVER stored)", ""]
                L += ["- `%s` %d bytes sha256 `%s` (pulled %s)" % (n, m["size"], m["sha256"], m["pulled_utc"]) for n, m in sorted(prev["artifacts"].items())]
                L += ["- not stored: `%s` sha256 `%s` - %s" % (n, m.get("sha256"), m["reason"]) for n, m in sorted(prev.get("artifacts_not_stored", {}).items())]
                L += ["", "<!-- ARTIFACTS:END -->"]
            L += ["", "Local files are never deleted by the sync. Re-run `scripts/sync_from_kaggle.py` to resume."]
            spath.write_text("\n".join(L) + "\n")
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in status.items()}, indent=1))
        return code

    # 1. remote listing (read-only)
    try:
        r = ssh(a.host, "python3 -I - %s %d '%s' '%s' '%s'" % (a.remote_root, MAX_FILE, json.dumps(sorted(EXCLUDE_DIRS)),
                json.dumps(list(EXCLUDE_SUFFIX)), json.dumps(sorted(ALWAYS))), stdin=REMOTE_LIST.encode())
        if r.returncode != 0:
            raise RuntimeError("remote listing failed rc=%d: %s" % (r.returncode, r.stderr.decode(errors="replace")[-300:]))
        listing = json.loads(r.stdout)
    except Exception as e:  # noqa
        status["result"], status["error"] = "FAILED (remote unreachable; last good local copy preserved)", str(e)[:300]
        return finish(2)
    remote, status["skipped_remote"] = listing["files"], listing["skipped"]

    # 2. plan
    to_fetch = []
    for rel, rm in sorted(remote.items()):
        lp = dest / rel
        last = pf.get(rel, {}).get("sha256")          # hash recorded at last successful sync
        if lp.exists():
            lh = sha256(lp)
            if lh == rm["sha256"]:
                status["unchanged"] += 1
                pf.setdefault(rel, {"sha256": rm["sha256"], "size": rm["size"], "synced_utc": now})
                continue
            if last is None:
                status["conflicts"].append({"path": rel, "why": "pre-existing local file differs from remote and was never synced"})
            elif lh != last and rm["sha256"] != last:
                status["conflicts"].append({"path": rel, "why": "changed both locally and on Kaggle since last sync"})
            elif lh != last:
                status["local_newer"].append({"path": rel, "why": "local edited, remote unchanged"})
            else:
                to_fetch.append(rel)                  # remote changed, local untouched
        else:
            to_fetch.append(rel)
    if a.dry_run:
        print("would fetch:", to_fetch)
        status["result"] = "dry-run"
        return finish(0)

    # 3. fetch into staging, verify, then move
    if to_fetch:
        stage = dest / ".sync_staging"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir()
        try:
            r = ssh(a.host, "cd %s && xargs -d '\\n' tar -cf -" % a.remote_root, stdin=("\n".join(to_fetch) + "\n").encode(), timeout=600)
            if r.returncode != 0 or not r.stdout:
                raise RuntimeError("remote tar failed rc=%d: %s" % (r.returncode, r.stderr.decode(errors="replace")[-300:]))
            with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:") as t:
                for m in t.getmembers():
                    if not m.isfile() or m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                        raise RuntimeError("unsafe member in stream: %s" % m.name)
                    t.extract(m, stage)
            for rel in to_fetch:
                sp = stage / rel
                if not sp.is_file() or sha256(sp) != remote[rel]["sha256"]:
                    raise RuntimeError("checksum mismatch after transfer: %s" % rel)
            for rel in to_fetch:
                lp = dest / rel
                lp.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / rel, lp)
                pf[rel] = {"sha256": remote[rel]["sha256"], "size": remote[rel]["size"], "synced_utc": now}
                status["copied"].append(rel)
        except Exception as e:  # noqa
            status["result"], status["error"] = "FAILED mid-transfer (nothing partially applied)", str(e)[:300]
            shutil.rmtree(stage, ignore_errors=True)
            return finish(3)
        shutil.rmtree(stage, ignore_errors=True)
    status["result"] = "OK" if not status["conflicts"] else "OK with conflicts"
    return finish(0 if not status["conflicts"] else 4)


if __name__ == "__main__":
    sys.exit(main())
