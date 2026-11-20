#!/usr/bin/env python3
"""Verify that this LOCAL directory holds everything needed to rebuild MAPLE and restore the existing
inference backend on a fresh server WITHOUT recompiling llama.cpp or downloading models.

Standard library only; runs on Windows/Linux. Read-only except for a temp dir it creates and removes.

    python scripts/verify_local_recovery.py                 # all checks
    python scripts/verify_local_recovery.py --extract-test  # + unpack the archives into a clean temp dir and run the unit tests there
    python scripts/verify_local_recovery.py --json

Exit code 0 only if every check passes. Each check prints PASS/FAIL with the reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SOURCE = [
    "README.md", "LICENSE", "NOTICE", "RECOVERY_GUIDE.md", ".gitignore", ".gitattributes", "requirements.txt", "pyproject.toml",
    "SYNC_STATUS.md", "LOCAL_BACKUP_MANIFEST.json",
    "config/models.yaml", "config/runtime.yaml",
    "src/maple/__init__.py", "src/maple/__main__.py", "src/maple/api.py", "src/maple/config.py", "src/maple/gguf_meta.py",
    "src/maple/registry.py", "src/maple/resource_manager.py", "src/maple/router.py", "src/maple/runtime.py",
    "scripts/restore_kaggle.py", "scripts/restore_kaggle.sh", "scripts/start_maple.sh", "scripts/verify_artifacts.py",
    "scripts/verify_artifacts.sh", "scripts/package_artifacts.py", "scripts/validate_models.py", "scripts/benchmark.py",
    "scripts/make_build_manifest.py", "scripts/make_notebooks.py", "scripts/sync_from_kaggle.py", "scripts/pull_artifacts.py",
    "scripts/push_to_kaggle.py", "scripts/backup_llamacpp.py", "scripts/clean_restore_probe.py", "scripts/smoke_api.py", "scripts/link_models.py", "scripts/verify_local_recovery.py",
    "notebooks/MAPLE_Kaggle_Build.ipynb", "notebooks/MAPLE_Kaggle_Restore.ipynb",
    "tests/test_registry_router.py", "tests/test_api.py", "tests/test_archives.py", "tests/test_live_models.py",
    "README.md", "LICENSE", "NOTICE", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "config/example.env", "scripts/clean_restore_probe.py", "tests/test_http_api.py", "docs/ARCHITECTURE.md", "docs/INSTALLATION.md", "docs/KAGGLE_DEPLOYMENT.md", "docs/RESTORATION.md", "docs/MODEL_CONFIGURATION.md", "docs/OPENAI_API.md", "docs/LOCAL_CLIENTS.md", "docs/TROUBLESHOOTING.md", "docs/HARDWARE_REQUIREMENTS.md", "docs/GITHUB_RELEASE.md", "docs/KAGGLE_FREE_TIER_DEPLOYMENT.md", "docs/LOCAL_API_USAGE.md", "docs/RECOVERY_GUIDE.md", "artifacts/README.md",
    "manifests/models_manifest.json", "manifests/build_manifest.json", "manifests/artifacts_manifest.json", "manifests/benchmark_results.json", "manifests/clean_restore_report.json",
]
STORED_ARCHIVES = ["MAPLE-llamacpp-cuda.tar.gz", "MAPLE-runtime.tar.gz", "MAPLE-python-deps.tar.gz"]
ARTIFACT_SIDE_FILES = ["checksums.sha256", "MANIFEST.json", "packaging-report.txt"]
ROSTER_IDS = ["deepseek", "gemma", "qwen3", "fast"]
MODEL_EXT = (".gguf", ".ggml", ".safetensors")

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name, (" - " + detail) if detail else ""))
    return ok


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audit_tar(path):
    """Full read (gzip CRC) + content policy. Returns (member_count, problems, names)."""
    problems, names = [], []
    with tarfile.open(path, "r:gz") as t:
        for m in t:
            n = m.name
            names.append(n)
            if n.startswith("/") or ".." in Path(n).parts:
                problems.append("unsafe path " + n)
            if n.lower().endswith(MODEL_EXT):
                problems.append("model file " + n)
            if "__pycache__" in n.split("/") or n.endswith((".pyc", ".pyo")):
                problems.append("cache file " + n)
            if "/.git/" in "/" + n or n.split("/")[0] == ".git":
                problems.append("git history " + n)
            if m.isfile():
                f = t.extractfile(m)
                while f.read(1 << 20):
                    pass
    return len(names), problems, names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--extract-test", action="store_true", help="unpack the archives into a clean temp dir and run the unit tests there")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    print("verifying local recovery package: %s\n" % root)

    # 1. required files
    missing = [f for f in REQUIRED_SOURCE if not (root / f).is_file()]
    check("required source/docs/scripts/notebooks/manifests present", not missing, "%d required files" % len(REQUIRED_SOURCE) if not missing else "MISSING: %s" % missing)

    # 2. checksums of every tracked source file vs LOCAL_BACKUP_MANIFEST.json
    man = {}
    try:
        man = json.loads((root / "LOCAL_BACKUP_MANIFEST.json").read_text())
        bad, n = [], 0
        for rel, meta in man.get("files", {}).items():
            p = root / rel
            n += 1
            if not p.is_file() or sha256(p) != meta.get("sha256"):
                bad.append(rel)
        # files edited locally after the last sync show up here (reported, not hidden)
        check("source files match LOCAL_BACKUP_MANIFEST.json sha256", not bad and n > 0, "%d files verified" % n if not bad else "differ/missing: %s" % bad[:8])
    except (OSError, ValueError) as e:
        check("LOCAL_BACKUP_MANIFEST.json readable", False, str(e))

    # 3. artifacts: presence, sha256 vs checksums.sha256 and vs the manifest, integrity, content policy
    adir = root / "artifacts"
    sums = {}
    try:
        for line in (adir / "checksums.sha256").read_text().splitlines():
            m = re.match(r"^([0-9a-f]{64})\s+\*?(\S+)$", line.strip())
            if m:
                sums[m.group(2)] = m.group(1)
    except OSError:
        pass
    check("artifact side files present", all((adir / f).is_file() for f in ARTIFACT_SIDE_FILES), ", ".join(ARTIFACT_SIDE_FILES))
    stored = man.get("artifacts", {})
    names_by_archive = {}
    for n in STORED_ARCHIVES:
        p = adir / n
        if not p.is_file():
            check("archive %s" % n, False, "missing - run scripts/pull_artifacts.py while a Kaggle session that has it is reachable")
            continue
        got = sha256(p)
        want_sum, want_man = sums.get(n), stored.get(n, {}).get("sha256")
        ok = got == want_sum and (want_man is None or got == want_man)
        try:
            cnt, probs, names = audit_tar(p)
            names_by_archive[n] = names
        except (tarfile.TarError, OSError, EOFError) as e:
            cnt, probs = 0, ["unreadable: %s" % e]
        check("archive %s" % n, ok and not probs, "sha256 %s.. size %d members %d" % (got[:12], p.stat().st_size, cnt) if ok and not probs else
              "sha256 ok=%s problems=%s" % (ok, probs[:3]))
    cuda = names_by_archive.get("MAPLE-llamacpp-cuda.tar.gz", [])
    need_bins = ["llama-server", "llama-cli", "llama-quantize", "libggml-cuda.so.1.0.0", "libllama.so.1.0.0", "libggml-base.so.1.0.0"]
    have = {os.path.basename(n) for n in cuda}
    check("llama.cpp archive has binaries + shared libraries", all(b in have for b in need_bins) or not cuda,
          "missing %s" % [b for b in need_bins if b not in have] if cuda and not all(b in have for b in need_bins) else "executables and libs present")
    if "MAPLE-runtime.tar.gz" in names_by_archive:
        rt = names_by_archive["MAPLE-runtime.tar.gz"]
        want = ["MAPLE-llama.cpp/src/maple/config.py", "MAPLE-llama.cpp/scripts/restore_kaggle.py", "MAPLE-llama.cpp/scripts/start_maple.sh",
                "MAPLE-llama.cpp/config/models.yaml", "MAPLE-llama.cpp/notebooks/MAPLE_Kaggle_Restore.ipynb"]
        check("runtime archive has application + restore scripts", all(w in rt for w in want), "missing %s" % [w for w in want if w not in rt] if not all(w in rt for w in want) else "ok")

    # 4. no model weights / binaries duplicated outside the archives
    stray = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and (p.suffix.lower() in MODEL_EXT or "models" in p.relative_to(root).parts[:1])]
    check("no GGUF/model files anywhere in the local package", not stray, "none found" if not stray else str(stray[:5]))

    # 5b. clean-restoration evidence
    try:
        cr = json.loads((root / "manifests" / "clean_restore_report.json").read_text())
        check("clean restoration report: PASS with no failed checks", cr.get("result") == "PASS" and cr["probe"]["failed"] == 0 and cr.get("restore_ok") and cr.get("no_recompilation"),
              "%d probe checks passed, restore steps ok=%s" % (cr["probe"]["passed"], cr.get("restore_ok")))
    except Exception as e:  # noqa
        check("clean restoration report: PASS with no failed checks", False, "%s: %s" % (type(e).__name__, e))

    # 5c. original compiled llama.cpp archive backup (kept OUTSIDE the source folder: <root>/../backups/llamacpp)
    lc = root.parent / "backups" / "llamacpp"
    try:
        lm = json.loads((lc / "LLAMACPP_BACKUP_MANIFEST.json").read_text())
        arc = lc / lm["file"]
        got = sha256(arc)
        check("llama.cpp original archive backup: size + SHA256 match manifest", arc.is_file() and arc.stat().st_size == lm["size_bytes"] and got == lm["sha256"],
              "%s %d bytes sha256 %s.." % (lm["file"], arc.stat().st_size, got[:12]))
        check("llama.cpp backup: SHA256 was verified against the Kaggle source and the original was left unchanged",
              lm.get("sha256_matches_kaggle_source") is True and lm.get("kaggle_original_unchanged_after_transfer") is True, lm.get("source", ""))
        members, g = 0, 0
        with tarfile.open(arc, "r:gz") as t:
            for m in t:
                members += 1
                if m.isfile():
                    f = t.extractfile(m)
                    while f.read(1 << 20):
                        pass
        check("llama.cpp backup: gzip + tar fully readable", members == lm["tar_members"], "%d members" % members)
        check("llama.cpp backup is not inside the source folder", lc.resolve() != root.resolve() and root.resolve() not in lc.resolve().parents, str(lc))
    except Exception as e:  # noqa
        check("llama.cpp original archive backup verified", False, "%s: %s" % (type(e).__name__, e))
    dup = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and (p.name.startswith("lib") and ".so" in p.name or p.name in ("llama-server", "llama-cli")) and "artifacts" not in p.parts]
    check("no loose compiled binaries (they live only inside the archive)", not dup, str(dup[:5]) if dup else "ok")

    # 5. MODEL_ROSTER + models manifest
    try:
        roster = runpy.run_path(str(root / "src" / "maple" / "config.py"))["MODEL_ROSTER"]
        mm = json.loads((root / "manifests" / "models_manifest.json").read_text())["models"]
        ok = list(roster) == ROSTER_IDS and set(mm) == set(ROSTER_IDS)
        why = []
        for k in ROSTER_IDS:
            r, m = roster.get(k, {}), mm.get(k, {})
            if m.get("original_path") != r.get("model_path"):
                why.append("%s path differs" % k)
            if not re.fullmatch(r"[0-9a-f]{64}", str(m.get("sha256"))):
                why.append("%s sha256 missing" % k)
            if not m.get("size_bytes") or not m.get("gguf", {}).get("architecture"):
                why.append("%s size/architecture missing" % k)
            if m.get("validation_status") != "ok":
                why.append("%s status %s" % (k, m.get("validation_status")))
        check("MODEL_ROSTER exact + models_manifest (path, size, sha256, architecture, status)", ok and not why, "; ".join(why) if why else
              ", ".join("%s=%s" % (k, mm[k]["gguf"]["architecture"]) for k in ROSTER_IDS))
        tested = {k: mm[k].get("tested", {}).get("max_validated_ctx") for k in ROSTER_IDS}
        check("tested context lengths recorded (no unvalidated 128K claim)", all(tested.values()) and tested.get("deepseek", 0) < 131072, str(tested))
    except Exception as e:  # noqa
        check("MODEL_ROSTER + models manifest", False, "%s: %s" % (type(e).__name__, e))

    # 6. compatibility information
    try:
        b = json.loads((root / "manifests" / "build_manifest.json").read_text())
        lc = json.dumps(b.get("llama_cpp"))
        ok = "ef3848f" in lc and b.get("cuda", {}).get("driver") and b.get("gpus") and "sm_75" in json.dumps(b) + (root / "README.md").read_text() + (root / "RECOVERY_GUIDE.md").read_text()
        check("build manifest records llama.cpp commit, CUDA, GPUs; sm_75 documented", bool(ok), "llama.cpp ef3848f, CUDA driver %s" % b.get("cuda", {}).get("driver"))
    except Exception as e:  # noqa
        check("build manifest", False, "%s: %s" % (type(e).__name__, e))

    # 7. unit tests of the source as it sits here (live GPU tests are skipped without MAPLE_LIVE=1)
    env = dict(os.environ, MAPLE_HOME=str(root))
    env.pop("MAPLE_LIVE", None)
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=root, env=env, capture_output=True, text=True, timeout=300)
    tail = (r.stderr.strip().splitlines() or ["?"])[-1]
    check("unit tests (non-GPU) pass from the local source", r.returncode == 0, tail)

    # 8. optional: reconstruct in a clean temp dir from the archives only
    if a.extract_test:
        tmp = Path(tempfile.mkdtemp(prefix="maple_recovery_check_"))
        try:
            ok = True
            with tarfile.open(adir / "MAPLE-runtime.tar.gz") as t:
                t.extractall(tmp, filter="data") if sys.version_info >= (3, 12) else t.extractall(tmp)
            with tarfile.open(adir / "MAPLE-llamacpp-cuda.tar.gz") as t:
                t.extractall(tmp / "MAPLE-llama.cpp" / "vendor" / "llamacpp", filter="data") if sys.version_info >= (3, 12) else t.extractall(tmp / "MAPLE-llama.cpp" / "vendor" / "llamacpp")
            proj = tmp / "MAPLE-llama.cpp"
            bins = proj / "vendor" / "llamacpp" / "llama.cpp" / "build" / "bin"
            check("clean extract: application + llama.cpp build tree reconstructed", (proj / "src" / "maple" / "api.py").is_file() and (bins / "llama-server").is_file(), str(proj))
            env2 = dict(os.environ, MAPLE_HOME=str(proj))
            env2.pop("MAPLE_LIVE", None)
            r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=proj, env=env2, capture_output=True, text=True, timeout=300)
            check("clean extract: unit tests pass from the extracted copy", r.returncode == 0, (r.stderr.strip().splitlines() or ["?"])[-1])
            same = [f for f in ("src/maple/config.py", "scripts/restore_kaggle.py", "config/models.yaml") if sha256(proj / f) != sha256(root / f)]
            check("extracted application identical to local source (key files)", not same, "identical" if not same else "differs: %s (runtime archive is older than the local source; repackage)" % same)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)       # the temp dir created above, nothing else

    fails = [x for x in results if not x["ok"]]
    print("\n%d checks, %d passed, %d failed" % (len(results), len(results) - len(fails), len(fails)))
    print("LOCAL RECOVERY PACKAGE: %s" % ("COMPLETE" if not fails else "INCOMPLETE"))
    if a.json:
        print(json.dumps(results, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
