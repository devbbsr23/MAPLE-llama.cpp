#!/usr/bin/env python3
"""Build MAPLE artifacts (TAR.GZ only). NEVER packages model weights or any *.gguf.

  package_artifacts.py estimate            # disk requirement report, no writes
  package_artifacts.py llamacpp            # MAPLE-llamacpp-cuda.tar.gz from the existing compiled archive
  package_artifacts.py runtime | deps | complete | manifest | all

The original compiled llama.cpp archive is only ever READ (streamed); it is not
modified, extracted into place, or rebuilt. Nothing is compiled."""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maple import resource_manager as rm  # noqa: E402
from maple.config import MODEL_ROSTER  # noqa: E402

DEFAULT_SRC_ARCHIVE = "/kaggle/working/downloaded-llama-cpp-full-cuda-t4.tar.gz"
DEFAULT_OUT = "/kaggle/working/artifacts"
EXCLUDE_GLOBS = ["*.gguf", "models/*", "*/models/*", "*.ggml", "*.safetensors", "__pycache__/*", "*.pyc", "*.log", ".git/*", "*/.git/*"]
MODEL_DIR_PREFIX = "/kaggle/working/models"
GIB = 1024 ** 3


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def forbidden(name: str) -> bool:
    n = name.lstrip("./")
    if "__pycache__" in n.split("/") or n.endswith((".pyc", ".pyo")):
        return True
    return any(fnmatch.fnmatch(n, g) or fnmatch.fnmatch(os.path.basename(n), g) for g in EXCLUDE_GLOBS) \
        or n.startswith("models/") or "/models/" in n or n.startswith(MODEL_DIR_PREFIX.lstrip("/"))


def model_stat():
    out = {}
    for mid, m in MODEL_ROSTER.items():
        p = m["model_path"]
        try:
            s = os.stat(p)
            out[mid] = {"path": p, "size": s.st_size, "mtime_ns": s.st_mtime_ns}
        except OSError as e:
            out[mid] = {"path": p, "error": str(e)}
    return out


# ---------------------------------------------------------------- llama.cpp
def wanted_llamacpp(name: str) -> bool:
    if forbidden(name):
        return False
    if name.startswith("llama.cpp/build/bin/"):
        return not os.path.basename(name).startswith("test-") and name != "llama.cpp/build/bin"
    return name in ("llama.cpp/build/CMakeCache.txt", "llama.cpp/LICENSE",
                    "llama.cpp/build/CMakeFiles/3.31.10/CMakeCUDACompiler.cmake",
                    "llama.cpp/build/CMakeFiles/3.31.10/CMakeCXXCompiler.cmake")


def parse_cmake_cache(text: str) -> dict:
    want = ("CMAKE_BUILD_TYPE", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER", "CMAKE_CXX_COMPILER",
            "BUILD_SHARED_LIBS", "CMAKE_CUDA_FLAGS", "CUDAToolkit_ROOT", "GGML_CUDA", "GGML_NATIVE", "GGML_OPENMP")
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line)
        if m and (m.group(1) in want or m.group(1).startswith("GGML_CUDA")):
            out[m.group(1)] = m.group(2)
    return out


def package_llamacpp(src: str, out_dir: str, probe_version: str | None = None, live_tree: str = "/kaggle/working/llama.cpp"):
    src_p = Path(src)
    if not src_p.is_file():
        raise SystemExit("source archive not found: %s" % src)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    src_size = src_p.stat().st_size
    rm.require_disk(out_dir_p, src_size, "MAPLE-llamacpp-cuda.tar.gz (upper bound = source archive size)")
    src_hash = sha256(src)
    src_stat_before = (src_p.stat().st_size, src_p.stat().st_mtime_ns)
    models_before = model_stat()

    out = out_dir_p / "MAPLE-llamacpp-cuda.tar.gz"
    tmp = out.with_suffix(".gz.partial")
    included, symlinks, excluded_gguf, cache_text, cuda_cmake = [], [], [], None, None
    with tarfile.open(src, "r|gz") as tin, gzip.GzipFile(tmp, "wb", compresslevel=6, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as tout:
        for ti in tin:
            if ti.name.endswith(".gguf") and ti.isfile():
                excluded_gguf.append({"path": ti.name, "size": ti.size})
            if not wanted_llamacpp(ti.name):
                continue
            if ti.isfile():
                f = tin.extractfile(ti)
                data = f.read() if ti.name.endswith(("CMakeCache.txt", "CMakeCUDACompiler.cmake")) else None
                if ti.name.endswith("CMakeCache.txt"):
                    cache_text = data.decode("utf-8", "replace")
                if ti.name.endswith("CMakeCUDACompiler.cmake"):
                    cuda_cmake = data.decode("utf-8", "replace")
                if data is not None:
                    tout.addfile(ti, io.BytesIO(data))
                else:
                    f2 = tin.extractfile(ti)
                    tout.addfile(ti, f2)
                included.append({"path": ti.name, "size": ti.size, "mode": oct(ti.mode)})
            elif ti.issym():
                tout.addfile(ti)
                symlinks.append({"path": ti.name, "target": ti.linkname})
            elif ti.isdir():
                tout.addfile(ti)
        # metadata
        cuda_ver = None
        if cuda_cmake:
            m = re.search(r'set\(CMAKE_CUDA_COMPILER_VERSION "([^"]+)"', cuda_cmake)
            cuda_ver = m.group(1) if m else None
        live_commit = None
        if live_tree and Path(live_tree, ".git").exists():
            try:
                live_commit = subprocess.run(["git", "-C", live_tree, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or None
            except Exception:
                pass
        info = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_archive": {"path": str(src_p), "sha256": src_hash, "size": src_size, "modified": False,
                               "note": "reusable shared artifact; only read (streamed) by this packager"},
            "llama_cpp": {"commit_from_live_tree_read_only": live_commit, "version_probe": probe_version,
                          "cmake_cache": parse_cmake_cache(cache_text or ""), "cuda_compiler_version": cuda_ver},
            "cuda_architectures_built": "75 (sm_75, Tesla T4/Turing) - see CMAKE_CUDA_ARCHITECTURES",
            "recompiled": False,
            "binaries_modified": False,
            "excluded": {"patterns": EXCLUDE_GLOBS, "test_binaries": "llama.cpp/build/bin/test-*",
                         "gguf_files": len(excluded_gguf), "gguf_bytes": sum(x["size"] for x in excluded_gguf)},
            "packaged_files": len(included), "packaged_symlinks": len(symlinks),
            "runpath_note": "binaries carry RUNPATH=/kaggle/working/llama.cpp/build/bin; MAPLE sets LD_LIBRARY_PATH to its own restored copy, which takes precedence. The protected /kaggle/working/llama.cpp is never modified.",
            "host_library_requirements": {
                "cuda_runtime": "libcudart.so.13 (CUDA 13.x runtime)", "cublas": "libcublas.so.13, libcublasLt.so.13 (CUDA 13.x)",
                "nccl": "libnccl.so.2", "driver": "libcuda.so.1 from an NVIDIA driver supporting CUDA 13.0 (>=580.x); on Kaggle at /usr/local/nvidia/lib64",
                "system": ["libssl.so.3", "libcrypto.so.3", "libgomp.so.1", "libstdc++.so.6 (GLIBCXX>=3.4.29)", "glibc>=2.38"],
                "bundled": False, "policy": "never installed or replaced silently; restore_kaggle.py stops with a diagnostic if missing"},
        }
        for name, content in (("MAPLE-BUILD-INFO/build_info.json", json.dumps(info, indent=2).encode()),
                              ("MAPLE-BUILD-INFO/excluded_gguf.json", json.dumps(excluded_gguf, indent=2).encode()),
                              ("MAPLE-BUILD-INFO/included_files.json", json.dumps({"files": included, "symlinks": symlinks}, indent=1).encode())):
            t = tarfile.TarInfo(name)
            t.size = len(content)
            t.mtime = 0
            t.mode = 0o644
            tout.addfile(t, io.BytesIO(content))
    os.replace(tmp, out)

    # post-conditions
    bad = [n for n in tarfile.open(out, "r:gz").getnames() if n.endswith(".gguf")]
    assert not bad, "gguf found in package: %s" % bad
    assert src_hash == sha256(src), "SOURCE ARCHIVE CHANGED"
    assert src_stat_before == (src_p.stat().st_size, src_p.stat().st_mtime_ns), "source archive stat changed"
    assert models_before == model_stat(), "model file stat changed"
    return {"artifact": str(out), "sha256": sha256(out), "size": out.stat().st_size, "info": info,
            "excluded_gguf": excluded_gguf, "included_files": len(included), "symlinks": len(symlinks)}


# ---------------------------------------------------------------- runtime/deps/complete
RUNTIME_INCLUDE = ["src", "config", "scripts", "tests", "docs", "notebooks", "manifests", "README.md", "RECOVERY_GUIDE.md", "LICENSE", "NOTICE",
                   ".gitignore", ".gitattributes", "requirements.txt", "pyproject.toml"]


def _tar_filter(ti: tarfile.TarInfo):
    if forbidden(ti.name) or "/logs/" in ti.name or ti.name.startswith("logs/") or "/.cache/" in ti.name:
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    return ti


def package_runtime(out_dir):
    out = Path(out_dir) / "MAPLE-runtime.tar.gz"
    rm.require_disk(out_dir, 20 << 20, "MAPLE-runtime.tar.gz")
    with gzip.GzipFile(out, "wb", compresslevel=9, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as t:
        for item in RUNTIME_INCLUDE:
            p = ROOT / item
            if p.exists():
                t.add(p, arcname="MAPLE-llama.cpp/" + item, filter=_tar_filter)
    names = tarfile.open(out, "r:gz").getnames()
    assert not [n for n in names if forbidden(n)], "forbidden member in runtime archive"
    return {"artifact": str(out), "sha256": sha256(out), "size": out.stat().st_size, "members": len(names)}


def package_deps(out_dir, wheel_dir=None):
    out = Path(out_dir) / "MAPLE-python-deps.tar.gz"
    stage = Path(os.environ.get("TMPDIR", "/tmp")) / ("maple_deps_%d" % os.getpid())
    (stage / "wheels").mkdir(parents=True)
    notes = []
    req = (ROOT / "requirements.txt").read_text()
    pins = [l.strip() for l in req.splitlines() if l.strip() and not l.startswith("#")]
    if wheel_dir and Path(wheel_dir).is_dir():
        for w in Path(wheel_dir).glob("*.whl"):
            shutil.copy2(w, stage / "wheels" / w.name)
    elif pins:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", str(stage / "wheels"), *pins],
                           capture_output=True, text=True)
        notes.append("pip download rc=%d" % r.returncode)
        if r.returncode:
            notes.append(r.stderr[-400:])
    (stage / "requirements.lock").write_text("# MAPLE python dependency lock (core runtime uses the standard library only)\n" + req)
    (stage / "INSTALL.md").write_text(
        "# Offline install\n\n```\npython3 -m pip install --no-index --find-links wheels -r requirements.lock\n```\n\n"
        "The MAPLE runtime itself needs only the Python standard library; PyYAML is optional (configs are JSON-compatible YAML).\n"
        "Wheels are for the Python/platform recorded in `wheels.json`; if incompatible, install from the lock file online instead.\n")
    wheels = sorted(p.name for p in (stage / "wheels").glob("*.whl"))
    (stage / "wheels.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "wheels": wheels, "notes": notes}, indent=2))
    with gzip.GzipFile(out, "wb", compresslevel=9, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as t:
        t.add(stage, arcname="MAPLE-python-deps", filter=_tar_filter)
    shutil.rmtree(stage, ignore_errors=True)
    return {"artifact": str(out), "sha256": sha256(out), "size": out.stat().st_size, "wheels": wheels, "notes": notes}


def package_complete(out_dir):
    out = Path(out_dir)
    parts = ["MAPLE-llamacpp-cuda.tar.gz", "MAPLE-runtime.tar.gz", "MAPLE-python-deps.tar.gz"]
    for p in parts:
        if not (out / p).exists():
            raise SystemExit("missing component: %s (run the earlier steps first)" % p)
    need = sum((out / p).stat().st_size for p in parts)
    rm.require_disk(out, need, "MAPLE-complete.tar.gz")
    dest = out / "MAPLE-complete.tar.gz"
    with gzip.GzipFile(dest, "wb", compresslevel=1, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as t:
        for p in parts:
            t.add(out / p, arcname="MAPLE-complete/" + p)
        for extra in ("scripts/restore_kaggle.py", "scripts/restore_kaggle.sh", "scripts/verify_artifacts.sh", "scripts/verify_artifacts.py"):
            if (ROOT / extra).exists():
                t.add(ROOT / extra, arcname="MAPLE-complete/" + os.path.basename(extra))
        for mf in ("MANIFEST.json", "checksums.sha256"):
            if (out / mf).exists():
                t.add(out / mf, arcname="MAPLE-complete/" + mf)
        for mf in (ROOT / "manifests").glob("*.json"):
            t.add(mf, arcname="MAPLE-complete/manifests/" + mf.name)
    assert not [n for n in tarfile.open(dest, "r:gz").getnames() if forbidden(n)], "forbidden member"
    return {"artifact": str(dest), "sha256": sha256(dest), "size": dest.stat().st_size, "components": parts}


def write_manifests(out_dir, results: dict, report_extra: dict | None = None):
    out = Path(out_dir)
    names = ["MAPLE-llamacpp-cuda.tar.gz", "MAPLE-runtime.tar.gz", "MAPLE-python-deps.tar.gz", "MAPLE-complete.tar.gz"]
    arts = []
    for n in names:
        p = out / n
        if p.exists():
            with tarfile.open(p, "r:gz") as t:
                members = t.getnames()
            arts.append({"name": n, "size": p.stat().st_size, "sha256": sha256(p), "members": len(members),
                         "contains_gguf": any(x.endswith(".gguf") for x in members),
                         "verified": False, "verified_note": "set to true only by verify_artifacts after checksum + integrity + smoke tests"})
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "format": "tar.gz", "artifacts": arts,
                "models_included": False, "model_paths": {k: v["model_path"] for k, v in MODEL_ROSTER.items()}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "checksums.sha256").write_text("".join("%s  %s\n" % (a["sha256"], a["name"]) for a in arts))
    (ROOT / "manifests").mkdir(exist_ok=True)
    (ROOT / "manifests" / "artifacts_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def estimate(out_dir, src):
    free = rm.disk_free(out_dir)
    srcs = Path(src).stat().st_size if Path(src).exists() else 0
    runtime_est = sum(f.stat().st_size for d in ("src", "config", "scripts", "tests", "docs", "notebooks") for f in (ROOT / d).rglob("*") if f.is_file())
    rows = [("MAPLE-llamacpp-cuda.tar.gz", srcs, "upper bound = source archive size (streamed, no staging copy)"),
            ("MAPLE-runtime.tar.gz", runtime_est, "uncompressed source size, actual is smaller"),
            ("MAPLE-python-deps.tar.gz", 50 << 20, "wheel download estimate"),
            ("MAPLE-complete.tar.gz", srcs + runtime_est + (50 << 20), "wraps the three archives above (compresslevel=1)")]
    total = sum(r[1] for r in rows)
    print("free on %s: %.2f GiB" % (out_dir, free / GIB))
    for n, b, why in rows:
        print("  %-30s <= %8.1f MiB  (%s)" % (n, b / 1048576, why))
    print("worst-case total needed: %.1f MiB (+10%% margin = %.1f MiB) -> %s" % (
        total / 1048576, total * 1.1 / 1048576, "OK" if free > total * 1.1 else "INSUFFICIENT"))
    return free > total * 1.1


def write_report(out_dir, results):
    L = ["MAPLE packaging report", "generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), ""]
    for k, v in results.items():
        L.append("== %s" % k)
        for kk in ("artifact", "sha256", "size", "members", "included_files", "symlinks", "components", "wheels", "notes"):
            if kk in v:
                L.append("  %s: %s" % (kk, v[kk]))
        if k == "llamacpp":
            info = v["info"]
            L.append("  source archive: %s" % info["source_archive"]["path"])
            L.append("  source archive sha256: %s (unmodified, verified after packaging)" % info["source_archive"]["sha256"])
            L.append("  recompiled: no; binaries modified: no")
            L.append("  excluded *.gguf files: %d (%d bytes) - vocabulary test files, NOT model weights:" % (len(v["excluded_gguf"]), sum(x["size"] for x in v["excluded_gguf"])))
            for x in v["excluded_gguf"]:
                L.append("    - %s (%d bytes)" % (x["path"], x["size"]))
            L.append("  excluded test binaries: llama.cpp/build/bin/test-*")
        L.append("")
    L.append("model files: not packaged, not copied, not modified; stat (size, mtime) recorded before and after:")
    for mid, s in model_stat().items():
        L.append("  %-9s %s %s" % (mid, s.get("path"), s.get("size", s.get("error"))))
    (Path(out_dir) / "packaging-report.txt").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["estimate", "llamacpp", "runtime", "deps", "complete", "manifest", "all"])
    ap.add_argument("--src-archive", default=DEFAULT_SRC_ARCHIVE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--probe-version", help="string from llama-server --version, recorded in build info")
    ap.add_argument("--wheel-dir")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    state_f = Path(a.out) / ".package_results.json"
    results = json.loads(state_f.read_text()) if state_f.exists() else {}
    if a.step == "estimate":
        return 0 if estimate(a.out, a.src_archive) else 3
    if a.step in ("llamacpp", "all"):
        results["llamacpp"] = package_llamacpp(a.src_archive, a.out, a.probe_version)
    if a.step in ("runtime", "all"):
        results["runtime"] = package_runtime(a.out)
    if a.step in ("deps", "all"):
        results["deps"] = package_deps(a.out, a.wheel_dir)
    if a.step in ("manifest", "all", "complete"):
        write_manifests(a.out, results)  # so complete embeds checksums of the three parts
    if a.step in ("complete", "all"):
        results["complete"] = package_complete(a.out)
    if a.step in ("manifest", "all", "complete"):
        write_manifests(a.out, results)
    state_f.write_text(json.dumps(results, indent=1, default=str))
    write_report(a.out, results)
    print(json.dumps({k: {x: v[x] for x in ("artifact", "sha256", "size") if x in v} for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
