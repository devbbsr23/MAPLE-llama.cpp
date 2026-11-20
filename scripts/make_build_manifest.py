#!/usr/bin/env python3
"""Write manifests/build_manifest.json from live probes + recorded results. No values are invented:
anything that cannot be probed is recorded as null."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maple import resource_manager as rm  # noqa: E402
from maple.config import build_env, llama_bin_dir, load_runtime_config  # noqa: E402

UPSTREAM = {"repo": "https://github.com/AmSach/maple", "commit": "5403b6f5ec3f904409d968ff6a3ea3a2b22767d1", "license": "Apache-2.0"}


def sh(cmd, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return None


def main():
    cfg = load_runtime_config()
    env = build_env(cfg)
    b = llama_bin_dir()
    ver = sh([str(b / "llama-server"), "--version"], env)
    info = {}
    for tf in ("/kaggle/working/artifacts/MAPLE-llamacpp-cuda.tar.gz",):
        if Path(tf).exists():
            import tarfile
            with tarfile.open(tf) as t:
                try:
                    info = json.loads(t.extractfile("MAPLE-BUILD-INFO/build_info.json").read())
                except KeyError:
                    pass
    models = {}
    mm = ROOT / "manifests" / "models_manifest.json"
    if mm.exists():
        for k, v in json.loads(mm.read_text())["models"].items():
            t = v.get("tested", {})
            models[k] = {"name": v["name"], "size_bytes": v["size_bytes"], "sha256": v["sha256"], "architecture": v["gguf"]["architecture"],
                         "is_moe": v["gguf"]["is_moe"], "tested_max_ctx": t.get("max_validated_ctx"), "tested_kv_type": t.get("max_validated_kv_type"),
                         "incomplete_runs": t.get("incomplete_runs", []), "validation_status": v["validation_status"]}
    bench = {}
    bf = ROOT / "manifests" / "benchmark_results.json"
    if bf.exists():
        for k, runs in json.loads(bf.read_text())["results"].items():
            bench[k] = [{"ctx": r["requested_ctx"], "ok": r["ok"], "load_s": r.get("load_time_s"), "gen_tps": r.get("short", {}).get("gen_tps"),
                         "prompt_tps": r.get("short", {}).get("prompt_tps"), "kv_mib": r.get("kv_cache_mib"), "layers": r.get("layers_offloaded"),
                         "devices": r.get("placement", {}).get("devices"), "split": r.get("placement", {}).get("split_mode")} for r in runs]
    gpus = rm.gpu_info(cfg)
    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": {"platform": platform.platform(), "glibc": (sh(["ldd", "--version"]) or "").splitlines()[:1]},
        "gpus": gpus,
        "cuda": {"toolkit": re.sub(r"\s+", " ", sh(["/usr/local/cuda/bin/nvcc", "--version"]) or "")[-60:] or None,
                 "driver": gpus[0]["driver"] if gpus else None},
        "compilers": {"gcc": (sh(["gcc", "--version"]) or "").splitlines()[:1], "cmake": (sh(["cmake", "--version"]) or "").splitlines()[:1]},
        "python": sys.version.split()[0],
        "llama_cpp": {"version_string": ver, "commit": (re.search(r"commit (\w+)", ver or "") or [None, None])[1], "recompiled_by_maple": False,
                      "source_archive_sha256": info.get("source_archive", {}).get("sha256"),
                      "build_flags": info.get("llama_cpp", {}).get("cmake_cache"), "cuda_architectures": info.get("cuda_architectures_built")},
        "maple": {"project_git_commit": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) if (ROOT / ".git").exists() else None,
                  "upstream": UPSTREAM},
        "models": models, "benchmarks": bench,
        "gpu_placement_note": "T4s are two independent 15 GiB devices; MAPLE layer-splits only when a model+KV estimate exceeds ~85% of one GPU's free memory.",
    }
    p = ROOT / "manifests" / "build_manifest.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
