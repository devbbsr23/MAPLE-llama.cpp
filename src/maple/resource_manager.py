"""GPU / RAM / disk inspection and model placement planning.

Two T4s are two independent 15 GiB devices; they are NOT treated as one 32 GiB
GPU. A model is placed on a single GPU when it fits with headroom, otherwise it
is layer-split across GPUs."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import build_env, load_runtime_config

GIB = 1024 ** 3


def _smi(cfg=None):
    cfg = cfg or load_runtime_config()
    env = build_env(cfg)
    for cand in cfg["nvidia_smi_candidates"]:
        if os.path.isabs(cand) and not Path(cand).exists():
            continue
        try:
            out = subprocess.run(
                [cand, "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,compute_cap,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, env=env)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    return None


def gpu_info(cfg=None) -> list[dict]:
    """Actual GPUs as reported by nvidia-smi (empty list if unavailable)."""
    raw = _smi(cfg)
    gpus = []
    if not raw:
        return gpus
    for line in raw.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8:
            continue
        gpus.append({
            "index": int(p[0]), "name": p[1], "mem_total_mib": int(p[2]), "mem_used_mib": int(p[3]),
            "mem_free_mib": int(p[4]), "util_pct": int(p[5]) if p[5].isdigit() else None,
            "compute_cap": p[6], "driver": p[7],
        })
    return gpus


def ram_info() -> dict:
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":")
            info[k] = int(v.split()[0]) * 1024
    except OSError:
        return {}
    return {"total": info.get("MemTotal"), "available": info.get("MemAvailable")}


def disk_free(path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def require_disk(path, needed_bytes: int, what: str = "operation", margin: float = 0.10) -> None:
    """Raise if free space (with margin) is below needed_bytes."""
    free = disk_free(path)
    need = int(needed_bytes * (1 + margin))
    if free < need:
        raise OSError("insufficient disk for %s on %s: need %.2f GiB (incl. %d%% margin), free %.2f GiB"
                      % (what, path, need / GIB, int(margin * 100), free / GIB))


def plan_placement(model_size_bytes: int, ctx: int, gpus: list[dict], kv_bytes_per_token: int = 160 * 1024,
                   headroom_mib: int = 700, usable_frac: float = 0.85) -> dict:
    """Decide which GPU(s) to use.

    Estimate = weights + KV(ctx) + compute-buffer headroom. Returns
    {"devices": [indices], "split_mode": "none"|"layer", "estimate_mib": n, "reason": str}.
    Without GPUs returns devices=[] (CPU)."""
    est = int((model_size_bytes + ctx * kv_bytes_per_token) / (1024 * 1024)) + headroom_mib
    if not gpus:
        return {"devices": [], "split_mode": "none", "estimate_mib": est, "reason": "no GPU detected; CPU only"}
    # Only 85% of free memory counts as usable: prefill compute buffers grow beyond the static
    # estimate (observed: a real CUDA OOM during a long prefill when a single GPU was ~93% full).
    fits = [g for g in gpus if g["mem_free_mib"] * usable_frac >= est]
    if fits:
        best = max(fits, key=lambda g: g["mem_free_mib"])
        return {"devices": [best["index"]], "split_mode": "none", "estimate_mib": est,
                "reason": "fits on single GPU %d (%d MiB free >= %d MiB est.)" % (best["index"], best["mem_free_mib"], est)}
    total_free = sum(g["mem_free_mib"] for g in gpus)
    if len(gpus) > 1 and total_free * usable_frac >= est:
        return {"devices": [g["index"] for g in gpus], "split_mode": "layer", "estimate_mib": est,
                "reason": "does not fit one GPU; layer-split across %d GPUs (independent devices, %d MiB free in total)" % (len(gpus), total_free)}
    return {"devices": [g["index"] for g in gpus], "split_mode": "layer" if len(gpus) > 1 else "none",
            "estimate_mib": est, "reason": "estimate %d MiB exceeds free GPU memory (%d MiB); partial CPU offload likely" % (est, total_free)}


def snapshot(cfg=None) -> dict:
    return {"gpus": gpu_info(cfg), "ram": ram_info()}
