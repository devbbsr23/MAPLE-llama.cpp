#!/usr/bin/env python3
"""Per-model runtime test through MAPLE's ModelManager (native llama-server).

For each model: load time, GPU memory delta, GPU placement, KV-cache size,
prompt-processing and generation speed (llama-server timings), and a context
ladder. A context length is reported as validated ONLY if the server loaded
with it and answered a request that actually filled a large part of it.
Results are merged into manifests/models_manifest.json under 'tested'.
Two GPUs are only claimed if two are actually detected."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maple import resource_manager as rm  # noqa: E402
from maple.config import MODEL_ROSTER, load_runtime_config, maple_home  # noqa: E402
from maple.registry import Registry  # noqa: E402
from maple.runtime import LoadError, ModelManager  # noqa: E402

PROMPT = "Write a C++ function that reverses a singly linked list and explain its complexity."


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_log(text):
    o = {}
    m = re.findall(r"(\w+) KV buffer size =\s+([\d.]+) MiB", text)
    o["kv_cache_mib"] = round(sum(float(x[1]) for x in m), 1) if m else None
    o["kv_buffers"] = [{"device": a, "mib": float(b)} for a, b in m]
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    o["layers_offloaded"] = "%s/%s" % m.groups() if m else None
    o["model_buffers"] = [{"device": a, "mib": float(b)} for a, b in re.findall(r"(\w+) model buffer size =\s+([\d.]+) MiB", text)]
    o["compute_buffers"] = [{"device": a, "mib": float(b)} for a, b in re.findall(r"(\w+) compute buffer size =\s+([\d.]+) MiB", text)]
    m = re.search(r"n_ctx_seq\s*=\s*(\d+)", text) or re.search(r"n_ctx\s*=\s*(\d+)", text)
    o["n_ctx_reported"] = int(m.group(1)) if m else None
    return o


def rss_mib(pid):
    try:
        for l in Path("/proc/%d/status" % pid).read_text().splitlines():
            if l.startswith("VmRSS"):
                return round(int(l.split()[1]) / 1024)
    except OSError:
        pass


def bench_model(mgr, mid, ctx, filler_frac=0.6, n_pred=64):
    # long_ok requires the prompt to reach >=90% of the requested fill fraction
    r = {"requested_ctx": ctx, "ok": False}
    try:
        with mgr.use(mid, ctx) as base:
            ll = mgr.last_load
            srv = mgr.server
            r.update(load_time_s=ll["load_time_s"], placement=ll["placement"],
                     gpu_mem_used_mib_before=[g["mem_used_mib"] for g in ll["gpu_before"]],
                     gpu_mem_used_mib_after_load=[g["mem_used_mib"] for g in ll["gpu_after"]],
                     cpu_rss_mib=rss_mib(srv.proc.pid), cmd=" ".join(srv.cmd))
            r.update(parse_log(srv.log_path.read_text(errors="replace")))
            # short generation
            t0 = time.time()
            out = post(base, "/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": PROMPT}],
                                                        "max_tokens": n_pred, "temperature": 0.0})
            tm = out.get("timings", {})
            r["short"] = {"wall_s": round(time.time() - t0, 2), "prompt_n": tm.get("prompt_n"), "prompt_tps": tm.get("prompt_per_second"),
                          "gen_n": tm.get("predicted_n"), "gen_tps": tm.get("predicted_per_second"),
                          "text_head": (out["choices"][0]["message"].get("content") or out["choices"][0]["message"].get("reasoning_content") or "")[:120]}
            # long-context fill: ~filler_frac of ctx
            n_fill = int(ctx * filler_frac)
            if n_fill > 512:
                word = "The quick brown fox jumps over the lazy dog. "
                text = word * (n_fill // 10 + 1)
                t0 = time.time()
                out = post(base, "/v1/completions", {"model": mid, "prompt": text, "max_tokens": 16, "temperature": 0.0})
                tm = out.get("timings", {})
                pn = tm.get("prompt_n") or out.get("usage", {}).get("prompt_tokens")
                r["long"] = {"wall_s": round(time.time() - t0, 2), "prompt_tokens": pn, "prompt_tps": tm.get("prompt_per_second"),
                             "fraction_of_ctx_filled": round(pn / ctx, 2) if pn else None}
                r["long_ok"] = bool(pn and pn >= 0.9 * filler_frac * ctx)
                r["fill_frac_requested"] = filler_frac
            g_after = rm.gpu_info(mgr.cfg)
            r["gpu_mem_used_mib_after_inference"] = [g["mem_used_mib"] for g in g_after]
            r["gpu_util_pct_sample"] = [g["util_pct"] for g in g_after]
            r["ok"] = True
            if r.get("kv_cache_mib"):
                mgr.kv_override[mid] = int(r["kv_cache_mib"] * 1048576 / ctx)  # conservative: average incl. fixed part
    except (LoadError, Exception) as e:  # noqa
        r["error"] = "%s: %s" % (type(e).__name__, str(e)[-1500:])
    finally:
        mgr.unload()
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-frac", type=float, default=0.6, help="fraction of ctx filled by the long-prompt test")
    ap.add_argument("--models", default="fast,qwen3,deepseek,gemma")
    ap.add_argument("--ctx", default="4096", help="comma list of context sizes, ascending; stops a model's ladder on first failure")
    ap.add_argument("--out", default=str(maple_home() / "manifests" / "models_manifest.json"))
    ap.add_argument("--report", default=str(maple_home() / "manifests" / "benchmark_results.json"))
    ap.add_argument("--kv-type", help="KV cache type for K and V (e.g. q8_0); recorded with results. Default f16.")
    ap.add_argument("--keep", action="store_true", help="merge into existing 'tested' instead of replacing")
    a = ap.parse_args(argv)
    cfg = load_runtime_config()
    if a.kv_type:
        for mid in MODEL_ROSTER:
            cfg.setdefault("models", {}).setdefault(mid, {})["extra_args"] = ["-ctk", a.kv_type, "-ctv", a.kv_type]
    cfg["log_verbosity"] = 4  # debug logs expose KV/model buffer sizes and layer offload
    mgr = ModelManager(Registry(), cfg)
    gpus = rm.gpu_info(cfg)
    names = [g["name"] for g in gpus]
    env = {"gpus": gpus, "ram": rm.ram_info(), "two_t4_detected": len(gpus) == 2 and all("T4" in n for n in names),
           "gpu_count": len(gpus), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print("GPUs detected: %d %s | two-T4 validation claimable: %s" % (len(gpus), names, env["two_t4_detected"]))
    results = {}
    for mid in [x for x in a.models.split(",") if x]:
        results[mid] = []
        for ctx in [int(c) for c in a.ctx.split(",")]:
            rm.require_disk(maple_home(), 50 << 20, "benchmark logs")
            res = bench_model(mgr, mid, ctx, filler_frac=a.fill_frac)
            results[mid].append(res)
            s = res.get("short", {})
            print("[%s ctx=%d] ok=%s load=%ss gen=%s tps pp=%s tps kv=%s MiB layers=%s %s" % (
                mid, ctx, res["ok"], res.get("load_time_s"), s.get("gen_tps"), s.get("prompt_tps"), res.get("kv_cache_mib"),
                res.get("layers_offloaded"), ("ERR " + res["error"][:300]) if not res["ok"] else ""))
            if not res["ok"] or res.get("long_ok") is False:
                break
    Path(a.report).write_text(json.dumps({"environment": env, "results": results}, indent=2, default=str))
    mp = Path(a.out)
    if mp.exists():
        man = json.loads(mp.read_text())
        for mid, runs in results.items():
            good = [r for r in runs if r["ok"] and r.get("long_ok") is not False]
            slot = man["models"][MODEL_ROSTER[mid]["id"]]
            prev_t = slot.get("tested", {}) if a.keep else {}
            for r in runs:
                r["kv_type"] = a.kv_type or "f16"
            all_runs = prev_t.get("runs", []) + [{k: v for k, v in r.items() if k not in ("cmd",)} for r in runs]
            okr = [r for r in all_runs if r.get("ok") and r.get("long_ok") is not False]
            top = max(okr, key=lambda r: r["requested_ctx"], default=None)
            slot["tested"] = {
                "max_validated_ctx": top["requested_ctx"] if top else None,
                "max_validated_kv_type": top.get("kv_type") if top else None,
                "kv_bytes_per_token": int(top["kv_cache_mib"] * 1048576 / top["requested_ctx"]) if top and top.get("kv_cache_mib") else None,
                "runs": all_runs,
                "env": {"gpu_count": env["gpu_count"], "gpus": names, "two_t4": env["two_t4_detected"], "time_utc": env["time_utc"]}}
        mp.write_text(json.dumps(man, indent=2, default=str))
    print("wrote", a.report)
    return 0 if all(any(r["ok"] for r in runs) for runs in results.values()) else 4


if __name__ == "__main__":
    sys.exit(main())
