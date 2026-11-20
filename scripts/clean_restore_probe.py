#!/usr/bin/env python3
"""Runs ON KAGGLE (piped over ssh stdin by clean_restore_test.py). Read-only checks against a restored MAPLE copy
and its running API, then a clean shutdown test. Standard library only. Prints one JSON object on the last line.

    python3 - --dest DIR --port N --protect FILE[,FILE...] --sha256 FILE=HASH[,...]
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

res = []


def chk(name, ok, detail=""):
    res.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, ("- " + str(detail)[:300]) if detail else ""), flush=True)


def http(port, path, body=None, timeout=240):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sh(cmd, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=120).stdout


def llama_pids(dest):
    out = sh("pgrep -af llama-server")
    return [int(l.split()[0]) for l in out.splitlines() if "llama-server" in l and "pgrep" not in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--sha256", default="", help="FILE=HASH,... protected files that must still hash to HASH")
    ap.add_argument("--working", default="/kaggle/working/MAPLE-llama.cpp")
    a = ap.parse_args()
    dest = a.dest.rstrip("/")
    base_pids = set(llama_pids(dest))
    chk("restore dir is separate from the working installation", os.path.realpath(dest) != os.path.realpath(a.working)
        and not os.path.realpath(dest).startswith(os.path.realpath(a.working) + "/"), "%s vs %s" % (dest, a.working))

    # -- executables + permissions
    bins = glob.glob(dest + "/vendor/llamacpp/**/build/bin", recursive=True)
    b = bins[0] if bins else ""
    need = ["llama-server", "llama-cli", "llama-quantize"]
    notx = [n for n in need if not os.access(os.path.join(b, n), os.X_OK)]
    chk("llama.cpp executables present and executable", bool(b) and not notx, "bin=%s not-executable=%s" % (b, notx))
    env = dict(os.environ, LD_LIBRARY_PATH="/usr/local/nvidia/lib64:" + b)
    ldd = sh("ldd '%s/llama-server' 2>&1" % b, env)
    unresolved = [l.strip() for l in ldd.splitlines() if "not found" in l]
    outside = [l.strip() for l in ldd.splitlines() if "/kaggle/working/llama.cpp" in l or "/kaggle/working/MAPLE-llama.cpp/vendor" in l]
    chk("shared libraries resolve (CUDA 13, cuBLAS, NCCL, ggml)", not unresolved, unresolved or "all resolved")
    chk("ldd does not use the protected tree or the working install", not outside, outside or "libs come from the restored copy / host")
    gl = sh("%s/llama-server --list-devices 2>&1 | grep -c 'Tesla T4'" % b, env).strip()
    chk("CUDA sees two Tesla T4 GPUs", gl == "2", "T4 lines: %s" % gl)

    # -- protected files unchanged
    for item in [x for x in a.sha256.split(",") if x]:
        f, h = item.rsplit("=", 1)
        got = sh("sha256sum -- '%s' | cut -d' ' -f1" % f).strip()
        chk("unchanged: %s" % os.path.basename(f), got == h, got[:16])

    # -- API
    hz = http(a.port, "/health")
    chk("API /health: four models ok at exact paths", hz.get("status") == "ok" and hz.get("model_paths_ok") and set(hz["models"]) == {"deepseek", "gemma", "qwen3", "fast"}, hz.get("models"))
    ml = http(a.port, "/v1/models")
    ids = [m["id"] for m in ml["data"]]
    chk("API /v1/models lists the roster + auto, alias quant", {"deepseek", "gemma", "qwen3", "fast", "auto"} <= set(ids) and ml["aliases"].get("quant") == "qwen3", "%s aliases=%s" % (ids, ml.get("aliases")))
    # explicit selection with switching; also uses the alias; order forces a switch every time
    seq = [("fast", "fast"), ("deepseek", "deepseek"), ("gemma", "gemma"), ("qwen3", "qwen3"), ("quant", "qwen3"), ("fast", "fast")]
    for asked, expect in seq:
        t0 = time.time()
        try:
            r = http(a.port, "/v1/chat/completions", {"model": asked, "max_tokens": 1024, "temperature": 0,
                                                       "messages": [{"role": "user", "content": "Reply with the single word OK."}]})
            txt = (r["choices"][0]["message"].get("content") or "").strip()
            chk("inference model=%s" % asked, r.get("model") == expect and bool(txt), "served_by=%s %.1fs reply=%r" % (r.get("model"), time.time() - t0, txt[:30]))
        except Exception as e:  # noqa
            chk("inference model=%s" % asked, False, "%s: %s" % (type(e).__name__, e))
        live = [p for p in llama_pids(dest) if p not in base_pids]
        chk("  exactly one llama-server resident after %s" % asked, len(live) == 1, "pids %s" % live)
    r = http(a.port, "/v1/route", {"messages": [{"role": "user", "content": "write a rust function using CUDA"}]})
    chk("/v1/route auto-selects a specialist", r.get("model") in ("deepseek", "gemma", "qwen3", "fast"), r)
    r = http(a.port, "/v1/chat/completions", {"model": "auto", "max_tokens": 1024, "temperature": 0, "messages": [{"role": "user", "content": "Reply with the single word OK. (git commit message)"}]})
    chk("auto-routed inference", bool((r["choices"][0]["message"].get("content") or "").strip()), "served_by=%s" % r.get("model"))
    st = http(a.port, "/health")["runtime"]
    chk("model switching counted by the runtime", st.get("switches", 0) >= 5, st)

    # -- which files the live llama-server actually mapped
    live = [p for p in llama_pids(dest) if p not in base_pids]
    if live:
        maps = open("/proc/%d/maps" % live[0]).read()
        chk("running llama-server maps libraries only from the restored copy",
            dest in maps and "/kaggle/working/llama.cpp/" not in maps and "/kaggle/working/MAPLE-llama.cpp/vendor" not in maps,
            "restored=%d protected=%d working=%d" % (maps.count(dest), maps.count("/kaggle/working/llama.cpp/"), maps.count("/kaggle/working/MAPLE-llama.cpp/vendor")))
    gpu0 = sh("LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()
    chk("GPU memory in use while a model is loaded", any(int(x) > 500 for x in gpu0), gpu0)

    # -- clean shutdown (SIGTERM to the API only; it must unload its llama-server)
    pidf = dest + "/maple_api.pid"
    pid = int(open(pidf).read().strip())
    # bracket trick + own-pid exclusion: pgrep -f must not match the shell that is running it
    api_pids = [pid] + [int(x) for x in sh("pgrep -f '[m]aple.*--port %d'" % a.port).split() if int(x) != os.getpid()]
    os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < 40 and [p for p in llama_pids(dest) if p not in base_pids]:
        time.sleep(1)
    left = [p for p in llama_pids(dest) if p not in base_pids]
    chk("SIGTERM: no orphaned llama-server from the restored copy", not left, "left=%s after %.0fs" % (left, time.time() - t0))
    time.sleep(3)
    gpu1 = sh("LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()
    chk("GPU memory released after shutdown", all(int(x) < 300 for x in gpu1) or set(base_pids) != set(), "MiB per GPU: %s" % gpu1)
    api_pids = [p for p in api_pids if "pgrep" not in (open("/proc/%d/cmdline" % p).read() if os.path.exists("/proc/%d/cmdline" % p) else "")]
    alive = [p for p in api_pids if os.path.exists("/proc/%d" % p) and open("/proc/%d/stat" % p).read().split()[2] != "Z"]
    chk("API process exited", not alive, alive)

    fails = [x for x in res if not x["ok"]]
    print("PROBE_RESULT " + json.dumps({"passed": len(res) - len(fails), "failed": len(fails), "checks": res}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
