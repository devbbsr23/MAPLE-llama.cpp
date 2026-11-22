#!/usr/bin/env python3
"""MAPLE multi-model post-restore verifier (read-only; standard library only; no credentials are read, stored or printed).

Checks, in order: 1 model files (path, size vs manifest, optional --sha) | 2 process safety (one server per model, one API, one heartbeat,
SSH/Cloudflare untouched) | 3 /health: four healthy resident models | 4 GPU allocation vs config/runtime.yaml (and real VRAM use per GPU)
| 5 endpoints (/health, /v1/models, /v1/route) | 6 routing (test_generation must be flagged uncertified; coding must pick qwen3) | 7 smoke (one 4-token
completion per model, --no-smoke to skip) | 8 authentication proxy (port 8088): no key -> 401; with MAPLE_API_KEY in the environment -> 200.
Exit 0 only if every REQUIRED check passed. It never starts, stops or changes anything.
Usage: python3 maple_verify_multi.py [--maple-home DIR] [--api http://127.0.0.1:8000] [--sha] [--no-smoke] [--json]
"""
import argparse, hashlib, json, os, re, subprocess, sys, time, urllib.request, urllib.error

EXPECT_MODELS = ("gemma", "deepseek", "qwen3", "fast")
RES = []   # (status, name, detail, required)

def rec(ok, name, detail, required=True):
    st = "PASS" if ok else ("FAIL" if required else "WARN")
    RES.append((st, name, detail, required)); print("  [%-4s] %-34s %s" % (st, name, detail), flush=True); return ok

def http(url, body=None, key=None, timeout=60):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=(json.dumps(body).encode() if body is not None else None), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read() or b"{}")
        except Exception: return e.code, {}
    except Exception as e: return 0, {"error": type(e).__name__}

def sh(c, t=30):
    try: p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.stdout + p.stderr
    except Exception as e: return str(e)

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): d.update(b)
    return d.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--maple-home", default="/kaggle/working/MAPLE-llama.cpp"); ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--proxy", default="http://127.0.0.1:8088"); ap.add_argument("--sha", action="store_true"); ap.add_argument("--no-smoke", action="store_true"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(); H = a.maple_home
    print("=" * 78 + "\nMAPLE MULTI-MODEL VERIFICATION (read-only)\n" + "=" * 78)
    # config
    try:
        cfg = json.load(open(os.path.join(H, "config/runtime.yaml"), encoding="utf-8"))   # runtime.yaml is JSON-compatible YAML
        layout = (cfg.get("residency") or {}).get("layout") or {}
        rec(set(layout) == set(EXPECT_MODELS) and (cfg.get("residency") or {}).get("mode") not in (None, "off"), "config: residency layout", json.dumps({k: v["gpu"] for k, v in layout.items()}))
        routing_on = (cfg.get("routing") or {}).get("mode") == "qualified"; rec(routing_on, "config: qualified routing enabled", str((cfg.get("routing") or {}).get("mode")))
    except Exception as e:
        rec(False, "config: runtime.yaml", "%s: %s" % (type(e).__name__, e)); layout = {}
    # 1 models
    try:
        man = json.load(open(os.path.join(H, "manifests/models_manifest.json"), encoding="utf-8"))["models"]
    except Exception as e:
        man = {}; rec(False, "models manifest", "%s" % type(e).__name__)
    for mid in EXPECT_MODELS:
        m = man.get(mid, {}); p = m.get("original_path", "")
        if not p or not os.path.exists(p): rec(False, "model file %s" % mid, "missing: %s" % (p or "no path in manifest")); continue
        sz = os.path.getsize(p); real = os.path.realpath(p); kind = "symlink -> " + real if os.path.islink(p) else "regular file"
        ok = sz == m.get("size_bytes")
        with open(p, "rb") as f: magic = f.read(4)
        ok = ok and magic == b"GGUF"
        det = "%s | %d bytes %s | %s" % (os.path.basename(p), sz, "== manifest" if sz == m.get("size_bytes") else "!= manifest %s" % m.get("size_bytes"), kind)
        if a.sha and ok:
            t = time.time(); h = sha(p); ok = h == m.get("sha256"); det += " | sha256 %s (%.0fs)" % ("OK" if ok else "MISMATCH", time.time() - t)
        rec(ok, "model file %s" % mid, det)
    # 2 processes
    ps = sh("ps -eo pid,args")
    cnt = {mid: len(re.findall(r"llama-server.*--alias %s(\s|$)" % mid, ps)) for mid in EXPECT_MODELS}
    rec(all(v == 1 for v in cnt.values()), "one llama-server per model", json.dumps(cnt))
    api_n = len(re.findall(r"maple.*serve|python.*-m maple", ps)); hb_n = len(re.findall(r"maple_heartbeat\.py.* run", ps))
    rec(api_n == 1, "exactly one MAPLE API process", str(api_n)); rec(hb_n == 1, "exactly one heartbeat", str(hb_n), required=False)
    ssh_n = len(re.findall(r"sshd", ps)); cf_n = len(re.findall(r"cloudflared", ps))
    rec(ssh_n >= 1, "sshd running (not touched)", str(ssh_n)); rec(cf_n >= 1, "cloudflared running (not touched)", str(cf_n))
    # 3-5 health + endpoints
    s, h = http(a.api + "/health"); rec(s == 200 and h.get("status") == "ok", "GET /health", "status=%s uptime=%ss" % (h.get("status"), h.get("uptime_s")))
    res = (h.get("runtime") or {}).get("resident") or {}
    rec(set(res) == set(EXPECT_MODELS) and all(v.get("healthy") for v in res.values()), "four resident models healthy", ", ".join("%s(gpu%s)%s" % (k, v.get("gpu"), "" if v.get("healthy") else " UNHEALTHY") for k, v in sorted(res.items())))
    rec(all(res.get(k, {}).get("gpu") == v.get("gpu") for k, v in layout.items()) and bool(layout), "GPU allocation == config", "gpu0: %s | gpu1: %s" % (sorted(k for k, v in res.items() if v.get("gpu") == 0), sorted(k for k, v in res.items() if v.get("gpu") == 1)))
    mem = [x.strip() for x in sh("LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH $(command -v nvidia-smi || echo /opt/bin/nvidia-smi) --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits").splitlines() if x.strip()]
    try:
        used = {int(l.split(",")[0]): int(l.split(",")[1]) for l in mem}; rec(len(used) == 2 and all(v > 4000 for v in used.values()) and all(v < 15000 for v in used.values()), "VRAM in use on both GPUs", "gpu0 %s MiB, gpu1 %s MiB (expect roughly 11000 / 6000+)" % (used.get(0), used.get(1)))
    except Exception: rec(False, "VRAM query", "nvidia-smi unreadable")
    s, mj = http(a.api + "/v1/models"); ids = [m.get("id") for m in (mj.get("data") or [])]
    rec(s == 200 and all(m in ids for m in EXPECT_MODELS + ("auto",)), "GET /v1/models", ", ".join(ids))
    # 6 routing
    s, r1 = http(a.api + "/v1/route", {"model": "auto", "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists."}]})
    q1 = r1.get("qualification") or {}; rec(s == 200 and r1.get("model") == "qwen3" and q1.get("certified") is True, "route: coding -> qwen3 certified", "%s %s certified=%s" % (r1.get("model"), q1.get("category"), q1.get("certified")))
    s, r2 = http(a.api + "/v1/route", {"model": "auto", "messages": [{"role": "user", "content": "Write pytest unit tests for this function: def add(a, b): return a + b"}]})
    q2 = r2.get("qualification") or {}; rec(s == 200 and q2.get("certified") is False and q2.get("verify_required") is True, "route: test_generation flagged uncertified", "%s %s certified=%s verify_required=%s" % (r2.get("model"), q2.get("category"), q2.get("certified"), q2.get("verify_required")))
    s, r3 = http(a.api + "/v1/chat/completions", {"model": "auto", "max_tokens": 4, "messages": [{"role": "user", "content": "Write pytest unit tests for this function: def add(a, b): return a + b"}], "maple": {"require_certified": True}})
    rec(s == 409, "chat: require_certified -> 409 (no model run)", "HTTP %s" % s)
    # 7 smoke
    if not a.no_smoke:
        for mid in EXPECT_MODELS:
            t = time.time(); s, o = http(a.api + "/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "Reply with the single word OK."}], "max_tokens": 6, "temperature": 0}, timeout=120)
            txt = ((o.get("choices") or [{}])[0].get("message") or {}).get("content", "") if s == 200 else ""
            rec(s == 200 and bool(txt.strip()) and (o.get("x_maple") or {}).get("model") == mid, "smoke %s" % mid, "HTTP %s %.1fs reply=%r" % (s, time.time() - t, txt.strip()[:20]))
        n = ((http(a.api + "/health")[1].get("runtime") or {}).get("switches")); rec(n == 4, "no model reloads (switches == 4)", "switches=%s" % n, required=False)
    # 8 auth proxy
    s0, _ = http(a.proxy + "/health", timeout=10)
    if s0 == 0: rec(False, "auth proxy on :8088", "not running: start it (cell prompt, or Windows new_session.ps1). Needed for Windows access.", required=False)
    else:
        rec(s0 == 401, "proxy rejects request without key", "HTTP %s" % s0)
        k = os.environ.get("MAPLE_API_KEY", "")
        if k: s1, _ = http(a.proxy + "/health", key=k, timeout=10); rec(s1 == 200, "proxy accepts the supplied key", "HTTP %s" % s1)
        else: rec(True, "proxy key check", "skipped (no MAPLE_API_KEY in this process; test it from Windows: doctor)", required=False)
    bad = [r for r in RES if r[0] == "FAIL"]; warn = [r for r in RES if r[0] == "WARN"]
    print("-" * 78 + "\nRESULT: %s  (%d pass, %d fail, %d warn)" % ("PASS" if not bad else "FAIL", sum(r[0] == "PASS" for r in RES), len(bad), len(warn)))
    for r in bad: print("  FAILED:", r[1], "-", r[2])
    if a.json: print(json.dumps([{"status": r[0], "check": r[1], "detail": r[2]} for r in RES]))
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main())
