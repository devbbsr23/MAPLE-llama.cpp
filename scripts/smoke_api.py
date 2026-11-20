#!/usr/bin/env python3
"""Check a running MAPLE API: /health, then one short chat completion per specialist (explicit selection),
then one auto-routed request. Standard library only. Exit 0 only if every check passes.

    python3 scripts/smoke_api.py --base http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def call(base, path, body=None, timeout=300):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--models", default="fast,qwen3,deepseek,gemma")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args(argv)
    ok = True
    try:
        h = call(a.base, "/health", timeout=30)
        print("health:", json.dumps(h)[:300])
    except Exception as e:  # noqa
        print("FAIL health: %s: %s" % (type(e).__name__, e))
        return 1
    for mid in a.models.split(","):
        t0 = time.time()
        try:
            r = call(a.base, "/v1/chat/completions", {"model": mid, "max_tokens": 1024, "temperature": 0,
                     "messages": [{"role": "user", "content": "Reply with the single word OK."}]}, a.timeout)
            msg = r["choices"][0]["message"]
            txt = (msg.get("content") or "").strip()
            think = (msg.get("reasoning_content") or "").strip()
            served = r.get("model")
            good = bool(txt)                                   # a real answer is required; thinking text alone is not a pass
            note = "" if txt else (" (only reasoning_content, %d chars: model spent its budget thinking)" % len(think) if think else " (empty)")
            print("%-4s %-9s %5.1fs served_by=%s reply=%r%s" % ("PASS" if good else "FAIL", mid, time.time() - t0, served, txt[:40], note))
            ok &= good
        except Exception as e:  # noqa
            print("FAIL %-9s %s: %s" % (mid, type(e).__name__, str(e)[:200]))
            ok = False
    try:
        r = call(a.base, "/v1/chat/completions", {"model": "auto", "max_tokens": 16,
                 "messages": [{"role": "user", "content": "Write a lock-free queue in C++20."}]}, a.timeout)
        print("PASS auto-route served_by=%s" % r.get("model"))
    except Exception as e:  # noqa
        print("FAIL auto-route %s: %s" % (type(e).__name__, str(e)[:200]))
        ok = False
    print("ALL SPECIALISTS OK" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
