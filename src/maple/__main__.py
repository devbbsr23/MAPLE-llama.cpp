"""python -m maple {serve,status,route,validate}"""
from __future__ import annotations

import argparse
import json
import sys

from .registry import Registry
from .router import Router


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maple")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the OpenAI-compatible API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    sub.add_parser("status", help="check model paths (no loading)")
    r = sub.add_parser("route", help="show which specialist a prompt routes to")
    r.add_argument("prompt")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .api import serve
        serve(a.host, a.port)
    elif a.cmd == "status":
        res = Registry().check_all()
        print(json.dumps(res, indent=2))
        return 0 if all(x["status"] == "ok" for x in res) else 2
    elif a.cmd == "route":
        d = Router().select(a.prompt)
        print(json.dumps({"model": d.model, "reason": d.reason, "scores": d.scores}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
