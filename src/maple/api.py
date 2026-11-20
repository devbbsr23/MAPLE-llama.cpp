"""OpenAI-compatible HTTP API (stdlib only) in front of llama-server.

Routes: GET /health, GET /v1/models, POST /v1/chat/completions,
POST /v1/completions, POST /v1/route (routing dry-run)."""
from __future__ import annotations

import http.client
import json
import logging
import signal
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import resource_manager as rm
from .config import MODEL_ROSTER, load_runtime_config, maple_home
from .registry import ModelMissing, Registry
from .router import Router, last_user_text
from .runtime import LoadError, ModelManager

log = logging.getLogger("maple.api")


def setup_logging(log_dir=None):
    d = log_dir or maple_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger("maple")
    root.setLevel(logging.INFO)
    if not root.handlers:
        for h in (logging.FileHandler(d / "maple.log"), logging.StreamHandler()):
            h.setFormatter(fmt)
            root.addHandler(h)


class App:
    def __init__(self, registry=None, manager=None, router=None, cfg=None):
        self.cfg = cfg or load_runtime_config()
        self.registry = registry or Registry()
        self.manager = manager or ModelManager(self.registry, self.cfg)
        self.router = router or Router(default=self.cfg.get("default_model", "gemma"))
        self.started = time.time()


def _err(status, msg, etype="invalid_request_error", code=None):
    return status, {"error": {"message": msg, "type": etype, "code": code}}


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "MAPLE/0.1"
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

        # -- helpers
        def _send(self, status, obj, extra=None):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                raise ValueError("empty request body")
            return json.loads(self.rfile.read(n))

        # -- GET
        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/health", "/healthz"):
                    return self._send(*self.health("deep=1" in self.path))
                if path == "/v1/models":
                    data = [{"id": i, "object": "model", "owned_by": "maple", "name": m["name"], "role": m["role"],
                             "specialties": m["specialties"]} for i, m in MODEL_ROSTER.items()]
                    data.append({"id": "auto", "object": "model", "owned_by": "maple", "name": "automatic specialist routing"})
                    return self._send(200, {"object": "list", "data": data, "aliases": app.registry.aliases()})
                return self._send(*_err(404, "not found: %s" % path, "not_found", "not_found"))
            except Exception as e:  # noqa
                log.exception("GET failed")
                return self._send(*_err(500, str(e), "server_error"))

        def health(self, deep):
            checks = app.registry.check_all()
            st = app.manager.status()
            ok = all(c["status"] == "ok" for c in checks)
            out = {"status": "ok" if ok else "degraded", "uptime_s": round(time.time() - app.started),
                   "models": {c["id"]: c["status"] for c in checks}, "model_paths_ok": ok,
                   "runtime": {k: st[k] for k in ("loaded", "inflight", "switches", "healthy")}}
            if not ok:
                out["missing"] = [{"id": c["id"], "path": c["path"], "status": c["status"]} for c in checks if c["status"] != "ok"]
            if deep:
                out["gpus"] = rm.gpu_info(app.cfg)
                out["ram"] = rm.ram_info()
            return 200 if ok else 503, out

        # -- POST
        def do_POST(self):
            t0 = time.time()
            try:
                body = self._body()
            except Exception as e:
                return self._send(*_err(400, "invalid JSON body: %s" % e))
            path = self.path.split("?")[0]
            try:
                if path == "/v1/route":
                    d = app.router.select(last_user_text(body.get("messages")) or str(body.get("prompt", "")), body.get("model"))
                    return self._send(200, {"model": d.model, "reason": d.reason, "scores": d.scores, "explicit": d.explicit})
                if path not in ("/v1/chat/completions", "/v1/completions"):
                    return self._send(*_err(404, "not found: %s" % path, "not_found", "not_found"))
                if path == "/v1/chat/completions":
                    if not body.get("messages"):
                        return self._send(*_err(400, "'messages' is required"))
                    text = last_user_text(body["messages"])
                else:
                    if "prompt" not in body:
                        return self._send(*_err(400, "'prompt' is required"))
                    text = body["prompt"] if isinstance(body["prompt"], str) else " ".join(map(str, body["prompt"]))
                try:
                    dec = app.router.select(text, body.get("model"))
                except KeyError as e:
                    return self._send(*_err(404, e.args[0], "model_not_found", "model_not_found"))
                m = MODEL_ROSTER[dec.model]
                body = dict(body)
                body["model"] = dec.model
                body.setdefault("temperature", m["temperature"])
                body.setdefault("max_tokens", m["max_tokens"])
                self.proxy(path, body, dec, m, t0)
            except ModelMissing as e:
                log.error("model missing: %s", e)
                self._send(*_err(503, str(e), "model_unavailable", "model_missing"))
            except LoadError as e:
                log.error("load error: %s", e)
                self._send(*_err(503, str(e), "model_load_failed", "load_failed"))
            except (BrokenPipeError, ConnectionResetError):
                log.warning("client disconnected")
            except Exception as e:  # noqa
                log.exception("POST failed")
                self._send(*_err(500, "%s: %s" % (type(e).__name__, e), "server_error"))

        def proxy(self, path, body, dec, m, t0):
            stream = bool(body.get("stream"))
            hdr = {"X-MAPLE-Model": dec.model, "X-MAPLE-Route": dec.reason}
            with app.manager.use(dec.model) as base:
                host, port = base.replace("http://", "").split(":")
                conn = http.client.HTTPConnection(host, int(port), timeout=600)
                conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                r = conn.getresponse()
                if stream and r.status == 200:
                    self.send_response(200)
                    self.send_header("Content-Type", r.getheader("Content-Type", "text/event-stream"))
                    for k, v in hdr.items():
                        self.send_header(k, v)
                    self.end_headers()
                    while True:
                        chunk = r.read1(8192) if hasattr(r, "read1") else r.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    raw = r.read()
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        return self._send(502, {"error": {"message": raw[:500].decode("utf-8", "replace"), "type": "upstream_error"}})
                    if r.status == 200 and isinstance(obj, dict):
                        obj["x_maple"] = {"model": dec.model, "route": dec.reason, "explicit": dec.explicit,
                                          "latency_s": round(time.time() - t0, 3)}
                    self._send(r.status, obj, hdr)
                conn.close()
            log.info("%s -> %s (%s) %.2fs", path, dec.model, dec.reason, time.time() - t0)

    return H


def serve(host=None, port=None, app: App | None = None):
    setup_logging()
    app = app or App()
    host = host or app.cfg["host"]
    port = port or app.cfg["api_port"]
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    log.info("MAPLE API listening on http://%s:%d", host, port)

    def _term(signum, frame):          # SIGTERM must unload the model too, or llama-server is orphaned on the GPU
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown requested")
    finally:
        app.manager.unload()
        httpd.server_close()
