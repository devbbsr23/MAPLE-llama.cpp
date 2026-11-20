"""HTTP API tests with a stub llama-server upstream (no GPU needed)."""
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maple.api import App, make_handler  # noqa: E402
from maple.config import MODEL_ROSTER  # noqa: E402
from maple.registry import ModelMissing, Registry  # noqa: E402
from maple.router import Router  # noqa: E402
from maple.runtime import LoadError  # noqa: E402


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = {"id": "stub", "object": "chat.completion", "model": body.get("model"),
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "stub-reply from " + str(body.get("model"))}, "finish_reason": "stop"}],
               "echo": {"temperature": body.get("temperature"), "max_tokens": body.get("max_tokens")}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class FakeManager:
    def __init__(self, base, fail=None):
        self.base, self.fail, self.loaded, self.history = base, fail, None, []

    @contextmanager
    def use(self, mid, ctx=None):
        if self.fail:
            raise self.fail
        self.history.append(mid)
        self.loaded = mid
        yield self.base

    def status(self):
        return {"loaded": self.loaded, "inflight": 0, "switches": len(self.history), "healthy": True, "last_load": {}}

    def unload(self):
        self.loaded = None


class FakeRegistry(Registry):
    def __init__(self, ok=True):
        super().__init__()
        self.ok = ok

    def check_all(self):
        return [{"id": i, "path": m["model_path"], "status": "ok" if self.ok else "missing"} for i, m in MODEL_ROSTER.items()]


class APITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d" % cls.stub.server_port
        cls.start_api(cls)

    def start_api(cls, mgr=None, reg=None):
        cls.mgr = mgr or FakeManager(cls.base)
        app = App(registry=reg or FakeRegistry(), manager=cls.mgr, router=Router(), cfg={"default_model": "gemma", "host": "127.0.0.1", "api_port": 0, "llama_port": 1})
        cls.api = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        threading.Thread(target=cls.api.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d" % cls.api.server_port

    @classmethod
    def tearDownClass(cls):
        cls.api.shutdown()
        cls.stub.shutdown()

    def req(self, path, body=None, raw=None, method=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        r = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read()), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), dict(e.headers)

    def chat(self, text, model=None):
        b = {"messages": [{"role": "user", "content": text}]}
        if model:
            b["model"] = model
        return self.req("/v1/chat/completions", b)

    def test_health_ok(self):
        s, h, _ = self.req("/health")
        self.assertEqual(s, 200)
        self.assertEqual(h["status"], "ok")
        self.assertEqual(set(h["models"]), {"deepseek", "gemma", "qwen3", "fast"})

    def test_models_list(self):
        s, d, _ = self.req("/v1/models")
        self.assertEqual(s, 200)
        ids = [m["id"] for m in d["data"]]
        self.assertEqual(ids, ["deepseek", "gemma", "qwen3", "fast", "auto"])
        self.assertEqual(d["aliases"], {"quant": "qwen3"})

    def test_explicit_selection(self):
        for m, expect in (("deepseek", "deepseek"), ("quant", "qwen3"), ("fast", "fast"), ("gemma", "gemma")):
            s, d, h = self.chat("hello", m)
            self.assertEqual(s, 200)
            self.assertEqual(h["X-MAPLE-Model"], expect)
            self.assertEqual(d["x_maple"]["explicit"], True)

    def test_auto_routing_and_defaults_injected(self):
        s, d, h = self.chat("implement a lock-free queue in C++20")
        self.assertEqual(h["X-MAPLE-Model"], "deepseek")
        self.assertEqual(d["echo"]["temperature"], MODEL_ROSTER["deepseek"]["temperature"])
        self.assertEqual(d["echo"]["max_tokens"], 4096)
        s, d, h = self.chat("git commit message please", "auto")
        self.assertEqual(h["X-MAPLE-Model"], "fast")
        self.assertEqual(d["echo"]["max_tokens"], 2048)

    def test_client_params_win(self):
        s, d, _ = self.req("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "model": "fast", "temperature": 0.9, "max_tokens": 7})
        self.assertEqual((d["echo"]["temperature"], d["echo"]["max_tokens"]), (0.9, 7))

    def test_route_dry_run(self):
        s, d, _ = self.req("/v1/route", {"messages": [{"role": "user", "content": "design a portfolio risk model"}]})
        self.assertEqual(d["model"], "qwen3")

    def test_errors(self):
        self.assertEqual(self.req("/v1/chat/completions", raw=b"{not json")[0], 400)
        self.assertEqual(self.req("/v1/chat/completions", {})[0], 400)
        s, d, _ = self.chat("x", "gpt-4")
        self.assertEqual((s, d["error"]["code"]), (404, "model_not_found"))
        self.assertEqual(self.req("/nope")[0], 404)
        self.assertEqual(self.req("/v1/completions", {"model": "fast"})[0], 400)

    def test_completions_endpoint(self):
        s, d, h = self.req("/v1/completions", {"prompt": "regex for emails", "model": "auto"})
        self.assertEqual((s, h["X-MAPLE-Model"]), (200, "fast"))


class APIFailureTest(APITest):
    @classmethod
    def setUpClass(cls):
        cls.stub = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d" % cls.stub.server_port

    def build(self, mgr, reg=None):
        APITest.start_api(self.__class__, mgr, reg)

    def tearDown(self):
        self.api.shutdown()

    def test_missing_model_gives_503_with_exact_path(self):
        self.build(FakeManager(self.base, fail=ModelMissing("fast", MODEL_ROSTER["fast"]["model_path"])))
        s, d, _ = self.chat("x", "fast")
        self.assertEqual((s, d["error"]["code"]), (503, "model_missing"))
        self.assertIn(MODEL_ROSTER["fast"]["model_path"], d["error"]["message"])

    def test_load_failure_gives_503(self):
        self.build(FakeManager(self.base, fail=LoadError("llama-server exited with code 1")))
        s, d, _ = self.chat("x", "fast")
        self.assertEqual((s, d["error"]["code"]), (503, "load_failed"))

    def test_health_degraded_lists_missing(self):
        self.build(FakeManager(self.base), FakeRegistry(ok=False))
        s, h, _ = self.req("/health")
        self.assertEqual((s, h["status"], len(h["missing"])), (503, "degraded", 4))

    test_health_ok = test_models_list = test_explicit_selection = test_auto_routing_and_defaults_injected = None
    test_client_params_win = test_route_dry_run = test_errors = test_completions_endpoint = None


if __name__ == "__main__":
    unittest.main()
