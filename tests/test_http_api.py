"""HTTP-level tests against the REAL `python -m maple serve` process over real sockets (no stub, no in-process handler).

Tier A  (always runs, no GPU, no models): model paths are pointed at nonexistent files through MAPLE_MODEL_<ID>.
        Covers health/degraded, model listing, routing, error responses, missing-model handling, auth behaviour, shutdown.
Tier B  (MAPLE_LIVE=1, needs the GPU host with the four models): real inference, streaming, alias, auto-routing,
        model switching, SIGTERM without orphaned llama-server processes.
"""
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = os.environ.get("MAPLE_LIVE") == "1"
POSIX = os.name == "posix"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(port, method, path, body=None, headers=None, raw=None, timeout=300):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    conn.request(method, path, data, h)
    r = conn.getresponse()
    payload = r.read()
    conn.close()
    try:
        obj = json.loads(payload)
    except ValueError:
        obj = None
    return r.status, obj, {k.lower(): v for k, v in r.getheaders()}, payload


def start_server(port, env_extra, home=None):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT / "src")
    if home:
        env["MAPLE_HOME"] = str(home)
    p = subprocess.Popen([sys.executable, "-s", "-m", "maple", "serve", "--host", "127.0.0.1", "--port", str(port)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 30:
        if p.poll() is not None:
            raise RuntimeError("maple serve exited early rc=%s" % p.returncode)
        try:
            request(port, "GET", "/health", timeout=3)
            return p
        except OSError:
            time.sleep(0.3)
    p.kill()
    raise RuntimeError("maple serve did not start")


def llama_pids():
    if not POSIX:
        return set()
    out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout.split()
    return {int(x) for x in out}


class TestRealServerNoModels(unittest.TestCase):
    """Tier A."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.port = free_port()
        cls.missing = {i: os.path.join(cls.tmp.name, "absent-%s.gguf" % i) for i in ("deepseek", "gemma", "qwen3", "fast")}
        cls.proc = start_server(cls.port, {"MAPLE_MODEL_" + k.upper(): v for k, v in cls.missing.items()}, home=cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
            cls.proc.wait(10)
        cls.tmp.cleanup()

    def test_health_reports_missing_models_with_503(self):
        st, o, _, _ = request(self.port, "GET", "/health")
        self.assertEqual(st, 503)
        self.assertEqual(o["status"], "degraded")
        self.assertFalse(o["model_paths_ok"])
        self.assertEqual(sorted(m["id"] for m in o["missing"]), ["deepseek", "fast", "gemma", "qwen3"])
        self.assertEqual({m["path"] for m in o["missing"]}, set(self.missing.values()))   # env override took effect
        self.assertEqual(o["runtime"]["loaded"], None)

    def test_models_listing_and_alias(self):
        st, o, _, _ = request(self.port, "GET", "/v1/models")
        self.assertEqual(st, 200)
        self.assertEqual(o["object"], "list")
        self.assertEqual([m["id"] for m in o["data"]], ["deepseek", "gemma", "qwen3", "fast", "auto"])
        self.assertEqual(o["aliases"], {"quant": "qwen3"})
        self.assertTrue(all(m["object"] == "model" for m in o["data"]))

    def test_route_endpoint_explicit_and_automatic(self):
        st, o, _, _ = request(self.port, "POST", "/v1/route", {"model": "quant", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((st, o["model"], o["explicit"]), (200, "qwen3", True))
        st, o, _, _ = request(self.port, "POST", "/v1/route", {"messages": [{"role": "user", "content": "Explain lock-free queues in C++20 with atomics"}]})
        self.assertEqual((st, o["model"], o["explicit"]), (200, "deepseek", False))
        st, o, _, _ = request(self.port, "POST", "/v1/route", {"model": "auto", "messages": [{"role": "user", "content": "write a regex for emails"}]})
        self.assertEqual((o["model"], o["explicit"]), ("fast", False))
        st, o, _, _ = request(self.port, "POST", "/v1/route", {"messages": [{"role": "user", "content": "good morning"}]})
        self.assertEqual(o["model"], "gemma")          # default when there is no specialist signal

    def test_invalid_model_is_404_with_openai_style_error(self):
        st, o, _, _ = request(self.port, "POST", "/v1/chat/completions", {"model": "gpt-4", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(st, 404)
        self.assertEqual(o["error"]["code"], "model_not_found")
        self.assertIn("valid: deepseek, gemma, qwen3, fast", o["error"]["message"])

    def test_missing_model_file_is_503_naming_the_exact_path(self):
        for name, mid in (("fast", "fast"), ("quant", "qwen3")):
            st, o, _, _ = request(self.port, "POST", "/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(st, 503, name)
            self.assertEqual(o["error"]["code"], "model_missing")
            self.assertIn(self.missing[mid], o["error"]["message"])
        st, o, _, _ = request(self.port, "POST", "/v1/completions", {"model": "deepseek", "prompt": "x"})
        self.assertEqual((st, o["error"]["code"]), (503, "model_missing"))

    def test_bad_requests(self):
        self.assertEqual(request(self.port, "POST", "/v1/chat/completions", {"model": "fast"})[0], 400)                 # no messages
        self.assertEqual(request(self.port, "POST", "/v1/chat/completions", raw=b"{not json")[0], 400)                   # invalid JSON
        self.assertEqual(request(self.port, "POST", "/v1/chat/completions", raw=b"")[0], 400)                            # empty body
        st, o, _, _ = request(self.port, "POST", "/v1/completions", {"model": "fast"})                                  # no prompt
        self.assertEqual(st, 400)
        self.assertIn("prompt", o["error"]["message"])
        self.assertEqual(request(self.port, "POST", "/v1/nope", {"a": 1})[0], 404)
        self.assertEqual(request(self.port, "GET", "/v1/nope")[0], 404)

    def test_no_authentication_is_enforced(self):
        """Documented behaviour: the API ignores Authorization. Any (or no) key is accepted - so never expose it publicly."""
        for h in ({}, {"Authorization": "Bearer maple-local"}, {"Authorization": "Bearer totally-wrong"}):
            st, _, _, _ = request(self.port, "GET", "/v1/models", headers=h)
            self.assertEqual(st, 200)

    @unittest.skipUnless(POSIX, "SIGTERM semantics are POSIX-only")
    def test_zz_sigterm_shuts_down_cleanly(self):
        self.proc.send_signal(signal.SIGTERM)
        self.assertEqual(self.proc.wait(15), 0)
        with self.assertRaises(OSError):
            request(self.port, "GET", "/health", timeout=2)


@unittest.skipUnless(LIVE and POSIX, "set MAPLE_LIVE=1 on the GPU host with the four models present")
class TestRealServerLive(unittest.TestCase):
    """Tier B."""

    @classmethod
    def setUpClass(cls):
        cls.base_pids = llama_pids()
        cls.port = free_port()
        cls.proc = start_server(cls.port, {})

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.send_signal(signal.SIGTERM)
            try:
                cls.proc.wait(30)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def chat(self, model, text="Reply with the single word OK.", **kw):
        body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1024, "temperature": 0}
        body.update(kw)
        return request(self.port, "POST", "/v1/chat/completions", body)

    def test_01_health_all_models_ok(self):
        st, o, _, _ = request(self.port, "GET", "/health")
        self.assertEqual((st, o["status"], o["model_paths_ok"]), (200, "ok", True))

    def test_02_explicit_chat_fast_has_content_and_maple_metadata(self):
        st, o, h, _ = self.chat("fast")
        self.assertEqual(st, 200)
        self.assertTrue(o["choices"][0]["message"]["content"].strip())
        self.assertEqual(h["x-maple-model"], "fast")
        self.assertEqual(o["x_maple"]["model"], "fast")
        self.assertTrue(o["x_maple"]["explicit"])

    def test_03_completions_endpoint(self):
        st, o, _, _ = request(self.port, "POST", "/v1/completions", {"model": "fast", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
        self.assertEqual(st, 200)
        self.assertTrue(o["choices"][0]["text"].strip())

    def test_04_streaming_sse(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": "fast", "stream": True, "max_tokens": 32, "temperature": 0,
                     "messages": [{"role": "user", "content": "Count from 1 to 5."}]}), {"Content-Type": "application/json"})
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("X-MAPLE-Model"), "fast")
        body = r.read().decode()
        conn.close()
        datas = [l[6:] for l in body.splitlines() if l.startswith("data: ")]
        self.assertGreater(len(datas), 2)
        self.assertEqual(datas[-1].strip(), "[DONE]")
        json.loads(datas[0])

    def test_05_switching_alias_and_automatic_routing(self):
        st, o, h, _ = self.chat("deepseek")
        self.assertEqual((st, h["x-maple-model"]), (200, "deepseek"))
        self.assertEqual(len(llama_pids() - self.base_pids), 1)               # exactly one model resident
        st, o, h, _ = self.chat("quant")
        self.assertEqual((st, h["x-maple-model"]), (200, "qwen3"))            # alias; Qwen3 thinks, hence the 1024-token budget
        self.assertTrue(o["choices"][0]["message"]["content"].strip())
        self.assertEqual(len(llama_pids() - self.base_pids), 1)
        st, o, h, _ = self.chat("auto", "Write a Rust function using atomics for a lock-free stack. Reply OK only.")
        self.assertEqual((st, h["x-maple-model"]), (200, "deepseek"))
        self.assertFalse(o["x_maple"]["explicit"])
        st, o, h, _ = self.chat("auto", "Write a regex for emails. Reply OK only.")
        self.assertEqual(h["x-maple-model"], "fast")
        sw = request(self.port, "GET", "/health")[1]["runtime"]["switches"]
        self.assertGreaterEqual(sw, 3)

    def test_06_invalid_model_does_not_disturb_the_loaded_model(self):
        before = request(self.port, "GET", "/health")[1]["runtime"]["loaded"]
        st, o, _, _ = self.chat("no-such-model")
        self.assertEqual((st, o["error"]["code"]), (404, "model_not_found"))
        self.assertEqual(request(self.port, "GET", "/health")[1]["runtime"]["loaded"], before)

    def test_zz_sigterm_leaves_no_llama_server(self):
        self.assertEqual(len(llama_pids() - self.base_pids), 1)
        self.proc.send_signal(signal.SIGTERM)
        self.assertEqual(self.proc.wait(60), 0)
        t0 = time.time()
        while time.time() - t0 < 20 and (llama_pids() - self.base_pids):
            time.sleep(0.5)
        self.assertEqual(llama_pids() - self.base_pids, set())


if __name__ == "__main__":
    unittest.main()
