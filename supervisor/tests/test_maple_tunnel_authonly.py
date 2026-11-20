"""Unit + loopback integration tests for the --auth-only mode of maple_tunnel.py.

No MAPLE, no llama-server, no cloudflared, no Kaggle and no external network are used. The proxy is exercised over REAL HTTP on
127.0.0.1 (ephemeral ports) against a FAKE upstream. The test key below is a throw-away constant, not a real credential.
"""
import contextlib
import http.client
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import maple_tunnel as tn   # noqa: E402

KEY = "TEST-ONLY-" + "k" * 40                 # throw-away test constant (>= 24 printable ASCII)
SCRIPT = HERE.parent / "maple_tunnel.py"
FIVE = [("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"), ("POST", "/v1/route")]
POSIX = os.name == "posix"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(port, method, path, headers=None, body=None):
    """-> (status, headers_dict_lower, body_bytes). Never raises on HTTP errors."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        h = dict(headers or {})
        if body is not None:
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        c.close()


def bearer(k=KEY):
    return {"Authorization": "Bearer " + k}


class FakeUpstream:
    """Stands in for MAPLE on an ephemeral loopback port; records everything it receives."""
    def __init__(self):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass   # noqa: E704

            def _rec(self, body=b""):
                outer.seen.append({"method": self.command, "path": self.path, "body": body,
                                   "headers": {k.lower(): v for k, v in self.headers.items()}})

            def do_GET(self):
                self._rec()
                b = json.dumps({"ok": True, "path": self.path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self._rec(body)
                req = json.loads(body or b"{}")
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i in range(3):
                        self.wfile.write(b'data: {"i": %d}\n\n' % i)
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                b = json.dumps({"echo": req}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-MAPLE-Model", "fast")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class ProxyCase(unittest.TestCase):
    """Real AuthProxy in auth-only configuration in front of the fake upstream."""
    @classmethod
    def setUpClass(cls):
        cls.up = FakeUpstream()
        cls.events = []
        cls.proxy = tn.AuthProxy(KEY, 0, "127.0.0.1", cls.up.port, lambda n, **k: cls.events.append((n, k)), allowed=tn.ALLOWED_AUTH_ONLY)
        cls.proxy.start()
        cls.port = cls.proxy.port

    @classmethod
    def tearDownClass(cls):
        cls.proxy.stop()
        cls.up.close()

    def setUp(self):
        self.up.seen.clear()
        self.events.clear()


class ValidKey(ProxyCase):
    def test_health_models_with_valid_key(self):
        for path in ("/health", "/v1/models"):
            st, h, b = call(self.port, "GET", path, bearer())
            self.assertEqual(st, 200, path)
            self.assertEqual(json.loads(b)["path"], path)
        self.assertEqual([s["path"] for s in self.up.seen], ["/health", "/v1/models"])

    def test_chat_completions_body_forwarded_unchanged(self):
        req = {"model": "fast", "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 5}
        st, h, b = call(self.port, "POST", "/v1/chat/completions", bearer(), req)
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(b)["echo"], req)
        self.assertEqual(json.loads(self.up.seen[0]["body"]), req)
        self.assertEqual(h.get("x-maple-model"), "fast")          # upstream response headers are passed back

    def test_completions_and_route(self):
        for path in ("/v1/completions", "/v1/route"):
            st, _, b = call(self.port, "POST", path, bearer(), {"prompt": "hi"})
            self.assertEqual(st, 200, path)
            self.assertEqual(json.loads(b)["echo"], {"prompt": "hi"})

    def test_streaming_passthrough(self):
        st, _, b = call(self.port, "POST", "/v1/chat/completions", bearer(), {"stream": True})
        self.assertEqual(st, 200)
        self.assertEqual(b.count(b"data: "), 4)
        self.assertTrue(b.rstrip().endswith(b"[DONE]"))

    def test_query_string_preserved(self):
        st, _, b = call(self.port, "GET", "/health?deep=1", bearer())
        self.assertEqual(st, 200)
        self.assertEqual(self.up.seen[0]["path"], "/health?deep=1")

    def test_lowercase_scheme_is_accepted(self):
        self.assertEqual(call(self.port, "GET", "/health", {"Authorization": "bearer " + KEY})[0], 200)


class MissingOrWrongKey(ProxyCase):
    def test_every_endpoint_401_without_key(self):
        for method, path in FIVE:
            st, h, b = call(self.port, method, path, None, {} if method == "POST" else None)
            self.assertEqual(st, 401, (method, path))
            self.assertEqual(json.loads(b)["error"]["type"], "invalid_api_key")
            self.assertEqual(h.get("www-authenticate"), "Bearer")
        self.assertEqual(self.up.seen, [], "an unauthenticated request must never reach MAPLE")

    def test_wrong_keys_401(self):
        bad = ["wrong", KEY[:-1], KEY + "x", KEY[1:], KEY.lower(), KEY.upper(), " ", "", "Bearer " + KEY, KEY * 2, "k" * 5000]
        for b_ in bad:
            for method, path in FIVE:
                st, _, _ = call(self.port, method, path, bearer(b_), {} if method == "POST" else None)
                self.assertEqual(st, 401, (method, path, len(b_)))
        self.assertEqual(self.up.seen, [])

    def test_wrong_authorization_scheme_401(self):
        for hdr in ("Basic " + KEY, "Token " + KEY, "Digest " + KEY, "Bearer" + KEY, "Bearer", "Bearer ", KEY, "ApiKey " + KEY, ""):
            self.assertEqual(call(self.port, "GET", "/health", {"Authorization": hdr})[0], 401, repr(hdr))
        self.assertEqual(self.up.seen, [])

    def test_other_key_headers_do_not_authenticate(self):
        for hdr in ({"X-API-Key": KEY}, {"api-key": KEY}, {"Proxy-Authorization": "Bearer " + KEY}, {"X-Authorization": "Bearer " + KEY}):
            self.assertEqual(call(self.port, "GET", "/health", hdr)[0], 401, hdr)
        self.assertEqual(self.up.seen, [])

    def test_key_in_query_string_does_not_authenticate(self):
        self.assertEqual(call(self.port, "GET", "/health?api_key=" + KEY)[0], 401)
        self.assertEqual(call(self.port, "GET", "/health?key=" + KEY)[0], 401)

    def test_non_ascii_authorization_does_not_crash(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(b"GET /health HTTP/1.0\r\nAuthorization: Bearer caf\xc3\xa9\xff\xfe\r\n\r\n")
        data = s.recv(4096)
        s.close()
        self.assertTrue(data.startswith(b"HTTP/1.0 401"), data[:40])
        self.assertEqual(call(self.port, "GET", "/health", bearer())[0], 200)      # still alive and serving

    def test_rejection_is_reported_without_the_key(self):
        call(self.port, "GET", "/health", bearer("nope-" + "x" * 30))
        self.assertTrue(any(n == "auth_rejected" for n, _ in self.events))
        self.assertNotIn(KEY, json.dumps(self.events))


class DisallowedRequests(ProxyCase):
    def test_disallowed_paths_404_with_valid_key(self):
        for path in ("/healthz", "/metrics", "/", "/admin", "/v1", "/v1/models/fast", "/v1/embeddings", "/props", "/slots", "/completion",
                     "/v1/chat/completions/", "/health/", "/../health", "/v1/../health", "/HEALTH", "/v1/models%2f", "/.env",
                     "/health;x", "/health%00", "/v1/models#x", "/v1/models/.", "/health%2e", "/%68ealth"):
            st, _, _ = call(self.port, "GET", path, bearer())
            self.assertEqual(st, 404, path)
        self.assertEqual(self.up.seen, [])

    def test_what_is_checked_is_exactly_what_is_forwarded(self):
        """Path confusion: whatever the proxy lets through must reach MAPLE as exactly an allowed path (checked string == forwarded string).
        (Python's http.server collapses a leading '//' to '/', so '//health' is served as '/health' - safe, because that is what is forwarded.)"""
        targets = ["//health", "///health", "//v1/models", "/health\t", "/health\x00", "http://evil.example/health", "http://127.0.0.1:8000/health",
                   "*", "/health?x=/../admin", "/health?a=b#frag", "/v1/models?x=1&y=2"]
        for t in targets:
            self.up.seen.clear()
            s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
            s.sendall(("GET %s HTTP/1.0\r\nAuthorization: Bearer %s\r\n\r\n" % (t, KEY)).encode("latin-1"))
            data = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
            s.close()
            status = int(data.split(b" ", 2)[1])
            self.assertIn(status, (200, 400, 404), (t, status))
            for rec in self.up.seen:
                self.assertIn(rec["path"].split("?")[0], ("/health", "/v1/models"), (t, rec["path"]))
                self.assertFalse(rec["path"].startswith("//") or "://" in rec["path"].split("?")[0], (t, rec["path"]))
            if status == 200:
                self.assertEqual(len(self.up.seen), 1, t)
            else:
                self.assertEqual(self.up.seen, [], t)

    def test_get_only_and_post_only_endpoints(self):
        self.assertEqual(call(self.port, "POST", "/health", bearer(), {})[0], 404)
        self.assertEqual(call(self.port, "POST", "/v1/models", bearer(), {})[0], 404)
        self.assertEqual(call(self.port, "GET", "/v1/chat/completions", bearer())[0], 404)
        self.assertEqual(call(self.port, "GET", "/v1/route", bearer())[0], 404)
        self.assertEqual(self.up.seen, [])

    def test_other_methods_404_with_key_401_without(self):
        for method in ("DELETE", "PUT", "PATCH", "HEAD", "OPTIONS"):
            self.assertEqual(call(self.port, method, "/health", bearer(), b"{}" if method in ("PUT", "PATCH", "DELETE") else None)[0], 404, method)
            self.assertEqual(call(self.port, method, "/health")[0], 401, method)
        self.assertEqual(self.up.seen, [])

    def test_unauthenticated_disallowed_path_gets_401_not_404(self):
        self.assertEqual(call(self.port, "GET", "/admin")[0], 401)         # no path probing without a key
        self.assertEqual(call(self.port, "GET", "/healthz")[0], 401)

    def test_llama_server_endpoints_are_unreachable(self):
        for path in ("/props", "/slots", "/completion", "/tokenize", "/v1/chat/completions/../../props"):
            self.assertIn(call(self.port, "GET", path, bearer())[0], (404,))
        self.assertEqual(self.up.seen, [])

    def test_oversized_and_missing_length(self):
        st, _, _ = call(self.port, "POST", "/v1/chat/completions", {**bearer(), "Content-Length": str(tn.MAX_BODY + 1)}, b"")
        self.assertEqual(st, 413)
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(("POST /v1/chat/completions HTTP/1.0\r\nAuthorization: Bearer %s\r\n\r\n" % KEY).encode())
        self.assertTrue(s.recv(4096).startswith(b"HTTP/1.0 411"))
        s.close()


class EarlyRejectionIsAlwaysAnHttpResponse(ProxyCase):
    """Regression: replying 401/404 before reading the body, then closing, made the OS send a TCP reset that could destroy the response
    (measured on loopback: ~1 in 40 requests surfaced as ConnectionAbortedError instead of an HTTP status)."""
    def hammer(self, n, size, headers, path="/v1/chat/completions", method="POST"):
        got = {}
        for _ in range(n):
            try:
                st = call(self.port, method, path, headers, b"x" * size)[0]
            except OSError as e:
                st = type(e).__name__
            got[st] = got.get(st, 0) + 1
        return got

    def test_no_key_with_body_never_resets(self):
        self.assertEqual(self.hammer(120, 4096, None), {401: 120})

    def test_wrong_key_with_large_body_never_resets(self):
        self.assertEqual(self.hammer(60, 200_000, bearer("wrong-" + "w" * 30)), {401: 60})

    def test_disallowed_path_with_body_never_resets(self):
        self.assertEqual(self.hammer(80, 50_000, bearer(), path="/admin"), {404: 80})
        self.assertEqual(self.hammer(40, 50_000, bearer(), path="/health"), {404: 40})

    def test_body_bigger_than_the_drain_limit_still_gets_a_status_or_a_clean_close(self):
        got = self.hammer(5, tn.LINGER_MAX + 300_000, None)
        self.assertEqual(sum(got.values()), 5)
        self.assertTrue(set(got) <= {401, "ConnectionAbortedError", "ConnectionResetError", "BrokenPipeError"}, got)   # never anything else
        self.assertEqual(self.up.seen, [])

    def test_drain_is_bounded_in_time(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=15)
        s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nContent-Length: 1000000\r\n\r\nabc")     # promises 1 MB, sends 3 bytes, then stalls
        t0 = time.monotonic()
        data = s.recv(65536)
        self.assertTrue(data.startswith(b"HTTP/1.0 401"))
        while s.recv(65536):
            pass                                                                    # the proxy must give up and close on its own
        self.assertLess(time.monotonic() - t0, 12)
        s.close()
        self.assertEqual(call(self.port, "GET", "/health", bearer())[0], 200)       # and the proxy keeps serving


class HeaderStripping(ProxyCase):
    def test_authorization_is_not_forwarded(self):
        call(self.port, "GET", "/health", {**bearer(), "X-Trace": "abc"})
        call(self.port, "POST", "/v1/chat/completions", {**bearer(), "Proxy-Authorization": "Basic Zm9vOmJhcg=="}, {"a": 1})
        self.assertEqual(len(self.up.seen), 2)
        for rec in self.up.seen:
            for name in ("authorization", "proxy-authorization"):
                self.assertNotIn(name, rec["headers"])
            self.assertNotIn(KEY, json.dumps(rec["headers"]))
        self.assertEqual(self.up.seen[0]["headers"].get("x-trace"), "abc")           # ordinary headers still pass
        self.assertEqual(self.up.seen[1]["headers"].get("content-type"), "application/json")

    def test_key_never_appears_in_any_response(self):
        for hdrs in (bearer(), bearer("bad-" + "z" * 30), None):
            st, h, b = call(self.port, "GET", "/health", hdrs)
            self.assertNotIn(KEY.encode(), b)
            self.assertNotIn(KEY, json.dumps(h))

    def test_no_request_logging_to_stderr(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            call(self.port, "GET", "/health", bearer())
            call(self.port, "GET", "/health")
        self.assertEqual(err.getvalue(), "")


class UpstreamDown(unittest.TestCase):
    def test_502_with_key_401_without_when_maple_is_down(self):
        dead = free_port()
        p = tn.AuthProxy(KEY, 0, "127.0.0.1", dead, allowed=tn.ALLOWED_AUTH_ONLY)
        p.start()
        try:
            self.assertEqual(call(p.port, "GET", "/health", bearer())[0], 502)
            self.assertEqual(call(p.port, "GET", "/health")[0], 401)
            self.assertEqual(call(p.port, "GET", "/health", bearer("x" * 30))[0], 401)
        finally:
            p.stop()


class BindingAndSource(unittest.TestCase):
    def test_proxy_binds_loopback_only(self):
        p = tn.AuthProxy(KEY, 0, "127.0.0.1", 9, allowed=tn.ALLOWED_AUTH_ONLY)
        try:
            self.assertEqual(p.httpd.server_address[0], "127.0.0.1")
        finally:
            p.httpd.server_close()

    def test_not_reachable_on_a_non_loopback_address(self):
        try:
            lan = socket.gethostbyname(socket.gethostname())
        except OSError:
            self.skipTest("no resolvable local address")
        if lan.startswith("127."):
            self.skipTest("host resolves only to loopback")
        up = FakeUpstream()
        p = tn.AuthProxy(KEY, 0, "127.0.0.1", up.port, allowed=tn.ALLOWED_AUTH_ONLY)
        p.start()
        try:
            with self.assertRaises(OSError):
                socket.create_connection((lan, p.port), timeout=3).close()
        finally:
            p.stop()
            up.close()

    def test_bind_error_is_clear(self):
        with socket.socket() as hog:
            hog.bind(("127.0.0.1", 0))
            hog.listen(1)
            with self.assertRaises(tn.ProxyBindError) as cm:
                tn.AuthProxy(KEY, hog.getsockname()[1], "127.0.0.1", 9)
        self.assertNotIn(KEY, str(cm.exception))

    def test_source_uses_constant_time_compare_and_never_binds_all_interfaces(self):
        src = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        self.assertIn("hmac.compare_digest(h[7:].strip().encode(\"utf-8\"), key_b)", code)
        self.assertNotIn('"0.0.0.0"', code)
        self.assertNotIn("'0.0.0.0'", code)
        binds = [l.strip() for l in code.splitlines() if "ThreadingHTTPServer(" in l and "class " not in l]
        self.assertTrue(binds, "proxy server construction not found")
        for l in binds:
            self.assertIn('("127.0.0.1", ', l, l)                # the only bind address the proxy ever uses

    def test_allowed_set_is_exactly_the_five_endpoints(self):
        self.assertEqual(sorted(tn.ALLOWED_AUTH_ONLY), sorted(FIVE))
        self.assertNotIn(("GET", "/healthz"), tn.ALLOWED_AUTH_ONLY)
        self.assertIn(("GET", "/healthz"), tn.ALLOWED)                # public mode behaviour is unchanged


class KeyLoading(unittest.TestCase):
    def test_valid_key(self):
        self.assertEqual(tn.load_auth_only_key({"API_KEY": KEY}), KEY)

    def test_missing_key_refuses_and_never_generates(self):
        for env in ({}, {"API_KEY": ""}):
            with self.assertRaises(SystemExit) as cm:
                tn.load_auth_only_key(env)
            self.assertIn("never generates", str(cm.exception))

    def test_bad_keys_refused_without_echoing_them(self):
        for bad in ("SHORTSECRET", "has space in it " + "x" * 30, "tab\t" + "x" * 30, "new\nline" + "x" * 30, "caf\u00e9" + "x" * 30, "x" * 257):
            with self.assertRaises(SystemExit) as cm:
                tn.load_auth_only_key({"API_KEY": bad})
            self.assertNotIn(bad, str(cm.exception))
            self.assertNotIn("SHORTSECRET", str(cm.exception))

    def test_token_urlsafe_style_key_is_accepted(self):
        import secrets
        self.assertEqual(len(tn.load_auth_only_key({"API_KEY": secrets.token_urlsafe(32)})), 43)

    def test_public_mode_key_behaviour_unchanged(self):
        k, gen = tn.get_api_key({})
        self.assertTrue(gen and len(k) >= 24)
        self.assertEqual(tn.get_api_key({"API_KEY": KEY}), (KEY, False))


class CliAndConfig(unittest.TestCase):
    def test_parser(self):
        p = tn.build_parser()
        self.assertTrue(p.parse_args(["run", "--auth-only"]).auth_only)
        self.assertFalse(p.parse_args(["run", "--auth-only"]).public)
        self.assertFalse(p.parse_args(["run"]).auth_only)
        self.assertTrue(p.parse_args(["start", "--key-stdin"]).key_stdin)
        self.assertTrue(p.parse_args(["status", "--auth-only"]).auth_only)

    def test_public_and_auth_only_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            tn.build_parser().parse_args(["run", "--public", "--auth-only"])
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit):
            tn.TunnelConfig(public=True, auth_only=True).validate()

    def test_port_and_host_guards(self):
        tn.TunnelConfig(auth_only=True).validate()
        for kw in ({"auth_port": 8000}, {"llama_port": 8088}, {"llama_port": 8000}, {"api_host": "0.0.0.0"}, {"api_host": "10.0.0.5"}):
            with self.assertRaises(SystemExit, msg=str(kw)):
                tn.TunnelConfig(auth_only=True, **kw).validate()

    def test_defaults_are_the_approved_ports(self):
        c = tn.TunnelConfig(auth_only=True)
        self.assertEqual((c.api_host, c.api_port, c.auth_port, c.llama_port), ("127.0.0.1", 8000, 8088, 8081))
        self.assertFalse(tn.TunnelConfig().auth_only)

    def test_main_without_key_exits_and_binds_nothing(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("API_KEY", None)
            with self.assertRaises(SystemExit) as cm:
                tn.main(["run", "--auth-only", "--state-dir", d, "--auth-port", str(port)])
            self.assertIn("API_KEY", str(cm.exception))
            self.assertFalse(tn.tcp_listening("127.0.0.1", port))
            self.assertFalse((Path(d) / "tunnel.lock").exists())


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProxy:
    instances = []

    def __init__(self, key, listen_port, up_host, up_port, on_event=None, allowed=None):
        self.key, self.port, self.up, self.allowed, self.on_event = key, listen_port, (up_host, up_port), allowed, on_event
        self.started = self.stopped = False
        self._alive = True
        FakeProxy.instances.append(self)

    def start(self):
        self.started = True

    def alive(self):
        return self._alive

    def stop(self):
        self.stopped, self._alive = True, False


class Edge:
    """Scripted http(): behaves like a correct proxy on the auth port unless `enforce` is False."""
    def __init__(self, auth_port=8088, api_port=8000):
        self.auth_port, self.api_port, self.enforce, self.api_up = auth_port, api_port, True, True
        self.calls = []

    def __call__(self, url, timeout, headers=None):
        auth = (headers or {}).get("Authorization", "")
        self.calls.append((url, auth))
        if url.startswith("http://127.0.0.1:%d" % self.auth_port):
            if not self.enforce or auth == "Bearer " + KEY:
                return tn.Reply(200 if self.api_up else 502, b"{}", None)
            return tn.Reply(401, b"{}", None)
        if url.startswith("http://127.0.0.1:%d" % self.api_port):
            return tn.Reply(200, json.dumps({"status": "ok", "models": {}}).encode(), None) if self.api_up else tn.Reply(None, b"", "unreachable")
        raise AssertionError("unexpected URL " + url)


class Harness:
    def __init__(self, tc, run_for=130, procs=None, **cfg_over):
        self.tmp = tempfile.TemporaryDirectory()
        tc.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "state"
        self.cfg = tn.TunnelConfig(state_dir=self.state, auth_only=True, check_interval=60, stable_after=300, backoff_base=15, **cfg_over)
        self.clock, self.edge, self.out = FakeClock(), Edge(), []
        self.popens, self.whiches, self.kills = [], [], []
        FakeProxy.instances = []
        self.stop_at = self.clock.t + run_for
        self.mgr = tn.TunnelManager(
            self.cfg, KEY, http=self.edge, popen=self._popen, which=self._which, clock=self.clock, sleep=self._sleep,
            list_procs=lambda: procs or [], alive=lambda p: False, cmdline=lambda p: None, proxy_factory=FakeProxy, out=self.out.append)
        tc.addCleanup(self.mgr.close)

    def _popen(self, *a, **k):
        self.popens.append(a)
        raise AssertionError("auth-only must never launch a process")

    def _which(self, name):
        self.whiches.append(name)
        raise AssertionError("auth-only must never look for cloudflared")

    def _sleep(self, s):
        self.clock.sleep(s)
        if self.clock.t >= self.stop_at:
            self.mgr.stopping = True

    def run(self):
        with mock.patch.object(tn.os, "kill", side_effect=lambda *a: self.kills.append(a)):
            return self.mgr.run_auth_only()

    @property
    def printed(self):
        return "\n".join(self.out)

    def state_json(self):
        return json.loads((self.state / tn.STATE_FILE).read_text())

    def all_text(self):
        parts = [self.printed]
        for f in self.state.rglob("*"):
            if f.is_file():
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)


class ManagerStartup(unittest.TestCase):
    def test_start_verify_run_stop_without_cloudflare_or_processes(self):
        h = Harness(self)
        rc = h.run()
        self.assertEqual(rc, 0)
        self.assertEqual((h.popens, h.whiches, h.kills), ([], [], []))        # no cloudflared, no subprocess, no signal sent
        self.assertEqual(len(FakeProxy.instances), 1)
        px = FakeProxy.instances[0]
        self.assertTrue(px.started and px.stopped)
        self.assertEqual((px.port, px.up), (8088, ("127.0.0.1", 8000)))
        self.assertEqual(px.allowed, tn.ALLOWED_AUTH_ONLY)
        self.assertEqual(h.state_json()["state"], "stopped")                   # clean shutdown state
        self.assertFalse((h.state / "tunnel.lock").exists())

    def test_self_test_requests_are_correct(self):
        h = Harness(self)
        h.run()
        first = [c for c in h.edge.calls if c[0].startswith("http://127.0.0.1:8088")][:4]
        self.assertEqual([bool(a) for _, a in first], [False, True, True, True])    # no key, wrong key, wrong scheme, right key
        self.assertTrue(first[1][1].startswith("Bearer wrong-"))
        self.assertTrue(first[2][1].startswith("Basic "))
        self.assertEqual(first[3][1], "Bearer " + KEY)

    def test_state_file_and_report_fields(self):
        h = Harness(self, run_for=10)
        px = []
        orig = h.mgr.write_state

        def spy(state=None):
            orig(state)
            if h.mgr.state == "AUTH_PROXY":
                px.append(h.state_json())
        h.mgr.write_state = spy
        h.run()
        st = px[0]
        self.assertEqual((st["mode"], st["state"], st["auth"], st["cloudflare"]), ("auth-only", "auth_proxy", "bearer-proxy-verified", "not used"))
        self.assertEqual((st["listen"], st["upstream"], st["url"]), ("127.0.0.1:8088", "127.0.0.1:8000", "http://127.0.0.1:8088/v1"))
        self.assertIsNone(st["cloudflared_pid"])

    def test_key_is_never_printed_logged_or_saved(self):
        h = Harness(self, run_for=700)                      # long enough for two 5-minute reports
        h.run()
        text = h.all_text()
        self.assertNotIn(KEY, text)
        self.assertIn("never printed, logged or saved", h.printed)
        self.assertIn("MAPLE AUTH PROXY", h.printed)
        self.assertIn("Auth proxy uptime", h.printed)

    def test_banner_gives_the_loopback_ssh_instructions(self):
        h = Harness(self, run_for=5)
        h.run()
        self.assertIn("ssh -N -L 8088:127.0.0.1:8088 kaggle2", h.printed)
        self.assertIn("http://127.0.0.1:8088/v1", h.printed)
        self.assertIn("Cloudflare", h.printed)

    def test_broken_proxy_that_does_not_enforce_is_stopped_and_reported(self):
        h = Harness(self)
        h.edge.enforce = False
        rc = h.run()
        self.assertEqual(rc, 4)
        self.assertTrue(FakeProxy.instances[0].stopped)
        self.assertEqual(h.state_json()["state"], "stopped")
        self.assertEqual(h.state_json()["auth"], "none")                       # and it never advertised itself as protected
        self.assertIn("self-test failed", h.printed)
        self.assertNotIn(KEY, h.all_text())

    def test_bind_failure_returns_6_and_leaves_nothing_behind(self):
        h = Harness(self)

        def boom(*a, **k):
            raise tn.ProxyBindError("cannot bind 127.0.0.1:8088 (Address already in use)")
        h.mgr.proxy_factory = boom
        self.assertEqual(h.run(), 6)
        self.assertIn("could not bind", h.printed)
        self.assertFalse((h.state / "tunnel.lock").exists())

    def test_maple_down_at_start_does_not_prevent_the_proxy(self):
        h = Harness(self, run_for=70)
        h.edge.api_up = False
        self.assertEqual(h.run(), 0)
        self.assertEqual(h.kills, [])
        self.assertEqual(h.popens, [])
        self.assertTrue(FakeProxy.instances[0].started)                       # proxy runs even though MAPLE is down
        self.assertIs(h.mgr.upstream_ok, False)                               # recorded as "MAPLE not answering", not as a proxy failure
        self.assertEqual(len(FakeProxy.instances), 1)                         # and the proxy was not restarted because of it

    def test_existing_ssh_cloudflared_is_reported_protected_and_untouched(self):
        ssh_cf = tn.ProcInfo(1971, "cloudflared", ("/usr/local/bin/cloudflared", "tunnel", "--url", "ssh://localhost:2222"))
        h = Harness(self, run_for=5, procs=[ssh_cf])
        h.run()
        self.assertIn("1971", h.printed)
        self.assertIn("protected", h.printed)
        self.assertEqual(h.kills, [])

    def test_leftover_public_mode_cloudflared_is_never_signalled(self):
        h = Harness(self, run_for=5)
        h.state.mkdir(parents=True, exist_ok=True)
        (h.state / "tunnel.lock").write_text(json.dumps({"pid": 424242}))            # stale lock from a crashed public-mode manager
        (h.state / tn.STATE_FILE).write_text(json.dumps({"mode": "public", "cloudflared_pid": 1971}))
        h.mgr.alive = lambda p: True if p == 1971 else False
        h.mgr.cmdline = lambda p: "cloudflared tunnel --url http://127.0.0.1:8088" if p == 1971 else None
        self.assertEqual(h.run(), 0)
        self.assertEqual(h.kills, [], "auth-only must not signal any cloudflared, even one that looks like a leftover of public mode")

    def test_lock_prevents_a_second_manager(self):
        h = Harness(self, run_for=5)
        h.state.mkdir(parents=True, exist_ok=True)
        (h.state / "tunnel.lock").write_text(json.dumps({"pid": os.getppid()}))
        h.mgr.alive = lambda p: p == os.getppid()
        h.mgr.cmdline = lambda p: "python maple_tunnel.py run --auth-only"
        with self.assertRaises(tn.DuplicateTunnelManager):
            h.run()
        self.assertEqual(FakeProxy.instances, [])


class ManagerSupervision(unittest.TestCase):
    def ready(self):
        h = Harness(self)
        h.mgr.start_proxy_auth_only()
        h.mgr.auth_verified, h.mgr.state = True, "AUTH_PROXY"
        return h

    def test_healthy_tick(self):
        h = self.ready()
        self.assertEqual(h.mgr.tick_auth_only(), "healthy")
        self.assertTrue(h.mgr.auth_verified)
        self.assertEqual(h.state_json()["auth"], "bearer-proxy-verified")

    def test_maple_down_is_reported_but_not_a_proxy_failure(self):
        h = self.ready()
        h.edge.api_up = False
        self.assertEqual(h.mgr.tick_auth_only(), "healthy")           # 401 probe still proves the key is enforced
        self.assertEqual(len(FakeProxy.instances), 1)                 # no restart
        self.assertIn("MAPLE not answering", h.state_json()["detail"])
        h.edge.api_up = True
        h.mgr.tick_auth_only()
        self.assertTrue(h.mgr.upstream_ok)

    def test_dead_proxy_is_restarted_and_reverified(self):
        h = self.ready()
        FakeProxy.instances[0]._alive = False
        self.assertEqual(h.mgr.tick_auth_only(), "recover:recovered")
        self.assertEqual(len(FakeProxy.instances), 2)
        self.assertTrue(FakeProxy.instances[1].started)
        self.assertEqual(FakeProxy.instances[1].allowed, tn.ALLOWED_AUTH_ONLY)
        self.assertEqual(h.state_json()["state"], "auth_proxy")

    def test_backoff_and_restart_limit(self):
        h = self.ready()
        outcomes = []
        for _ in range(7):
            FakeProxy.instances[-1]._alive = False
            outcomes.append(h.mgr.tick_auth_only())
            h.clock.t += 400
        self.assertEqual(outcomes[:5], ["recover:recovered"] * 5)
        self.assertEqual(outcomes[5], "recover:limit")
        self.assertEqual(len(FakeProxy.instances), 6)                 # 1 original + exactly 5 restarts

    def test_backoff_blocks_an_immediate_second_restart(self):
        h = self.ready()
        FakeProxy.instances[-1]._alive = False
        self.assertEqual(h.mgr.tick_auth_only(), "recover:recovered")
        FakeProxy.instances[-1]._alive = False
        self.assertEqual(h.mgr.tick_auth_only(), "recover:backoff")   # no clock advance
        self.assertEqual(len(FakeProxy.instances), 2)

    def test_proxy_that_stops_enforcing_is_shut_down_as_insecure(self):
        h = self.ready()
        h.edge.enforce = False
        self.assertEqual(h.mgr.tick_auth_only(), "insecure")
        self.assertTrue(FakeProxy.instances[0].stopped)
        self.assertTrue(h.mgr.insecure)
        self.assertEqual(h.state_json()["auth"], "none")
        self.assertEqual(h.mgr.tick_auth_only(), "insecure")          # stays failed, never restarts itself into an open state

    def test_restart_that_comes_back_open_is_refused(self):
        h = self.ready()
        FakeProxy.instances[0]._alive = False
        h.edge.enforce = False
        self.assertEqual(h.mgr.tick_auth_only(), "recover:insecure")
        self.assertTrue(FakeProxy.instances[1].stopped)

    def test_rejections_are_counted_but_logged_sparingly(self):
        h = self.ready()
        for _ in range(25):
            h.mgr._on_proxy_event("auth_rejected")
        h.mgr._on_proxy_event("proxied", path="/health")
        self.assertEqual(h.mgr.rejected, 25)
        log = (h.state / "tunnel_manager.log").read_text() if (h.state / "tunnel_manager.log").exists() else ""
        self.assertLessEqual(log.count("proxy_auth_rejected"), 3)         # 1st and 10th only


class StatusCommand(unittest.TestCase):
    def run_status(self, addrs, unauth=401, state_mode="auth-only", as_json=True):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / tn.STATE_FILE).write_text(json.dumps({"mode": state_mode, "state": "auth_proxy", "auth": "bearer-proxy-verified"}))
            cfg = tn.TunnelConfig(state_dir=Path(d), auth_only=True)

            def fake_http(url, timeout, headers=None):
                self.assertFalse(headers, "status must probe WITHOUT a key")
                return tn.Reply(unauth if ":8088" in url else 200, b"{}", None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), mock.patch.object(tn, "list_processes", return_value=[]):
                self.assertEqual(tn.cmd_status(cfg, as_json, http=fake_http, addresses=lambda p: addrs), 0)
            return json.loads(buf.getvalue()) if as_json else buf.getvalue()

    def test_loopback_only_and_enforced(self):
        out = self.run_status(["127.0.0.1"])
        ap = out["auth_proxy"]
        self.assertTrue(ap["loopback_only"] and ap["auth_enforced"])
        self.assertEqual((ap["unauthenticated_request_status"], ap["maple_api_status"], ap["cloudflare"]), (401, 200, "not used"))

    def test_wildcard_listener_is_flagged_as_exposed(self):
        self.assertFalse(self.run_status(["0.0.0.0"])["auth_proxy"]["loopback_only"])
        self.assertFalse(self.run_status(["127.0.0.1", "192.168.1.5"])["auth_proxy"]["loopback_only"])
        self.assertIn("EXPOSED", self.run_status(["0.0.0.0"], as_json=False))

    def test_unknown_and_not_listening(self):
        self.assertIsNone(self.run_status(None)["auth_proxy"]["loopback_only"])
        self.assertFalse(self.run_status([])["auth_proxy"]["loopback_only"])

    def test_open_proxy_is_reported_as_not_enforced(self):
        self.assertFalse(self.run_status(["127.0.0.1"], unauth=200)["auth_proxy"]["auth_enforced"])
        self.assertIn("NO / not answering", self.run_status(["127.0.0.1"], unauth=200, as_json=False))

    def test_text_mode(self):
        t = self.run_status(["127.0.0.1"], as_json=False)
        for s in ("Mode          : auth-only", "loopback only: yes", "Key enforced  : yes", "MAPLE API     : HTTP 200"):
            self.assertIn(s, t)

    def test_ssh_mode_status_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), mock.patch.object(tn, "list_processes", return_value=[]):
                tn.cmd_status(tn.TunnelConfig(state_dir=Path(d)), True)
            self.assertNotIn("auth_proxy", json.loads(buf.getvalue()))


class ProcNetParsing(unittest.TestCase):
    def make(self, tcp_rows=(), tcp6_rows=()):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        hdr = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        row = "   %d: %s 00000000:0000 %s 00000000:00000000 00:00000000 00000000  1000        0 12345 1 0 100 0 0 10 0\n"
        for name, rows in (("tcp", tcp_rows), ("tcp6", tcp6_rows)):
            (Path(d.name) / name).write_text(hdr + "".join(row % (i, a, st) for i, (a, st) in enumerate(rows)))
        return d.name

    def test_ipv4_loopback_wildcard_and_state_filter(self):
        p = self.make(tcp_rows=[("0100007F:1F98", "0A"), ("00000000:1F98", "01"), ("00000000:1F90", "0A")])
        self.assertEqual(tn.listening_addresses(8088, p), ["127.0.0.1"])       # other ports and non-LISTEN rows ignored
        self.assertEqual(tn.listening_addresses(8080, p), ["0.0.0.0"])

    def test_ipv6(self):
        p = self.make(tcp6_rows=[("00000000000000000000000001000000:1F98", "0A"), ("0000000000000000FFFF00000100007F:1F98", "0A")])
        addrs = tn.listening_addresses(8088, p)
        self.assertIn("::1", addrs)
        self.assertTrue(all(tn.is_loopback_ip(a) for a in addrs), addrs)
        p2 = self.make(tcp6_rows=[("00000000000000000000000000000000:1F98", "0A")])
        self.assertEqual(tn.listening_addresses(8088, p2), ["::"])
        self.assertFalse(tn.is_loopback_ip("::"))

    def test_loopback_helper(self):
        for ip, want in (("127.0.0.1", True), ("127.5.5.5", True), ("::1", True), ("::ffff:127.0.0.1", True), ("0.0.0.0", False),
                         ("192.168.0.2", False), ("::", False), ("garbage", False), ("::ffff:10.0.0.1", False)):
            self.assertEqual(tn.is_loopback_ip(ip), want, ip)

    def test_unavailable_proc(self):
        self.assertIsNone(tn.listening_addresses(8088, "/definitely/not/here"))


class StopCommand(unittest.TestCase):
    def run_stop(self, lock_pid, cmdline, mode="auth-only", survives=False):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tunnel.lock").write_text(json.dumps({"pid": lock_pid}))
            (Path(d) / tn.STATE_FILE).write_text(json.dumps({"mode": mode}))
            cfg = tn.TunnelConfig(state_dir=Path(d), auth_only=True)
            kills, alive = [], {"v": True}

            def fake_kill(pid, sig):
                kills.append((pid, sig))
                alive["v"] = survives
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), mock.patch.object(tn, "pid_alive", side_effect=lambda p: alive["v"] and p == lock_pid), \
                    mock.patch.object(tn, "cmdline_of", return_value=cmdline), mock.patch.object(tn.os, "kill", side_effect=fake_kill), \
                    mock.patch.object(tn.time, "sleep"), mock.patch.object(tn, "tcp_listening", return_value=False), \
                    mock.patch.object(tn.time, "monotonic", side_effect=iter([0, 1, 2, 3, 4, 5, 100, 101, 102, 103])):
                rc = tn.cmd_stop(cfg)
            return rc, kills, buf.getvalue()

    def test_stops_only_the_manager_with_sigterm(self):
        rc, kills, out = self.run_stop(7777, "python3 -B /x/maple_tunnel.py run --auth-only")
        self.assertEqual((rc, kills), (0, [(7777, signal.SIGTERM)]))
        self.assertIn("only its own auth proxy", out)
        self.assertIn("MAPLE, llama-server, the heartbeat and SSH access are not touched", out)
        self.assertIn("port 8088 listening: False", out)

    def test_never_signals_the_kaggle_ssh_cloudflared(self):
        rc, kills, out = self.run_stop(1971, "/usr/local/bin/cloudflared tunnel --url ssh://localhost:2222")
        self.assertEqual(kills, [])
        self.assertIn("nothing was signalled", out)

    def test_never_signals_the_heartbeat_or_maple(self):
        for cmd in ("python3 -B /x/maple_heartbeat.py run", "python3 -s -m maple serve --host 127.0.0.1 --port 8000", "llama-server -m x.gguf"):
            self.assertEqual(self.run_stop(4044, cmd)[1], [], cmd)

    def test_no_lock_means_no_signal(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(tn.os, "kill") as k:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(tn.cmd_stop(tn.TunnelConfig(state_dir=Path(d))), 0)
            k.assert_not_called()

    def test_never_escalates_to_sigkill(self):
        rc, kills, out = self.run_stop(7777, "python3 maple_tunnel.py run --auth-only", survives=True)
        self.assertEqual(rc, 1)
        self.assertEqual(kills, [(7777, signal.SIGTERM)])
        self.assertIn("not forcing it", out)


class StartCommand(unittest.TestCase):
    class P:
        pid = 5150

        def __init__(self, rc=None):
            self.rc = rc

        def poll(self):
            return self.rc

    LOCK_CMD = "python3 -B /x/maple_tunnel.py run --auth-only"

    def run_start(self, env_key=KEY, stdin_text=None, key_stdin=False, proc_rc=None, ready=True, lock=None, lock_cmd=LOCK_CMD):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        state = Path(d.name)
        cfg = tn.TunnelConfig(state_dir=state, auth_only=True)
        if lock:
            (state / "tunnel.lock").write_text(json.dumps({"pid": lock}))
        seen = {}

        def fake_popen(cmd, **kw):
            seen["cmd"], seen["kw"] = cmd, kw
            if ready:
                (state / tn.STATE_FILE).write_text(json.dumps({"mode": "auth-only", "state": "auth_proxy", "manager_pid": 5150}))
            return self.P(proc_rc)
        buf = io.StringIO()
        env = {"PATH": os.environ.get("PATH", "")}
        if env_key:
            env["API_KEY"] = env_key
        with contextlib.redirect_stdout(buf), mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(tn, "pid_alive", side_effect=lambda p: p == lock), \
                mock.patch.object(tn, "cmdline_of", return_value=lock_cmd):      # hermetic: never read the real /proc of the test host
            rc = tn.cmd_start(cfg, key_stdin, 5, popen=fake_popen, stdin=io.StringIO(stdin_text) if stdin_text is not None else None,
                              sleep=lambda s: None, clock=iter(range(0, 1000)).__next__)
        return rc, seen, buf.getvalue(), state

    def test_key_goes_to_child_environment_only(self):
        rc, seen, out, state = self.run_start()
        self.assertEqual(rc, 0)
        self.assertNotIn(KEY, " ".join(seen["cmd"]))                 # never on the command line (visible in ps)
        self.assertEqual(seen["kw"]["env"]["API_KEY"], KEY)
        self.assertEqual(seen["cmd"][seen["cmd"].index("run") + 1], "--auth-only")
        self.assertNotIn("--public", seen["cmd"])
        self.assertTrue(seen["kw"]["start_new_session"])
        self.assertNotIn(KEY, out)
        self.assertIn("Cloudflare not used", out)

    def test_key_from_stdin(self):
        rc, seen, out, _ = self.run_start(env_key=None, stdin_text=KEY + "\n", key_stdin=True)
        self.assertEqual(rc, 0)
        self.assertEqual(seen["kw"]["env"]["API_KEY"], KEY)
        self.assertNotIn(KEY, out + " ".join(seen["cmd"]))

    def test_missing_or_bad_key_refuses_before_launching(self):
        for kwargs in ({"env_key": None}, {"env_key": "short"}, {"env_key": None, "stdin_text": "\n", "key_stdin": True}):
            with self.assertRaises(SystemExit):
                self.run_start(**kwargs)

    def test_already_running_returns_3_and_launches_nothing(self):
        rc, seen, out, _ = self.run_start(lock=os.getppid())
        self.assertEqual((rc, seen), (3, {}))

    def test_lock_held_by_an_unrelated_live_process_is_stale_not_a_duplicate(self):
        # a live pid whose command line is NOT maple_tunnel (pid reuse after a crash) must not block a start - and is never signalled
        with mock.patch.object(tn.os, "kill") as k:
            rc, seen, out, _ = self.run_start(lock=os.getppid(), lock_cmd="bash -l")
        self.assertEqual(rc, 0)
        self.assertIn("cmd", seen)                  # the launch did happen
        k.assert_not_called()                       # and nothing was signalled

    def test_child_that_exits_is_reported_with_exit_code_and_no_key(self):
        rc, seen, out, state = self.run_start(proc_rc=6, ready=False)
        self.assertEqual(rc, 5)
        self.assertIn("did NOT start (exit code 6)", out)
        self.assertNotIn(KEY, out)

    def test_console_log_path_and_detachment(self):
        rc, seen, out, state = self.run_start()
        self.assertTrue((state / tn.CONSOLE_LOG).exists())
        self.assertEqual(seen["kw"]["stdin"], subprocess.DEVNULL)


class RealProcess(unittest.TestCase):
    """The real script as a real process, real sockets, fake upstream (no MAPLE)."""
    def launch(self, env_extra=None, args_extra=()):
        self.up = FakeUpstream()
        self.addCleanup(self.up.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.auth_port, llama = free_port(), free_port()
        env = {k: v for k, v in os.environ.items() if k != "API_KEY"}
        env.update({"PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})})
        self.proc = subprocess.Popen(
            [sys.executable, "-B", str(SCRIPT), "run", "--auth-only", "--state-dir", str(self.state), "--api-port", str(self.up.port),
             "--auth-port", str(self.auth_port), "--llama-port", str(llama), "--check-interval", "5", *args_extra],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        self.addCleanup(self._reap)

    def _reap(self):
        if self.proc.poll() is None:
            self.proc.kill()
        with contextlib.suppress(Exception):
            self.out = self.proc.communicate(timeout=10)[0]

    def wait_ready(self, timeout=25):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                self.fail("process exited early: rc=%s" % self.proc.returncode)
            try:
                st = json.loads((self.state / tn.STATE_FILE).read_text())
                if st.get("state") == "auth_proxy":
                    return st
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        self.fail("auth proxy did not become ready")

    def test_end_to_end(self):
        self.launch({"API_KEY": KEY})
        st = self.wait_ready()
        p = self.auth_port
        self.assertEqual((st["mode"], st["auth"]), ("auth-only", "bearer-proxy-verified"))
        self.assertEqual(call(p, "GET", "/health?probe=nokey")[0], 401)                                  # no key
        self.assertEqual(call(p, "GET", "/health?probe=wrongkey", bearer("wrong-" + "q" * 30))[0], 401)  # wrong key
        self.assertEqual(call(p, "GET", "/health?probe=basic", {"Authorization": "Basic " + KEY})[0], 401)   # wrong scheme
        self.assertEqual(call(p, "POST", "/v1/chat/completions?probe=nokey-post", None, {"model": "fast", "messages": []})[0], 401)
        self.assertEqual(call(p, "GET", "/healthz?probe=disallowed", bearer())[0], 404)                  # not an approved path
        self.assertEqual(call(p, "GET", "/health?probe=good", bearer())[0], 200)                         # right key
        st2, h2, b2 = call(p, "POST", "/v1/chat/completions", bearer(), {"model": "fast", "messages": []})
        self.assertEqual((st2, json.loads(b2)["echo"]["model"]), (200, "fast"))
        for rec in self.up.seen:
            self.assertNotIn("authorization", rec["headers"])
        probes = [r["path"] for r in self.up.seen if "probe=" in r["path"]]
        self.assertEqual(probes, ["/health?probe=good"])                  # of all the probes, only the correctly keyed one reached MAPLE
        self.assertTrue(any(r["method"] == "POST" for r in self.up.seen))  # (plus the keyed POST above and the manager's own keyed self-tests)
        # loopback only: not reachable through a non-loopback address
        try:
            lan = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan = "127.0.0.1"
        if not lan.startswith("127."):
            with self.assertRaises(OSError):
                socket.create_connection((lan, p), timeout=3).close()
        # status through the real CLI
        r = subprocess.run([sys.executable, "-B", str(SCRIPT), "status", "--json", "--state-dir", str(self.state), "--api-port", str(self.up.port),
                            "--auth-port", str(p), "--auth-only"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        ap = json.loads(r.stdout)["auth_proxy"]
        self.assertTrue(ap["auth_enforced"])
        self.assertEqual(ap["cloudflare"], "not used")
        if POSIX:
            self.assertTrue(ap["loopback_only"])
        # stopping the proxy never touches the upstream (the MAPLE stand-in)
        if POSIX:
            self.proc.send_signal(signal.SIGTERM)
            self.assertEqual(self.proc.wait(timeout=30), 0)
            self.assertFalse(tn.tcp_listening("127.0.0.1", p))
            self.assertFalse((self.state / "tunnel.lock").exists())
            self.assertEqual(json.loads((self.state / tn.STATE_FILE).read_text())["state"], "stopped")
        else:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.assertEqual(call(self.up.port, "GET", "/health")[0], 200)                    # upstream still serving
        # the key appears nowhere: not in the console output, not in any state/log file
        out = ""
        with contextlib.suppress(Exception):
            out = self.proc.communicate(timeout=10)[0].decode("utf-8", "replace")
        blob = out + "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in self.state.rglob("*") if f.is_file())
        self.assertNotIn(KEY, blob)
        self.assertNotIn("cloudflared", json.dumps(json.loads((self.state / tn.STATE_FILE).read_text())).replace('"cloudflared_pid": null', ""))

    def test_no_key_refuses_to_start(self):
        self.launch(None)
        out = self.proc.communicate(timeout=30)[0].decode("utf-8", "replace")
        self.assertNotEqual(self.proc.returncode, 0)
        self.assertIn("API_KEY is not set", out)
        self.assertFalse(tn.tcp_listening("127.0.0.1", self.auth_port))

    def test_short_key_refuses_to_start_without_echoing_it(self):
        self.launch({"API_KEY": "SHORTVALUE123"})
        out = self.proc.communicate(timeout=30)[0].decode("utf-8", "replace")
        self.assertNotEqual(self.proc.returncode, 0)
        self.assertNotIn("SHORTVALUE123", out)
        self.assertFalse(tn.tcp_listening("127.0.0.1", self.auth_port))


class HeartbeatCompatibility(unittest.TestCase):
    """The UNMODIFIED heartbeat reads tunnel_state.json; check it renders what the auth-only manager writes."""
    def test_heartbeat_report_shows_the_loopback_endpoint_and_bearer_requirement(self):
        import maple_heartbeat as hb
        import test_maple_heartbeat as thb
        th = Harness(self)
        th.mgr.start_proxy_auth_only()
        th.mgr.auth_verified, th.mgr.state = True, "AUTH_PROXY"
        th.mgr.write_state("AUTH_PROXY")
        doc = json.loads((th.state / tn.STATE_FILE).read_text())
        hh = thb.Harness(self)
        hh.state.mkdir(parents=True, exist_ok=True)
        (hh.state / hb.TUNNEL_STATE_FILE).write_text(json.dumps(doc))
        hh.sup.tick()
        st = hh.sup.build_status()
        rep = hh.sup.format_report(st)
        self.assertEqual(st["tunnel"]["mode"], "auth-only")
        self.assertIn("http://127.0.0.1:8088/v1", rep)
        self.assertIn("Bearer key required", rep)
        self.assertNotIn("DISABLED", rep)                          # must not claim "no tunnel" while the proxy is running
        self.assertNotIn(KEY, json.dumps(st) + rep)
        self.assertNotIn("trycloudflare", rep)

    def test_heartbeat_does_not_treat_an_unverified_proxy_as_protected(self):
        import maple_heartbeat as hb
        import test_maple_heartbeat as thb
        th = Harness(self)
        th.mgr.start_proxy_auth_only()
        th.mgr.write_state("FAILED_INSECURE")
        doc = json.loads((th.state / tn.STATE_FILE).read_text())
        self.assertIsNone(doc["url"])
        self.assertEqual(doc["auth"], "none")
        hh = thb.Harness(self)
        hh.state.mkdir(parents=True, exist_ok=True)
        (hh.state / hb.TUNNEL_STATE_FILE).write_text(json.dumps(doc))
        hh.sup.tick()
        self.assertNotIn("Bearer key required", hh.sup.format_report(hh.sup.build_status()))


class StaticSafety(unittest.TestCase):
    SRC = SCRIPT.read_text(encoding="utf-8")

    def code(self):
        return "\n".join(l for l in self.SRC.splitlines() if not l.strip().startswith("#"))

    def test_auth_only_paths_never_launch_or_signal_anything(self):
        import inspect
        for fn in (tn.TunnelManager.run_auth_only, tn.TunnelManager.start_proxy_auth_only, tn.TunnelManager.verify_auth_only,
                   tn.TunnelManager.tick_auth_only, tn.TunnelManager.restart_proxy):
            src = inspect.getsource(fn)
            for bad in ("popen(", "os.kill", "start_cloudflared", "which(", "SIGKILL", "subprocess"):
                self.assertNotIn(bad, src, "%s uses %s" % (fn.__name__, bad))

    def test_key_never_reaches_logging_or_output_calls(self):
        sinks = re.compile(r"\b(print|say|event|logger\.\w+|log_message|write_text|write|dump|atomic_write_json)\(")
        ident = re.compile(r"(self\.key\b|\bkey_b\b|(?<![\w.\"'])key(?![\w\"'=:]))")
        lit = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
        strip = lambda l: lit.sub('""', l)   # noqa: E731
        # self-check: the patterns must really match what they are meant to catch, otherwise this test would pass vacuously
        self.assertTrue(sinks.search("self.say(x)") and sinks.search("print(x)") and sinks.search("f.write(x)"))
        self.assertTrue(ident.search(strip("self.say(self.key)")) and ident.search(strip('print("x %s" % key)')))
        self.assertFalse(ident.search(strip('print("api key ok", "key required")')))
        for line in self.code().splitlines():
            if sinks.search(line) and ident.search(strip(line)):
                if "_bearer()" in line or "def " in line:
                    continue
                self.fail("key variable reaches an output call: " + line.strip())

    def test_no_key_written_to_disk_and_env_scrubbed(self):
        import inspect
        self.assertIn('os.environ.pop("API_KEY", None)', inspect.getsource(tn.main))
        self.assertNotIn("write_text(key", self.code())
        self.assertNotIn("write_text(self.key", self.code())

    def test_only_one_subprocess_launch_site_in_run_and_one_in_start(self):
        import inspect
        self.assertEqual(self.code().count("self.popen("), 1)         # cloudflared, public mode only
        self.assertEqual(self.code().count("popen("), 2)              # + cmd_start's detached launcher of this same script
        self.assertIn("sys.executable", inspect.getsource(tn.cmd_start))
        self.assertIn('"--auth-only"', inspect.getsource(tn.cmd_start))

    def test_no_forbidden_process_tools(self):
        for bad in ("pkill", "killall", "taskkill", "os.killpg", "signal.SIGKILL", "pip install", "-m maple"):
            self.assertNotIn(bad, self.code())

    def test_default_cli_unchanged(self):
        ns = tn.build_parser().parse_args(["run"])
        self.assertFalse(ns.public or ns.auth_only)


if __name__ == "__main__":
    unittest.main(verbosity=1)
