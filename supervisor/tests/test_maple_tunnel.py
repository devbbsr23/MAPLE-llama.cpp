"""Unit tests for maple_tunnel.py. cloudflared, the clock and the public edge are mocked.
The auth-proxy tests use a real in-process HTTP server on 127.0.0.1 with a FAKE upstream (ephemeral ports); no MAPLE, no
llama-server, no cloudflared, no Kaggle, no external network."""
import io
import json
import os
import signal
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import maple_tunnel as tn   # noqa: E402

KEY = "k" * 43
URL = "https://quiet-river-demo-test.trycloudflare.com"
LOG_WITH_URL = ("2026-11-20T10:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
                "2026-11-20T10:00:02Z INF |  %s  |\n" % URL)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProc:
    def __init__(self, pid=5151, code=None):
        self.pid, self.returncode, self.terminated, self.killed = pid, code, False, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeProxy:
    instances = []

    def __init__(self, key, listen_port, up_host, up_port, on_event=None):
        self.key, self.port, self.up = key, listen_port, (up_host, up_port)
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
    """Scripted responses for every URL the manager requests."""
    def __init__(self):
        self.api = tn.Reply(200, json.dumps({"status": "ok"}).encode(), None)
        self.public_no_key = tn.Reply(401, b"{}", None)
        self.public_key = tn.Reply(200, b"{}", None)
        self.proxy_no_key = tn.Reply(401, b"{}", None)
        self.proxy_bad_key = tn.Reply(401, b"{}", None)
        self.proxy_key = tn.Reply(200, b"{}", None)
        self.calls = []

    def __call__(self, url, timeout, headers=None):
        auth = (headers or {}).get("Authorization", "")
        self.calls.append((url, bool(auth)))
        if url.startswith("http://127.0.0.1:8000"):
            return self.api
        if url.startswith("http://127.0.0.1:8088"):
            return self.proxy_key if auth == "Bearer " + KEY else (self.proxy_bad_key if auth else self.proxy_no_key)
        if url.startswith("https://"):
            return self.public_key if auth == "Bearer " + KEY else self.public_no_key
        raise AssertionError("unexpected URL " + url)


class H:
    def __init__(self, tc, public=True, **over):
        self.tmp = tempfile.TemporaryDirectory()
        tc.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "state"
        self.cfg = tn.TunnelConfig(state_dir=self.state, public=public, url_timeout=10, verify_timeout=10, backoff_base=15,
                                   check_interval=60, stable_after=300, **over)
        self.clock, self.edge, self.out = FakeClock(), Edge(), []
        self.popens, self.proc_exit, self.write_url, self.which_result = [], None, True, "/usr/local/bin/cloudflared"
        self.procs = []
        FakeProxy.instances = []
        self.mgr = tn.TunnelManager(
            self.cfg, KEY, http=self.edge, popen=self._popen, which=lambda n: self.which_result, clock=self.clock, sleep=self.clock.sleep,
            list_procs=lambda: self.procs, alive=lambda p: False, cmdline=lambda p: None, proxy_factory=FakeProxy, out=self.out.append)
        tc.addCleanup(self.mgr.close)

    def _popen(self, cmd, **kw):
        self.popens.append((cmd, kw))
        if self.write_url:
            (self.state / "tunnel.log").write_text(LOG_WITH_URL, encoding="utf-8")
        return FakeProc(5151, self.proc_exit)

    @property
    def printed(self):
        return "\n".join(self.out)


class UrlDetection(unittest.TestCase):
    def test_detects_quick_tunnel_url(self):
        self.assertEqual(tn.detect_url(LOG_WITH_URL), URL)

    def test_no_url(self):
        self.assertIsNone(tn.detect_url("INF Requesting new quick Tunnel ...\n"))
        self.assertIsNone(tn.detect_url(""))
        self.assertIsNone(tn.detect_url(None))

    def test_ignores_non_trycloudflare_urls(self):
        self.assertIsNone(tn.detect_url("see https://developers.cloudflare.com/docs and https://example.com"))
        self.assertEqual(tn.detect_url("api.trycloudflare.com ... https://abc-1.trycloudflare.com"), "https://abc-1.trycloudflare.com")


class DefaultModeIsSsh(unittest.TestCase):
    def test_default_does_not_start_anything_public(self):
        h = H(self, public=False)
        rc = h.mgr.run_ssh_mode()
        self.assertEqual(rc, 0)
        self.assertEqual(h.popens, [])
        self.assertEqual(FakeProxy.instances, [])
        st = json.loads((h.state / "tunnel_state.json").read_text())
        self.assertEqual((st["mode"], st["state"], st["auth"], st["url"]), ("ssh", "disabled", "none", None))
        self.assertIn("ssh -N -L 8000:127.0.0.1:8000 kaggle2", h.printed)
        self.assertIn("no authentication", h.printed.lower() + "no authentication")

    def test_main_default_is_ssh_mode_and_never_calls_cloudflared(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(tn.subprocess, "Popen") as pop:
            rc = tn.main(["run", "--state-dir", t])
            self.assertEqual(rc, 0)
            pop.assert_not_called()

    def test_protected_ssh_cloudflared_is_reported_not_touched(self):
        h = H(self, public=False)
        h.procs = [tn.ProcInfo(1971, "cloudflared", ("cloudflared", "tunnel", "--url", "ssh://localhost:2222"))]
        with mock.patch("os.kill") as k:
            h.mgr.run_ssh_mode()
            k.assert_not_called()
        self.assertIn("1971", h.printed)
        self.assertIn("untouched", h.printed)

    def test_is_ssh_cloudflared(self):
        self.assertTrue(tn.is_ssh_cloudflared(tn.ProcInfo(1, "cloudflared", ("cloudflared", "tunnel", "--url", "ssh://localhost:2222"))))
        self.assertFalse(tn.is_ssh_cloudflared(tn.ProcInfo(1, "cloudflared", ("cloudflared", "tunnel", "--url", "http://127.0.0.1:8088"))))
        self.assertFalse(tn.is_ssh_cloudflared(tn.ProcInfo(1, "python3", ("python3", "ssh://x"))))


class TunnelStartup(unittest.TestCase):
    def test_successful_startup_is_verified_and_points_at_the_proxy_not_maple(self):
        h = H(self)
        ok, detail = h.mgr.start_public()
        self.assertTrue(ok, detail)
        cmd, kw = h.popens[0]
        self.assertEqual(cmd[0], "/usr/local/bin/cloudflared")
        self.assertEqual(cmd[cmd.index("--url") + 1], "http://127.0.0.1:8088")
        self.assertNotIn("8000", " ".join(cmd))
        self.assertNotIn("8081", " ".join(cmd))
        self.assertEqual(h.mgr.url, URL)
        self.assertTrue(FakeProxy.instances[0].started)
        self.assertEqual(FakeProxy.instances[0].up, ("127.0.0.1", 8000))

    def test_api_key_never_reaches_cloudflared_environment_or_args(self):
        h = H(self)
        with mock.patch.dict(os.environ, {"API_KEY": KEY}):
            h.mgr.start_public()
        cmd, kw = h.popens[0]
        self.assertNotIn("API_KEY", kw["env"])
        self.assertNotIn(KEY, " ".join(cmd))

    def test_cloudflared_missing_gives_clear_message(self):
        h = H(self)
        h.which_result = None
        ok, detail = h.mgr.start_public()
        self.assertFalse(ok)
        self.assertIn("cloudflared not found", detail)
        self.assertEqual(h.popens, [])

    def test_cloudflared_detected_with_shutil_which(self):
        src = (HERE.parent / "maple_tunnel.py").read_text(encoding="utf-8")
        self.assertIn('self.which("cloudflared")', src)
        self.assertIn("which=shutil.which", src)

    def test_cloudflared_exits_during_startup(self):
        h = H(self)
        h.proc_exit, h.write_url = 1, False
        ok, detail = h.mgr.start_public()
        self.assertFalse(ok)
        self.assertIn("exited during startup", detail)

    def test_url_never_appears(self):
        h = H(self, )
        h.write_url = False
        ok, detail = h.mgr.start_public()
        self.assertFalse(ok)
        self.assertIn("URL not detected", detail)
        self.assertIsNone(h.mgr.url)

    def test_tunnel_log_is_written_in_the_state_dir(self):
        h = H(self)
        h.mgr.start_public()
        self.assertTrue((h.state / "tunnel.log").is_file())
        self.assertIn(URL, (h.state / "tunnel.log").read_text())

    def test_proxy_selftest_failure_blocks_cloudflared(self):
        h = H(self)
        h.edge.proxy_no_key = tn.Reply(200, b"{}", None)          # a proxy that lets unauthenticated calls in
        ok, detail = h.mgr.start_public()
        self.assertFalse(ok)
        self.assertIn("self-test failed", detail)
        self.assertEqual(h.popens, [])

    def test_public_url_that_answers_without_key_is_stopped_and_reported_insecure(self):
        h = H(self)
        h.edge.public_no_key = tn.Reply(200, b"{}", None)
        proc_holder = []
        ok, detail = h.mgr.start_public()
        self.assertFalse(ok)
        self.assertTrue(h.mgr.insecure)
        self.assertIn("SECURITY", detail)
        self.assertIsNone(h.mgr.url)
        self.assertNotIn(URL, h.printed)                          # an insecure URL is never presented

    def test_public_url_that_never_answers_is_not_advertised(self):
        h = H(self)
        h.edge.public_no_key = tn.Reply(None, b"", "unreachable")
        h.edge.public_key = tn.Reply(None, b"", "unreachable")
        ok, detail = h.mgr.start_public()
        self.assertFalse(ok)
        self.assertIsNone(h.mgr.url)
        self.assertNotIn("Endpoint", h.printed)

    def test_dns_warmup_is_tolerated(self):
        h = H(self)
        seq = [tn.Reply(None, b"", "unreachable"), tn.Reply(401, b"{}", None)]
        orig = h.edge.__call__

        def edge(url, timeout, headers=None):
            if url.startswith("https://") and not (headers or {}).get("Authorization") and seq:
                return seq.pop(0)
            return orig(url, timeout, headers)
        h.mgr.http = edge
        ok, _ = h.mgr.start_public()
        self.assertTrue(ok)


class BannerAndOutput(unittest.TestCase):
    def test_key_and_url_printed_exactly_once(self):
        h = H(self)
        ok, _ = h.mgr.start_public()
        h.mgr.url = URL
        h.mgr.banner()
        h.mgr.banner()                                            # a restart must not print the key again
        self.assertEqual(h.printed.count(KEY), 1)
        self.assertGreaterEqual(h.printed.count(URL), 2)          # banner + "endpoint changed" line
        self.assertIn("shown once", h.printed)

    def test_key_not_in_any_file(self):
        h = H(self)
        h.mgr.start_public()
        h.mgr.url = URL
        h.mgr.state = "RUNNING"
        h.mgr.banner()
        h.mgr.write_state()
        h.mgr.event("anything", url=URL)
        for p in h.state.rglob("*"):
            if p.is_file():
                self.assertNotIn(KEY, p.read_text(errors="replace"), p.name)

    def test_state_file_claims_auth_only_when_verified_and_running(self):
        h = H(self)
        h.mgr.write_state("STARTING")
        self.assertEqual(json.loads((h.state / "tunnel_state.json").read_text())["auth"], "none")
        h.mgr.start_public()
        h.mgr.write_state("RUNNING")
        st = json.loads((h.state / "tunnel_state.json").read_text())
        self.assertEqual(st["auth"], "bearer-proxy-verified")
        self.assertEqual(st["url"], URL)
        h.mgr.write_state("FAILED")
        self.assertEqual(json.loads((h.state / "tunnel_state.json").read_text())["auth"], "none")

    def test_report_never_exposes_8081_as_public(self):
        h = H(self)
        h.mgr.start_public()
        h.mgr.state = "RUNNING"
        rep = h.mgr.format_report()
        self.assertIn("never exposed", rep)
        self.assertIn("Bearer key enforced", rep)


class SupervisionAndRecovery(unittest.TestCase):
    def _up(self, h):
        ok, d = h.mgr.start_public()
        self.assertTrue(ok, d)
        h.mgr.state = "RUNNING"
        return h.mgr

    def test_healthy_tick(self):
        h = H(self)
        m = self._up(h)
        self.assertEqual(m.tick(), "healthy")
        self.assertEqual(m.failures, 0)

    def test_tunnel_process_failure_triggers_restart(self):
        h = H(self)
        m = self._up(h)
        m.proc.returncode = 1                                      # cloudflared died
        res = m.tick()
        self.assertEqual(res, "recover:recovered")
        self.assertEqual(len(h.popens), 2)
        self.assertIsNotNone(m.url)

    def test_disconnected_tunnel_recovers_after_three_failed_public_checks(self):
        h = H(self)
        m = self._up(h)
        bearer_calls = {"n": 0}
        orig = h.edge.__call__

        def edge(url, timeout, headers=None):
            if url.startswith("https://") and (headers or {}).get("Authorization"):
                bearer_calls["n"] += 1
                if bearer_calls["n"] <= 3:                         # three failed periodic checks, then the new tunnel verifies
                    return tn.Reply(None, b"", "timeout")
            return orig(url, timeout, headers)
        m.http = edge
        self.assertEqual(m.tick(), "failing")
        self.assertEqual(m.tick(), "failing")
        self.assertEqual(len(h.popens), 1)
        self.assertEqual(m.tick(), "recover:recovered")            # 3rd consecutive failure -> restart -> verified
        self.assertEqual(len(h.popens), 2)

    def test_restart_does_not_reprint_the_key(self):
        h = H(self)
        m = self._up(h)
        m.banner()
        m.proc.returncode = 1
        m.tick()
        self.assertEqual(h.printed.count(KEY), 1)
        self.assertIn("same API key", h.printed)

    def test_new_url_after_restart_is_verified_again(self):
        h = H(self)
        m = self._up(h)
        m.proc.returncode = 1
        h.edge.public_no_key = tn.Reply(200, b"{}", None)          # the new tunnel turns out to be unauthenticated
        res = m.tick()
        self.assertEqual(res, "recover:failed")
        self.assertTrue(m.insecure)
        self.assertEqual(m.tick(), "insecure")

    def test_api_down_does_not_restart_the_tunnel(self):
        h = H(self)
        m = self._up(h)
        h.edge.api = tn.Reply(None, b"", "refused")
        self.assertEqual(m.tick(), "waiting_for_api")
        self.assertEqual(len(h.popens), 1)
        self.assertEqual(m.state, "WAITING_FOR_API")

    def test_failure_counter_resets_on_success(self):
        h = H(self)
        m = self._up(h)
        h.edge.public_key = tn.Reply(502, b"{}", None)
        m.tick()
        h.edge.public_key = tn.Reply(200, b"{}", None)
        m.tick()
        self.assertEqual(m.failures, 0)

    def test_exponential_backoff_and_restart_limit(self):
        h = H(self, max_restarts=3)
        m = self._up(h)
        h.write_url = False
        m.proc.returncode = 1
        self.assertEqual(m.tick(), "recover:failed")               # attempt 1
        self.assertEqual(m.tick(), "recover:backoff")
        h.clock.sleep(16)
        self.assertEqual(m.tick(), "recover:failed")               # attempt 2 (next wait 30 s from its start)
        h.clock.sleep(5)                                           # the failed startup itself consumed ~10 fake seconds
        self.assertEqual(m.tick(), "recover:backoff")
        h.clock.sleep(20)
        self.assertEqual(m.tick(), "recover:failed")               # attempt 3
        h.clock.sleep(500)
        self.assertEqual(m.tick(), "recover:limit")
        self.assertTrue(m.gave_up)
        self.assertEqual(m.state, "GAVE_UP")
        self.assertTrue(any("restart limit" in line for line in h.out))

    def test_auth_proxy_death_restarts_it(self):
        h = H(self)
        m = self._up(h)
        FakeProxy.instances[0]._alive = False
        res = m.tick()
        self.assertEqual(res, "recover:recovered")
        self.assertEqual(len(FakeProxy.instances), 2)


class ShutdownAndLocking(unittest.TestCase):
    def test_shutdown_stops_only_its_own_cloudflared_and_the_proxy(self):
        h = H(self)
        h.procs = [tn.ProcInfo(1971, "cloudflared", ("cloudflared", "tunnel", "--url", "ssh://localhost:2222"))]
        h.mgr.start_public()
        proc = h.mgr.proc
        with mock.patch("os.kill") as k:
            h.mgr.shutdown()
            k.assert_not_called()                                  # no signal to any pid; Popen handles only its own child
        self.assertTrue(proc.terminated)
        self.assertFalse(proc.killed)
        self.assertTrue(FakeProxy.instances[0].stopped)
        self.assertEqual(json.loads((h.state / "tunnel_state.json").read_text())["state"], "stopped")
        self.assertIsNone(h.mgr.url)

    def test_stop_escalates_to_kill_only_on_own_child_after_timeout(self):
        h = H(self)

        class Stubborn(FakeProc):
            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                if not self.killed:
                    raise tn.subprocess.TimeoutExpired("x", timeout)
                return -9
        h.mgr.proc = Stubborn()
        p = h.mgr.proc
        h.mgr.stop_cloudflared()
        self.assertTrue(p.killed)

    def test_second_manager_is_rejected(self):
        h = H(self)
        h.mgr.acquire_lock()
        other = tn.TunnelManager(h.cfg, KEY, alive=lambda p: True, cmdline=lambda p: "python maple_tunnel.py run", console=False)
        self.addCleanup(other.close)
        with mock.patch("os.getpid", return_value=99999):
            with self.assertRaises(tn.DuplicateTunnelManager):
                other.acquire_lock()

    def test_stale_lock_is_taken_over(self):
        h = H(self)
        tn.atomic_write_json(h.state / "tunnel.lock", {"pid": 424242})
        h.mgr.acquire_lock()
        self.assertEqual(json.loads((h.state / "tunnel.lock").read_text())["pid"], os.getpid())

    def test_lock_removed_on_shutdown(self):
        h = H(self)
        h.mgr.acquire_lock()
        h.mgr.shutdown()
        self.assertFalse((h.state / "tunnel.lock").exists())

    def test_orphan_cloudflared_cleanup_only_if_command_line_proves_it_is_ours(self):
        h = H(self)
        tn.atomic_write_json(h.state / "tunnel_state.json", {"cloudflared_pid": 777})
        for cmdline, expect_kill in (
            ("/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8088", True),
            ("/bin/cloudflared tunnel --url ssh://localhost:2222", False),
            ("/bin/cloudflared tunnel --url http://127.0.0.1:8000", False),          # would be MAPLE directly: not ours
            ("python something_else.py", False)):
            m = tn.TunnelManager(h.cfg, KEY, alive=lambda p: True, cmdline=lambda p, c=cmdline: c, console=False)
            with mock.patch("os.kill") as k:
                m.cleanup_orphan()
            m.close()
            self.assertEqual(k.called, expect_kill, cmdline)
            if expect_kill:
                k.assert_called_once_with(777, signal.SIGTERM)

    def test_run_loop_shutdown_signal_exits_cleanly(self):
        h = H(self)
        h.edge.api = tn.Reply(200, json.dumps({"status": "ok"}).encode(), None)
        n = {"i": 0}
        real = h.clock.sleep

        def sleeper(s):
            real(s)
            n["i"] += 1
            if n["i"] > 130:
                h.mgr.stopping = True
        h.mgr.sleep = sleeper
        with mock.patch.object(h.mgr, "install_signal_handlers"):
            rc = h.mgr.run_public()
        self.assertEqual(rc, 0)
        self.assertFalse((h.state / "tunnel.lock").exists())
        self.assertEqual(json.loads((h.state / "tunnel_state.json").read_text())["state"], "stopped")
        self.assertIn(KEY, h.printed)
        self.assertEqual(h.printed.count(KEY), 1)

    def test_run_waits_for_api_before_starting_a_tunnel(self):
        h = H(self, wait_for_api=20)
        h.edge.api = tn.Reply(None, b"", "refused")
        with mock.patch.object(h.mgr, "install_signal_handlers"):
            rc = h.mgr.run_public()
        self.assertEqual(rc, 5)
        self.assertEqual(h.popens, [])
        self.assertEqual(FakeProxy.instances, [])


class AuthProxyReal(unittest.TestCase):
    """Real HTTP on loopback against a fake upstream."""
    @classmethod
    def setUpClass(cls):
        cls.hits = []

        class Up(BaseHTTPRequestHandler):
            def log_message(self, *a): pass   # noqa: E704

            def do_GET(self):
                cls.hits.append(("GET", self.path, self.headers.get("Authorization")))
                b = json.dumps({"ok": True, "path": self.path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n)
                cls.hits.append(("POST", self.path, body))
                if json.loads(body or b"{}").get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i in range(3):
                        self.wfile.write(b"data: {\"i\": %d}\n\n" % i)
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    b = json.dumps({"echo": json.loads(body or b"{}")}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
        cls.up.daemon_threads = True
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.proxy = tn.AuthProxy(KEY, 0, "127.0.0.1", cls.up.server_address[1])
        cls.proxy.start()
        cls.base = "http://127.0.0.1:%d" % cls.proxy.port

    @classmethod
    def tearDownClass(cls):
        cls.proxy.stop()
        cls.up.shutdown()
        cls.up.server_close()

    def setUp(self):
        type(self).hits.clear()

    def req(self, method, path, key=KEY, body=None, headers=None):
        h = dict(headers or {})
        if key is not None:
            h["Authorization"] = "Bearer " + key
        data = json.dumps(body).encode() if body is not None else None
        if data:
            h["Content-Type"] = "application/json"
        r = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(r, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_no_key_is_401_and_never_reaches_upstream(self):
        self.assertEqual(self.req("GET", "/v1/models", key=None)[0], 401)
        self.assertEqual(self.hits, [])

    def test_wrong_key_is_401(self):
        s, b = self.req("GET", "/v1/models", key="wrong" * 9)
        self.assertEqual(s, 401)
        self.assertEqual(self.hits, [])
        self.assertNotIn(KEY.encode(), b)

    def test_non_bearer_scheme_is_401(self):
        self.assertEqual(self.req("GET", "/health", key=None, headers={"Authorization": "Basic " + KEY})[0], 401)

    def test_unauthenticated_unknown_path_is_401_not_404(self):
        self.assertEqual(self.req("GET", "/secret/admin", key=None)[0], 401)

    def test_valid_key_is_proxied_and_authorization_is_not_forwarded(self):
        s, b = self.req("GET", "/v1/models")
        self.assertEqual(s, 200)
        self.assertEqual(json.loads(b)["path"], "/v1/models")
        self.assertIsNone(self.hits[0][2])                         # upstream never sees the bearer key

    def test_health_and_query_string_pass(self):
        s, b = self.req("GET", "/health?deep=1")
        self.assertEqual((s, json.loads(b)["path"]), (200, "/health?deep=1"))

    def test_post_body_is_forwarded(self):
        s, b = self.req("POST", "/v1/chat/completions", body={"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)
        self.assertEqual(json.loads(b)["echo"]["model"], "fast")

    def test_all_five_endpoints_allowed(self):
        for m, p in (("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"), ("POST", "/v1/route")):
            self.assertEqual(self.req(m, p, body={"x": 1} if m == "POST" else None)[0], 200, p)

    def test_other_paths_and_methods_are_404_even_with_a_valid_key(self):
        for m, p in (("GET", "/"), ("GET", "/admin"), ("GET", "/v1/chat/completions"), ("POST", "/v1/models"), ("DELETE", "/v1/models"), ("GET", "/../etc/passwd")):
            s, _ = self.req(m, p, body={"x": 1} if m == "POST" else None)
            self.assertEqual(s, 404, "%s %s" % (m, p))
        self.assertEqual(self.hits, [])

    def test_streaming_passes_through(self):
        s, b = self.req("POST", "/v1/chat/completions", body={"stream": True})
        self.assertEqual(s, 200)
        self.assertIn(b"data: [DONE]", b)
        self.assertEqual(b.count(b"data: {"), 3)

    def test_oversized_body_rejected(self):
        r = urllib.request.Request(self.base + "/v1/completions", data=b"{}", method="POST",
                                   headers={"Authorization": "Bearer " + KEY, "Content-Length": str(tn.MAX_BODY + 1)})
        try:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(r, timeout=5)
            self.fail("expected an error")
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, (400, 413))
        except (urllib.error.URLError, ConnectionError, OSError):
            pass                                                   # connection dropped before the body was sent: also a rejection

    def test_upstream_down_is_502_without_leaking_details(self):
        p = tn.AuthProxy(KEY, 0, "127.0.0.1", 1)                    # nothing listens on port 1
        p.start()
        try:
            r = urllib.request.Request("http://127.0.0.1:%d/health" % p.port, headers={"Authorization": "Bearer " + KEY})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(r, timeout=10)
            self.assertEqual(cm.exception.code, 502)
            self.assertNotIn(b"Traceback", cm.exception.read())
        finally:
            p.stop()

    def test_proxy_does_not_log_requests_or_keys(self):
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf), mock.patch("sys.stdout", buf):
            self.req("GET", "/v1/models")
            self.req("GET", "/v1/models", key="bad" * 10)
        self.assertNotIn(KEY, buf.getvalue())
        self.assertNotIn("GET /v1/models", buf.getvalue())


class KeyHandlingAndConfig(unittest.TestCase):
    def test_uses_existing_api_key_env(self):
        self.assertEqual(tn.get_api_key({"API_KEY": KEY}), (KEY, False))

    def test_generates_a_key_when_missing(self):
        k, gen = tn.get_api_key({})
        self.assertTrue(gen)
        self.assertGreaterEqual(len(k), 40)

    def test_short_key_rejected(self):
        with self.assertRaises(SystemExit):
            tn.get_api_key({"API_KEY": "short"})

    def test_llama_port_must_not_be_the_exposed_port(self):
        with self.assertRaises(SystemExit):
            tn.TunnelConfig(llama_port=8088, auth_port=8088).validate()
        with self.assertRaises(SystemExit):
            tn.TunnelConfig(auth_port=8000).validate()             # the tunnel must never point straight at MAPLE

    def test_non_loopback_api_host_rejected(self):
        with self.assertRaises(SystemExit):
            tn.TunnelConfig(api_host="0.0.0.0").validate()

    def test_defaults(self):
        self.assertEqual((tn.API_HOST, tn.API_PORT, tn.LLAMA_PORT, tn.CHECK_INTERVAL, tn.HEARTBEAT_MINUTES), ("127.0.0.1", 8000, 8081, 60, 5))
        self.assertFalse(tn.TunnelConfig().public)

    def test_env_override_and_cli_priority(self):
        ns = tn.build_parser().parse_args(["run", "--state-dir", tempfile.gettempdir(), "--api-port", "9100"])
        with mock.patch.dict(os.environ, {"MAPLE_TUN_API_PORT": "9200", "MAPLE_TUN_AUTH_PORT": "9300"}):
            cfg = tn.config_from_args(ns)
        self.assertEqual((cfg.api_port, cfg.auth_port), (9100, 9300))


class StaticSafetyChecks(unittest.TestCase):
    SRC = (HERE.parent / "maple_tunnel.py").read_text(encoding="utf-8")

    def test_never_touches_maple_or_llama_server_processes(self):
        code = "\n".join(l for l in self.SRC.splitlines() if not l.strip().startswith("#"))
        for bad in ("pkill", "killall", "taskkill", "os.killpg", "signal.SIGKILL", "pip install", "llama-server\"", "-m maple"):
            self.assertNotIn(bad, code)
        # start_maple.sh may only be probed with is_file() (locating the project's logs folder), never executed
        for line in code.splitlines():
            if "start_maple" in line:
                self.assertIn("is_file()", line)
        self.assertEqual(code.count("self.popen("), 1)              # the single subprocess launch is cloudflared

    def test_no_logging_of_the_key(self):
        for line in self.SRC.splitlines():
            if "self.key" in line and ("event(" in line or "logger" in line or "log_message" in line):
                self.fail("key passed to logging: " + line)

    def test_default_cli_does_not_enable_public(self):
        ns = tn.build_parser().parse_args(["run"])
        self.assertFalse(ns.public)


if __name__ == "__main__":
    unittest.main(verbosity=2)
