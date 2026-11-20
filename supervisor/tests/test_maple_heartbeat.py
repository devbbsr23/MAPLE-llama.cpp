"""Unit tests for maple_heartbeat.py. Everything external is mocked: HTTP, processes, clock, nvidia-smi.
No server, no Kaggle, no network, no real process is started or signalled."""
import json
import os
import re
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import maple_heartbeat as hb   # noqa: E402

MODELS = ("deepseek", "gemma", "qwen3", "fast")


def health_body(status="ok", loaded=None, inflight=0, switches=0, healthy=False, missing=()):
    d = {"status": status, "uptime_s": 5, "models": {m: ("ok" if m not in missing else "missing") for m in MODELS},
         "model_paths_ok": not missing, "runtime": {"loaded": loaded, "inflight": inflight, "switches": switches, "healthy": healthy}}
    if missing:
        d["missing"] = [{"id": m, "path": "/x/%s.gguf" % m, "status": "missing"} for m in missing]
    return d


def models_body(drop=(), aliases=None):
    data = [{"id": m, "object": "model", "name": "Name-%s" % m} for m in MODELS if m not in drop]
    data.append({"id": "auto", "object": "model", "name": "automatic specialist routing"})
    return {"object": "list", "data": data, "aliases": {"quant": "qwen3"} if aliases is None else aliases}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakePopen:
    def __init__(self, pid=4242, exits_with=None):
        self.pid, self._code = pid, exits_with
        self.returncode = exits_with

    def poll(self):
        return self._code


class Net:
    """Scripted MAPLE API. Mutate .health / .models to simulate states."""
    def __init__(self):
        self.health = hb.HttpResult(200, health_body(), None)
        self.models = hb.HttpResult(200, models_body(), None)
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if url.endswith("/health"):
            return self.health
        if url.endswith("/v1/models"):
            return self.models
        raise AssertionError("unexpected URL requested by the heartbeat: " + url)


API_CMD = ("python3", "-s", "-m", "maple", "serve", "--host", "127.0.0.1", "--port", "8000")


class Harness:
    def __init__(self, tc, **cfg_over):
        self.tmp = tempfile.TemporaryDirectory()
        tc.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.home = root / "maple"
        (self.home / "scripts").mkdir(parents=True)
        (self.home / "scripts" / "start_maple.sh").write_text("#!/bin/bash\n")
        self.state = root / "state"
        kw = dict(maple_home=self.home, state_dir=self.state, disk_path=root, backoff_base=30, backoff_max=600, stop_timeout=5,
                  start_timeout=20, check_interval=60, max_restarts=5)
        kw.update(cfg_over)
        self.cfg = hb.Config(**kw)
        self.clock, self.net = FakeClock(), Net()
        self.procs = [hb.ProcInfo(100, 1, "python3", API_CMD)]
        self.signals, self.popens, self.dead = [], [], set()
        self.popen_exit = None            # None -> started process stays alive
        self.on_start = None              # callable run when popen is invoked
        self.port_busy = False
        self.sigterm_works = True
        self.out = []
        self.sup = hb.Supervisor(
            self.cfg, http_get=self.net, popen=self._popen, list_procs=lambda: list(self.procs), send_signal=self._signal,
            pid_alive=lambda p: p not in self.dead and any(x.pid == p for x in self.procs), clock=self.clock, sleep=self.clock.sleep,
            gpu_query=lambda libs=(): {"available": False, "error": "nvidia-smi not found", "devices": []},
            ram_read=lambda: {"total_mib": 32000, "available_mib": 30000, "available_pct": 93.8},
            disk_usage=lambda p: SimpleNamespace(free=20 * 2**30, total=100 * 2**30), tcp_probe=self._tcp,
            which=lambda n: "/bin/" + n, out=self.out.append)
        tc.addCleanup(self.sup.close)

    def _tcp(self, host, port, timeout=1.0):
        return self.port_busy and port == self.cfg.api_port

    def _signal(self, pid, sig):
        self.signals.append((pid, sig))
        if self.sigterm_works:
            self.procs = [p for p in self.procs if p.pid != pid]

    def _popen(self, cmd, **kw):
        self.popens.append((cmd, kw))
        if self.on_start:
            self.on_start()
        elif self.popen_exit is None:
            self.procs = [p for p in self.procs if not hb.is_maple_api(p, 8000)] + [hb.ProcInfo(777, 1, "python3", API_CMD)]
        return FakePopen(777, self.popen_exit)

    def kill_api(self):
        self.procs = [p for p in self.procs if not hb.is_maple_api(p, 8000)]

    def refuse(self):
        self.net.health = hb.HttpResult(None, None, "refused")

    def timeout(self):
        self.net.health = hb.HttpResult(None, None, "timeout")


class HealthyAndFailures(unittest.TestCase):
    def test_healthy_api(self):
        h = Harness(self)
        self.assertEqual(h.sup.tick(), "healthy")
        self.assertEqual(h.sup.api_state, "ONLINE")
        self.assertEqual(h.sup.failures, 0)
        self.assertEqual(h.sup.last_probe.models_state, "ok")

    def test_only_health_and_models_are_requested_never_inference(self):
        h = Harness(self)
        h.sup.tick()
        self.assertTrue(h.net.calls)
        for u in h.net.calls:
            self.assertRegex(u, r"/(health|v1/models)$")
            self.assertNotIn("chat", u)
            self.assertNotIn("completions", u)

    def test_api_unavailable_counts_but_does_not_recover_immediately(self):
        h = Harness(self)
        h.refuse()                                   # process still present, port refusing -> failure 1/3
        self.assertEqual(h.sup.tick(), "failing")
        self.assertEqual(h.sup.failures, 1)
        self.assertEqual(h.popens, [])

    def test_three_consecutive_failures_trigger_recovery(self):
        h = Harness(self)
        h.timeout()
        self.assertEqual(h.sup.tick(), "failing")
        self.assertEqual(h.sup.tick(), "failing")
        self.assertEqual(h.popens, [])
        h.net.health = hb.HttpResult(None, None, "timeout")

        def healed():
            h.net.health = hb.HttpResult(200, health_body(), None)
            h.procs.append(hb.ProcInfo(777, 1, "python3", API_CMD))
        h.on_start = healed
        res = h.sup.tick()
        self.assertEqual(res, "recover:recovered")
        self.assertEqual(len(h.popens), 1)

    def test_a_good_check_resets_the_failure_counter(self):
        h = Harness(self)
        h.timeout()
        h.sup.tick()
        h.sup.tick()
        h.net.health = hb.HttpResult(200, health_body(), None)
        h.sup.tick()
        self.assertEqual(h.sup.failures, 0)
        h.timeout()
        self.assertEqual(h.sup.tick(), "failing")          # starts again at 1, so no recovery

    def test_maple_process_exit_recovers_without_waiting_for_three_failures(self):
        h = Harness(self)
        h.kill_api()
        h.refuse()

        def healed():
            h.net.health = hb.HttpResult(200, health_body(), None)
            h.procs.append(hb.ProcInfo(777, 1, "python3", API_CMD))
        h.on_start = healed
        self.assertEqual(h.sup.tick(), "recover:recovered")
        self.assertEqual(h.sup.failures, 0)
        self.assertEqual(h.signals, [])                      # nothing to stop: it had already exited

    def test_recovery_stops_the_old_api_with_sigterm_only_then_starts_via_official_script(self):
        h = Harness(self)
        h.timeout()
        h.sup.tick()
        h.sup.tick()

        def healed():
            h.net.health = hb.HttpResult(200, health_body(), None)
            h.procs.append(hb.ProcInfo(777, 1, "python3", API_CMD))
        h.on_start = healed
        h.sup.tick()
        self.assertEqual(h.signals, [(100, signal.SIGTERM)])
        cmd, kw = h.popens[0]
        self.assertEqual((Path(cmd[1]).parent.name, Path(cmd[1]).name), ("scripts", "start_maple.sh"))
        self.assertEqual(cmd[2:], ["--host", "127.0.0.1", "--port", "8000"])
        self.assertTrue(kw["start_new_session"])
        self.assertEqual(kw["cwd"], str(h.home))

    def test_failed_recovery_when_maple_exits_during_startup(self):
        h = Harness(self)
        h.kill_api()
        h.refuse()
        h.popen_exit = 1
        self.assertEqual(h.sup.tick(), "recover:failed")
        self.assertEqual(h.sup.api_state, "RECOVERING")
        self.assertEqual(h.sup.failures, 1)                  # not reset

    def test_failed_recovery_on_startup_timeout_leaves_process_alone(self):
        h = Harness(self, start_timeout=10)
        h.kill_api()
        h.refuse()
        self.assertEqual(h.sup.tick(), "recover:failed")
        self.assertEqual(h.signals, [])                       # a slow starter is not killed
        self.assertIn("timeout", " ".join(h.out).lower() + str(h.sup.last_probe))

    def test_start_timeout_default_is_600(self):
        self.assertEqual(hb.SERVER_START_TIMEOUT, 600)
        self.assertEqual(hb.Config().start_timeout, 600)


class LimitsAndBackoff(unittest.TestCase):
    def _failing_start(self, h):
        h.kill_api()
        h.refuse()
        h.popen_exit = 1

    def test_exponential_backoff_between_attempts(self):
        h = Harness(self, backoff_base=30)
        self._failing_start(h)
        self.assertEqual(h.sup.tick(), "recover:failed")      # attempt 1, next allowed in 30 s
        self.assertEqual(h.sup.tick(), "recover:backoff")     # immediately again -> held back
        h.clock.sleep(31)
        self.assertEqual(h.sup.tick(), "recover:failed")      # attempt 2, next allowed in 60 s
        h.clock.sleep(31)
        self.assertEqual(h.sup.tick(), "recover:backoff")
        h.clock.sleep(30)
        self.assertEqual(h.sup.tick(), "recover:failed")
        self.assertEqual(h.sup.attempts, 3)
        self.assertAlmostEqual(h.sup.next_restart_at - h.clock(), 120, delta=1)

    def test_backoff_is_capped(self):
        h = Harness(self, backoff_base=30, backoff_max=100, max_restarts=50)
        self._failing_start(h)
        for _ in range(6):
            h.sup.tick()
            h.clock.sleep(1000)
        self.assertLessEqual(h.sup.next_restart_at - h.clock() + 1000, 100 + 1)

    def test_restart_limit_stops_automatic_recovery(self):
        h = Harness(self, max_restarts=2, backoff_base=1)
        self._failing_start(h)
        self.assertEqual(h.sup.tick(), "recover:failed")
        h.clock.sleep(5)
        self.assertEqual(h.sup.tick(), "recover:failed")
        h.clock.sleep(5)
        self.assertEqual(h.sup.tick(), "recover:limit")
        self.assertTrue(h.sup.gave_up)
        self.assertEqual(h.sup.api_state, "GAVE_UP")
        self.assertEqual(len(h.popens), 2)
        h.clock.sleep(5)
        h.sup.tick()
        self.assertEqual(len(h.popens), 2)                    # still no new start
        self.assertTrue(any("restart limit" in line.lower() for line in h.out))

    def test_restart_window_expires(self):
        h = Harness(self, max_restarts=1, backoff_base=1, restart_window=100)
        self._failing_start(h)
        h.sup.tick()
        h.clock.sleep(5)
        self.assertEqual(h.sup.tick(), "recover:limit")
        h.clock.sleep(200)
        self.assertEqual(h.sup.tick(), "recover:failed")      # old restart aged out of the window

    def test_sigterm_ignored_means_blocked_never_sigkill(self):
        h = Harness(self)
        h.sigterm_works = False
        h.timeout()
        h.sup.tick()
        h.sup.tick()
        self.assertEqual(h.sup.tick(), "recover:blocked")
        self.assertEqual(h.sup.api_state, "BLOCKED")
        self.assertEqual([s for _, s in h.signals], [signal.SIGTERM])
        self.assertEqual(h.popens, [])
        self.assertIn("SIGKILL", h.sup.blocked_reason)

    def test_foreign_process_on_api_port_blocks_start(self):
        h = Harness(self)
        h.kill_api()
        h.refuse()
        h.port_busy = True
        self.assertEqual(h.sup.tick(), "recover:blocked")
        self.assertEqual(h.popens, [])
        self.assertIn("not MAPLE", h.sup.blocked_reason)

    def test_orphan_llama_server_is_reported_not_touched(self):
        h = Harness(self)
        h.kill_api()
        h.procs.append(hb.ProcInfo(555, 1, "llama-server", ("/x/llama-server", "--port", "8081")))
        h.refuse()
        h.popen_exit = 1
        h.sup.tick()
        self.assertNotIn(555, [p for p, _ in h.signals])
        log = (h.state / "heartbeat.log").read_text()
        self.assertIn("orphan_llama_server_pids", log)
        self.assertIn("555", log)


class ModelSwitchingAndInvalid(unittest.TestCase):
    def test_slow_health_during_model_switch_is_busy_not_a_failure(self):
        h = Harness(self, busy_grace=180)
        h.net.health = hb.HttpResult(200, health_body(loaded="gemma", switches=1), None)
        h.sup.tick()
        h.net.health = hb.HttpResult(200, health_body(loaded="deepseek", switches=2), None)    # a switch happened
        h.sup.tick()
        h.timeout()                                                                             # then /health stalls
        self.assertEqual(h.sup.tick(), "busy")
        self.assertEqual(h.sup.failures, 0)
        self.assertEqual(h.sup.api_state, "BUSY")
        self.assertEqual(h.popens, [])

    def test_inflight_request_makes_a_timeout_busy(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(200, health_body(loaded="qwen3", inflight=1, switches=1), None)
        h.sup.tick()
        h.timeout()
        self.assertEqual(h.sup.tick(), "busy")

    def test_busy_grace_expires_and_failures_count_again(self):
        h = Harness(self, busy_grace=100)
        h.net.health = hb.HttpResult(200, health_body(loaded="a", inflight=1), None)
        h.sup.tick()
        h.timeout()
        self.assertEqual(h.sup.tick(), "busy")
        h.clock.sleep(150)
        self.assertEqual(h.sup.tick(), "failing")
        self.assertEqual(h.sup.failures, 1)

    def test_busy_never_masks_a_dead_process(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(200, health_body(loaded="a", inflight=1), None)
        h.sup.tick()
        h.kill_api()
        h.refuse()
        h.popen_exit = 1
        self.assertEqual(h.sup.tick(), "recover:failed")    # recovery was attempted despite recent activity

    def test_no_model_loaded_is_still_healthy(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(200, health_body(loaded=None, switches=0), None)
        self.assertEqual(h.sup.tick(), "healthy")
        self.assertIn("none loaded", h.sup.format_report(h.sup.build_status()))

    def test_loaded_model_name_is_reported(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(200, health_body(loaded="qwen3", switches=1, healthy=True), None)
        h.sup.tick()
        self.assertIn("qwen3 (Name-qwen3)", h.sup.format_report(h.sup.build_status()))

    def test_invalid_json_from_health(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(200, None, "invalid_json")
        self.assertEqual(h.sup.tick(), "failing")
        self.assertEqual(h.sup.last_probe.health_state, "invalid")

    def test_html_error_page_is_not_healthy(self):
        self.assertEqual(hb.interpret_health(hb.HttpResult(200, {"hello": "world"}, None))[0], "invalid")
        self.assertEqual(hb.interpret_health(hb.HttpResult(200, ["a"], None))[0], "invalid")
        self.assertEqual(hb.interpret_health(hb.HttpResult(502, None, "invalid_json"))[0], "invalid")

    def test_unexpected_status_codes(self):
        self.assertEqual(hb.interpret_health(hb.HttpResult(500, health_body(), None))[0], "http_error")
        self.assertEqual(hb.interpret_health(hb.HttpResult(404, {}, None))[0], "http_error")

    def test_status_ok_but_wrong_shape(self):
        b = health_body()
        del b["runtime"]
        self.assertEqual(hb.interpret_health(hb.HttpResult(200, b, None))[0], "invalid")

    def test_models_endpoint_garbage(self):
        h = Harness(self)
        h.net.models = hb.HttpResult(200, {"data": "nope"}, None)
        self.assertEqual(h.sup.tick(), "failing")
        h.net.models = hb.HttpResult(None, None, "timeout")
        self.assertEqual(h.sup.tick(), "failing")

    def test_missing_model_file_is_degraded_and_not_restarted(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(503, health_body("degraded", missing=("gemma",)), None)
        for _ in range(5):
            self.assertEqual(h.sup.tick(), "healthy")
        self.assertEqual(h.sup.api_state, "DEGRADED")
        self.assertEqual(h.popens, [])
        self.assertIn("gemma", h.sup.last_probe.health_detail)

    def test_roster_mismatch_does_not_restart(self):
        h = Harness(self)
        h.net.models = hb.HttpResult(200, models_body(drop=("fast",)), None)
        self.assertEqual(h.sup.tick(), "healthy")
        self.assertEqual(h.sup.api_state, "DEGRADED")
        self.assertIn("fast", " ".join(h.sup.build_status()["warnings"]))

    def test_quant_alias_and_auto_are_recognised(self):
        r = hb.interpret_models(hb.HttpResult(200, models_body(), None))
        self.assertEqual(r[0], "ok")
        self.assertIn("auto", r[2]["ids"])
        self.assertEqual(r[2]["aliases"], {"quant": "qwen3"})
        bad = hb.interpret_models(hb.HttpResult(200, models_body(aliases={}), None))
        self.assertEqual(bad[0], "incomplete")
        self.assertEqual(bad[2]["bad_aliases"], ["quant"])


class ProcessIdentification(unittest.TestCase):
    def test_is_maple_api(self):
        P = lambda *c: hb.ProcInfo(1, 0, "python3", c)   # noqa: E731
        self.assertTrue(hb.is_maple_api(P("python3", "-s", "-m", "maple", "serve"), 8000))
        self.assertTrue(hb.is_maple_api(P("python3", "-m", "maple", "serve", "--port", "8000"), 8000))
        self.assertTrue(hb.is_maple_api(P("python3", "-m", "maple", "serve", "--port=9000"), 9000))
        self.assertFalse(hb.is_maple_api(P("python3", "-m", "maple", "serve", "--port", "9000"), 8000))
        self.assertFalse(hb.is_maple_api(P("python3", "-m", "maple", "status"), 8000))
        self.assertFalse(hb.is_maple_api(P("python3", "other.py"), 8000))
        self.assertFalse(hb.is_maple_api(P("python3", "-m", "maple", "serve", "--port", "x"), 8000))

    def test_llama_server_is_never_identified_as_the_api(self):
        p = hb.ProcInfo(9, 1, "llama-server", ("/x/llama-server", "--port", "8081", "-m", "maple"))
        self.assertFalse(hb.is_maple_api(p, 8081))
        self.assertFalse(hb.is_maple_api(p, 8000))

    def test_ssh_cloudflared_is_protected_and_reported(self):
        h = Harness(self)
        h.procs.append(hb.ProcInfo(1971, 1, "cloudflared", ("cloudflared", "tunnel", "--url", "ssh://localhost:2222")))
        h.sup.tick()
        st = h.sup.build_status()
        self.assertEqual(st["ssh_tunnel"]["cloudflared_pids"], [1971])
        self.assertTrue(st["ssh_tunnel"]["protected_not_managed"])
        self.assertIn("protected", h.sup.format_report(st))

    def test_a_recovery_never_signals_cloudflared_or_llama_server(self):
        h = Harness(self)
        h.procs += [hb.ProcInfo(1971, 1, "cloudflared", ("cloudflared", "tunnel", "--url", "ssh://localhost:2222")),
                    hb.ProcInfo(555, 100, "llama-server", ("/x/llama-server", "--port", "8081"))]
        h.timeout()
        h.sup.tick()
        h.sup.tick()
        h.popen_exit = 1
        h.sup.tick()
        self.assertEqual({p for p, _ in h.signals}, {100})

    def test_stop_refuses_if_process_no_longer_looks_like_maple(self):
        h = Harness(self)
        h.procs = [hb.ProcInfo(100, 1, "python3", ("python3", "evil.py"))]
        ok = h.sup.stop_maple_api(hb.ProcInfo(100, 1, "python3", API_CMD))
        self.assertFalse(ok)
        self.assertEqual(h.signals, [])

    def test_maple_start_does_not_pass_the_tunnel_api_key(self):
        h = Harness(self)
        h.kill_api()
        h.refuse()
        h.popen_exit = 1
        with mock.patch.dict(os.environ, {"API_KEY": "super-secret-test-key-0123456789"}):
            h.sup.tick()
        self.assertNotIn("API_KEY", h.popens[0][1]["env"])
        self.assertEqual(h.popens[0][1]["env"]["MAPLE_HOME"], str(h.home))

    def test_start_refused_when_script_missing(self):
        h = Harness(self)
        (h.home / "scripts" / "start_maple.sh").unlink()
        h.kill_api()
        h.refuse()
        self.assertEqual(h.sup.tick(), "recover:failed")
        self.assertEqual(h.popens, [])


class DuplicateSupervisor(unittest.TestCase):
    def test_second_instance_is_rejected(self):
        h = Harness(self)
        h.sup.acquire_lock()
        other = hb.Supervisor(h.cfg, pid_alive=lambda p: True, console=False)
        self.addCleanup(other.close)
        with mock.patch("os.getpid", return_value=99999), mock.patch.object(hb, "is_supervisor_pid", return_value=True):
            with self.assertRaises(hb.DuplicateSupervisor):
                other.acquire_lock()

    def test_stale_lock_from_dead_pid_is_taken_over(self):
        h = Harness(self)
        hb.atomic_write_json(h.state / "supervisor.lock", {"pid": 424242})
        h.sup.acquire_lock()
        self.assertEqual(json.loads((h.state / "supervisor.lock").read_text())["pid"], os.getpid())
        self.assertIn("stale_lock_removed", (h.state / "heartbeat.log").read_text())

    def test_lock_pid_reused_by_unrelated_process_is_not_a_duplicate(self):
        h = Harness(self)
        hb.atomic_write_json(h.state / "supervisor.lock", {"pid": 31337})
        h.sup.pid_alive = lambda p: True
        with mock.patch.object(hb, "is_supervisor_pid", return_value=False):
            h.sup.acquire_lock()                                 # the live pid is not a heartbeat: lock is stale

    def test_lock_released_on_shutdown(self):
        h = Harness(self)
        h.sup.acquire_lock()
        h.sup.shutdown()
        self.assertFalse((h.state / "supervisor.lock").exists())

    def test_main_returns_3_for_duplicate(self):
        h = Harness(self)
        live_other_pid = os.getppid()                     # a real, live pid that is not this process
        hb.atomic_write_json(h.state / "supervisor.lock", {"pid": live_other_pid})
        with mock.patch.object(hb, "is_supervisor_pid", return_value=True):
            rc = hb.main(["run", "--maple-home", str(h.home), "--state-dir", str(h.state)])
        self.assertEqual(rc, 3)


class ShutdownAndRun(unittest.TestCase):
    def _run_for(self, h, ticks):
        n = {"i": 0}
        real = h.clock.sleep

        def sleeper(s):
            real(s)
            n["i"] += 1
            if n["i"] >= ticks:
                h.sup.stopping = True
        h.sup.sleep = sleeper
        with mock.patch.object(h.sup, "install_signal_handlers"):
            return h.sup.run()

    def test_graceful_shutdown_leaves_maple_running_by_default(self):
        h = Harness(self)
        rc = self._run_for(h, 130)
        self.assertEqual(rc, 0)
        self.assertEqual(h.signals, [])
        self.assertFalse((h.state / "supervisor.lock").exists())
        st = json.loads((h.state / "status.json").read_text())
        self.assertEqual(st["supervisor"]["state"], "stopped")
        self.assertTrue(any("left running" in line for line in h.out))
        self.assertIn("supervisor_stopped", (h.state / "heartbeat.log").read_text())

    def test_stop_maple_on_exit_sends_sigterm_to_the_api_only(self):
        h = Harness(self, stop_maple_on_exit=True)
        self._run_for(h, 5)
        self.assertEqual(h.signals, [(100, signal.SIGTERM)])

    def test_periodic_status_report_every_5_minutes_and_json_lines(self):
        h = Harness(self, heartbeat_minutes=5, check_interval=60)
        self._run_for(h, 700)
        lines = [json.loads(x) for x in (h.state / "reports.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(lines), 3)                   # start + two 5-minute reports
        for r in lines:
            for key in ("timestamp", "maple_api", "llama_server", "gpu", "ram", "disk", "ssh_tunnel", "tunnel", "warnings"):
                self.assertIn(key, r)

    def test_health_checked_every_check_interval(self):
        h = Harness(self, check_interval=60, heartbeat_minutes=999)
        self._run_for(h, 245)
        health_calls = [u for u in h.net.calls if u.endswith("/health")]
        self.assertEqual(len(health_calls), 1 + 4)               # one initial probe + checks at 60,120,180,240 s

    def test_run_starts_maple_via_script_when_not_running(self):
        h = Harness(self)
        h.kill_api()
        h.refuse()

        def started():
            h.net.health = hb.HttpResult(200, health_body(), None)
            h.procs.append(hb.ProcInfo(777, 1, "python3", API_CMD))
        h.on_start = started
        self._run_for(h, 3)
        self.assertEqual(len(h.popens), 1)
        self.assertTrue(h.popens[0][0][1].endswith("start_maple.sh"))

    def test_no_start_option_only_watches(self):
        h = Harness(self, no_start=True)
        h.kill_api()
        h.refuse()
        self._run_for(h, 3)
        self.assertEqual(h.popens, [])

    def test_existing_healthy_maple_is_adopted_not_restarted(self):
        h = Harness(self)
        self._run_for(h, 3)
        self.assertEqual(h.popens, [])
        self.assertEqual(h.signals, [])


class SystemMonitoring(unittest.TestCase):
    def test_gpu_when_nvidia_smi_missing(self):
        r = hb.query_gpus(which=lambda n: None, exists=lambda p: False)
        self.assertFalse(r["available"])
        self.assertEqual(r["devices"], [])
        self.assertIn("not found", r["error"])

    def test_gpu_when_nvidia_smi_fails(self):
        run = lambda *a, **k: SimpleNamespace(returncode=9, stdout="", stderr="NVIDIA-SMI has failed\nmore")   # noqa: E731
        r = hb.query_gpus(run=run, which=lambda n: "/usr/bin/nvidia-smi", exists=lambda p: True)
        self.assertFalse(r["available"])
        self.assertIn("exit 9", r["error"])

    def test_gpu_when_nvidia_smi_times_out_or_oserror(self):
        import subprocess
        def boom(*a, **k): raise subprocess.TimeoutExpired("x", 10)   # noqa: E704
        self.assertFalse(hb.query_gpus(run=boom, which=lambda n: "/x/smi", exists=lambda p: True)["available"])
        def oserr(*a, **k): raise OSError("nope")   # noqa: E704
        self.assertFalse(hb.query_gpus(run=oserr, which=lambda n: "/x/smi", exists=lambda p: True)["available"])

    def test_gpu_parsing_two_t4s(self):
        out = "0, Tesla T4, 5189, 15360, 3\n1, Tesla T4, 0, 15360, 0\n"
        run = lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr="")   # noqa: E731
        r = hb.query_gpus(run=run, which=lambda n: "/x/smi", exists=lambda p: True)
        self.assertTrue(r["available"])
        self.assertEqual([d["mem_used_mib"] for d in r["devices"]], [5189, 0])
        self.assertEqual(r["devices"][0]["util_pct"], 3)

    def test_gpu_garbage_output(self):
        run = lambda *a, **k: SimpleNamespace(returncode=0, stdout="garbage\n", stderr="")   # noqa: E731
        r = hb.query_gpus(run=run, which=lambda n: "/x/smi", exists=lambda p: True)
        self.assertFalse(r["available"])

    def test_status_still_works_without_gpu(self):
        h = Harness(self)
        h.sup.tick()
        st = h.sup.build_status()
        self.assertFalse(st["gpu"]["available"])
        self.assertIn("GPU         : unavailable", h.sup.format_report(st))

    def test_ram_uses_memavailable(self):
        txt = "MemTotal:       32000000 kB\nMemFree:          100000 kB\nMemAvailable:   30000000 kB\n"
        r = hb.read_ram(txt)
        self.assertEqual(r["total_mib"], 31250)
        self.assertAlmostEqual(r["available_pct"], 93.8, places=1)
        self.assertIsNone(hb.read_ram("garbage"))

    def test_low_disk_and_ram_warnings(self):
        h = Harness(self)
        h.sup.disk_usage = lambda p: SimpleNamespace(free=int(0.5 * 2**30), total=100 * 2**30)
        h.sup.ram_read = lambda: {"total_mib": 1000, "available_mib": 50, "available_pct": 5.0}
        h.sup.tick()
        w = " ".join(h.sup.build_status()["warnings"])
        self.assertIn("low disk", w)
        self.assertIn("low available RAM", w)

    def test_gpu_memory_without_llama_server_warns_about_possible_orphan(self):
        h = Harness(self)
        h.sup.gpu_query = lambda libs=(): {"available": True, "devices": [{"index": 0, "name": "T4", "mem_used_mib": 5000, "mem_total_mib": 15360, "util_pct": 0}]}
        h.sup.tick()
        self.assertTrue(any("orphan" in w for w in h.sup.build_status()["warnings"]))


class ReportAndTunnelState(unittest.TestCase):
    def test_report_matches_requested_layout(self):
        h = Harness(self)
        h.net.health = hb.HttpResult(200, health_body(loaded="qwen3", switches=1, healthy=True), None)
        h.sup.tick()
        text = h.sup.format_report(h.sup.build_status())
        self.assertRegex(text, r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] Uptime: \d+ minutes")
        for label in ("Model       :", "Llama API   :", "Tunnel      :", "Endpoint    :"):
            self.assertIn(label, text)
        self.assertIn("Llama API   : ONLINE", text)
        self.assertIn("managed by MAPLE", text)

    def test_ssh_mode_reports_no_public_tunnel(self):
        h = Harness(self)
        h.sup.tick()
        text = h.sup.format_report(h.sup.build_status())
        self.assertIn("DISABLED", text)
        self.assertIn("ssh -N -L 8000:127.0.0.1:8000", text)

    def test_running_public_tunnel_state_is_shown(self):
        h = Harness(self)
        hb.atomic_write_json(h.state / "tunnel_state.json", {"mode": "public", "state": "running", "url": "https://abc-def.trycloudflare.com",
                                                             "auth": "bearer-proxy-verified", "updated_epoch": time.time()})
        h.sup.tick()
        text = h.sup.format_report(h.sup.build_status())
        self.assertIn("Tunnel      : RUNNING", text)
        self.assertIn("Endpoint    : https://abc-def.trycloudflare.com", text)
        self.assertIn("Bearer key required", text)

    def test_stale_tunnel_state_is_unknown_not_running(self):
        h = Harness(self)
        hb.atomic_write_json(h.state / "tunnel_state.json", {"mode": "public", "state": "running", "url": "https://x.trycloudflare.com",
                                                             "updated_epoch": time.time() - 3600})
        self.assertEqual(h.sup.read_tunnel_state()["state"], "unknown")

    def test_status_never_contains_an_api_key_field(self):
        h = Harness(self)
        with mock.patch.dict(os.environ, {"API_KEY": "super-secret-test-key-0123456789"}):
            h.sup.tick()
            blob = json.dumps(h.sup.build_status()) + h.sup.format_report(h.sup.build_status())
            h.sup.write_report(h.sup.build_status())
        self.assertNotIn("super-secret", blob)
        for f in h.state.iterdir():
            self.assertNotIn("super-secret", f.read_text(errors="replace"))


class ConfigTests(unittest.TestCase):
    def test_defaults_match_the_requested_values(self):
        self.assertEqual((hb.API_HOST, hb.API_PORT, hb.LLAMA_PORT, hb.CHECK_INTERVAL, hb.HEARTBEAT_MINUTES, hb.SERVER_START_TIMEOUT, hb.MAX_API_FAILURES),
                         ("127.0.0.1", 8000, 8081, 60, 5, 600, 3))
        self.assertEqual(hb.EXPECTED_MODELS, ("deepseek", "gemma", "qwen3", "fast"))
        self.assertEqual(hb.ALIASES, {"quant": "qwen3"})

    def test_stop_timeout_default_is_180(self):
        self.assertEqual(hb.STOP_TIMEOUT, 180)
        self.assertEqual(hb.Config().stop_timeout, 180)

    def test_stop_timeout_cli_env_and_priority(self):
        ns = hb.build_parser().parse_args(["run", "--maple-home", tempfile.gettempdir()])
        self.assertEqual(hb.config_from_args(ns).stop_timeout, 180)
        with mock.patch.dict(os.environ, {"MAPLE_SUP_STOP_TIMEOUT": "240"}):
            self.assertEqual(hb.config_from_args(ns).stop_timeout, 240)
            ns2 = hb.build_parser().parse_args(["run", "--maple-home", tempfile.gettempdir(), "--stop-timeout", "90"])
            self.assertEqual(hb.config_from_args(ns2).stop_timeout, 90)          # CLI beats env
        ns3 = hb.build_parser().parse_args(["stop", "--maple-home", tempfile.gettempdir(), "--stop-timeout", "30"])
        self.assertEqual(hb.config_from_args(ns3).stop_timeout, 30)             # also on the stop command

    def test_stop_timeout_invalid_values(self):
        with self.assertRaises(SystemExit):
            hb.Config(stop_timeout=5).validate()
        ns = hb.build_parser().parse_args(["run", "--maple-home", tempfile.gettempdir()])
        with mock.patch.dict(os.environ, {"MAPLE_SUP_STOP_TIMEOUT": "soon"}):
            with self.assertRaises(SystemExit):
                hb.config_from_args(ns)

    def test_slow_exit_inside_180s_is_success_not_blocked(self):
        h = Harness(self, stop_timeout=180)
        h.sigterm_works = False                                   # API keeps running (finishing a long request) ...
        real_signal = h._signal
        t0 = h.clock()

        def exits_after_100s(pid, sig):
            real_signal(pid, sig)
            h.sup.sleep = lambda s: (h.clock.sleep(s), h.procs.clear() if h.clock() - t0 >= 100 else None)[0]
        h.sup.send_signal = exits_after_100s
        api = h.procs[0]
        self.assertTrue(h.sup.stop_maple_api(api))                 # ... and exits after 100 s: inside the 180 s budget
        self.assertEqual([s for _, s in h.signals], [signal.SIGTERM])
        self.assertGreaterEqual(h.clock() - t0, 100)

    def test_non_loopback_host_rejected(self):
        with self.assertRaises(SystemExit):
            hb.Config(api_host="0.0.0.0").validate()

    def test_same_api_and_llama_port_rejected(self):
        with self.assertRaises(SystemExit):
            hb.Config(api_port=8081, llama_port=8081).validate()

    def test_env_and_cli_override(self):
        ns = hb.build_parser().parse_args(["run", "--maple-home", tempfile.gettempdir(), "--check-interval", "30"])
        with mock.patch.dict(os.environ, {"MAPLE_SUP_API_PORT": "9100", "MAPLE_SUP_CHECK_INTERVAL": "99"}):
            cfg = hb.config_from_args(ns)
        self.assertEqual(cfg.api_port, 9100)          # from env
        self.assertEqual(cfg.check_interval, 30)      # CLI beats env
        self.assertEqual(cfg.api_url, "http://127.0.0.1:9100")

    def test_bad_env_value(self):
        ns = hb.build_parser().parse_args(["run", "--maple-home", tempfile.gettempdir()])
        with mock.patch.dict(os.environ, {"MAPLE_SUP_API_PORT": "abc"}):
            with self.assertRaises(SystemExit):
                hb.config_from_args(ns)

    def test_maple_home_autodetect(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "scripts").mkdir()
            (Path(t) / "scripts" / "start_maple.sh").write_text("")
            sub = Path(t) / "tools" / "x"
            sub.mkdir(parents=True)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MAPLE_HOME", None)
                self.assertEqual(hb.find_maple_home(start=sub / "maple_heartbeat.py"), Path(t).resolve())


class NoStartIsWatchOnly(unittest.TestCase):
    """--no-start must prevent EVERY automatic start, restart and stop of MAPLE (regression tests)."""

    def _healed(self, h):
        def started():
            h.net.health = hb.HttpResult(200, health_body(), None)
            h.procs.append(hb.ProcInfo(777, 1, "python3", API_CMD))
        h.on_start = started                       # if anything DID launch MAPLE, the API would come back; popens would show it

    def _assert_untouched(self, h):
        self.assertEqual(h.popens, [], "MAPLE was launched under --no-start")
        self.assertEqual(h.signals, [], "a signal was sent under --no-start")

    def test_process_exit_does_not_restart(self):
        h = Harness(self, no_start=True)
        self._healed(h)
        h.kill_api()
        h.refuse()
        for _ in range(6):
            self.assertEqual(h.sup.tick(), "recover:disabled")
            h.clock.sleep(60)
        self._assert_untouched(h)
        self.assertEqual(h.sup.attempts, 0)
        self.assertEqual(h.sup.restart_times, [])

    def test_three_consecutive_failures_with_live_process_do_not_restart_or_signal(self):
        h = Harness(self, no_start=True)
        self._healed(h)
        h.timeout()                                # process alive, /health silent
        results = [h.sup.tick() for _ in range(8)]
        self.assertEqual(results[:2], ["failing", "failing"])
        self.assertTrue(all(r == "recover:disabled" for r in results[2:]), results)
        self._assert_untouched(h)                  # in particular: no SIGTERM to the live API, no new Popen

    def test_no_start_survives_long_outage_backoff_expiry_and_limit_logic(self):
        h = Harness(self, no_start=True, max_restarts=1)
        self._healed(h)
        h.kill_api()
        h.refuse()
        for _ in range(30):
            h.sup.tick()
            h.clock.sleep(700)                     # past every backoff window
        self._assert_untouched(h)
        self.assertNotEqual(h.sup.api_state, "RECOVERING")

    def test_notice_is_logged_once_and_state_is_unhealthy(self):
        h = Harness(self, no_start=True)
        h.kill_api()
        h.refuse()
        for _ in range(5):
            h.sup.tick()
        self.assertEqual(sum("WATCH-ONLY" in line for line in h.out), 1)
        self.assertEqual(h.sup.api_state, "UNHEALTHY")
        self.assertEqual(h.sup.failures, 5)        # failures are still counted and reported
        log = (h.state / "heartbeat.log").read_text()
        self.assertEqual(log.count("recovery_disabled_watch_only"), 1)
        self.assertNotIn("recovery_attempt", log)

    def test_start_maple_itself_refuses_under_no_start(self):
        h = Harness(self, no_start=True)
        h.kill_api()
        res = h.sup.start_maple()
        self.assertFalse(res.ok)
        self.assertIn("--no-start", res.detail)
        self._assert_untouched(h)

    def test_initial_launch_is_skipped(self):
        h = Harness(self, no_start=True)
        self._healed(h)
        h.kill_api()
        h.refuse()
        ShutdownAndRun._run_for(self, h, 4)
        self._assert_untouched(h)

    def test_stop_maple_on_exit_is_ignored_in_watch_only_mode(self):
        h = Harness(self, no_start=True, stop_maple_on_exit=True)
        ShutdownAndRun._run_for(self, h, 3)
        self._assert_untouched(h)
        self.assertTrue(any("left running" in line for line in h.out))

    def test_cli_flag_sets_watch_only(self):
        ns = hb.build_parser().parse_args(["run", "--no-start", "--maple-home", self.id() and "."])
        self.assertTrue(ns.no_start)
        self.assertTrue(hb.config_from_args(ns).no_start)
        self.assertFalse(hb.config_from_args(hb.build_parser().parse_args(["run", "--maple-home", "."])).no_start)

    def test_without_the_flag_recovery_still_works(self):
        h = Harness(self)                          # control: the same scenario DOES recover when not watch-only
        self._healed(h)
        h.kill_api()
        h.refuse()
        self.assertEqual(h.sup.tick(), "recover:recovered")
        self.assertEqual(len(h.popens), 1)


class StaticSafetyChecks(unittest.TestCase):
    SRC = (HERE.parent / "maple_heartbeat.py").read_text(encoding="utf-8")

    def code_only(self):
        # drop comments and docstring-ish lines so prose about "never SIGKILL" does not trigger
        return "\n".join(l for l in self.SRC.splitlines() if not l.strip().startswith("#"))

    def test_never_uses_sigkill_or_kill_by_name(self):
        code = self.code_only()
        self.assertNotRegex(code, r"signal\.SIGKILL|(?<!os)\.kill\(")      # os.kill(pid, SIGTERM) is the only allowed form
        for bad in ("pkill", "killall", "taskkill", "os.killpg"):
            self.assertNotIn(bad, code)
        for call in re.findall(r"send_signal\(([^)]*)\)", code):
            if "def " in call or ":" in call or call.strip() in ("pid, sig", ""):     # skip the signature itself
                continue
            self.assertIn("SIGTERM", call)

    def test_never_launches_llama_server_directly(self):
        self.assertNotRegex(self.SRC, r"bin/llama-server|\"llama-server\"\s*,\s*\"--|Popen\(\s*\[\s*[^\]]*llama")
        self.assertIn("start_maple.sh", self.SRC)

    def test_only_get_requests(self):
        self.assertNotRegex(self.SRC, r"method\s*=\s*['\"]POST|chat/completions")

    def test_no_install_or_network_install_commands(self):
        for bad in ("pip install", "apt-get", "subprocess.run([\"pip"):
            self.assertNotIn(bad, self.SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
