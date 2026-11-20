"""Model lifecycle on top of native llama-server (one model resident at a time).

Sequential loading/unloading with resource-aware placement. No inference code
lives here: requests are proxied to llama-server's OpenAI-compatible API."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from . import resource_manager as rm
from .config import build_env, llama_bin_dir, load_runtime_config, maple_home
from .registry import ModelMissing, Registry

log = logging.getLogger("maple.runtime")


class LoadError(Exception):
    pass


def http_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


class LlamaServer:
    def __init__(self, model: dict, cfg: dict, placement: dict, ctx: int, log_dir: Path):
        self.model, self.cfg, self.placement, self.ctx = model, cfg, placement, ctx
        self.port = cfg["llama_port"]
        self.host = cfg["host"]
        self.log_path = Path(log_dir) / ("llama-%s.log" % model["id"])
        self.proc = None
        self.cmd = []

    def build_cmd(self):
        mc = self.cfg.get("models", {}).get(self.model["id"], {})
        c = [str(llama_bin_dir() / "llama-server"), "-m", self.model["model_path"], "--host", self.host,
             "--port", str(self.port), "-c", str(self.ctx), "-np", str(self.cfg["parallel_slots"]),
             "-fa", str(mc.get("flash_attn", self.cfg["flash_attn"])), "--alias", self.model["id"], "--metrics"]
        devs = self.placement["devices"]
        if devs:
            c += ["-ngl", str(mc.get("gpu_layers", self.cfg["gpu_layers"])),
                  "-dev", ",".join("CUDA%d" % d for d in devs)]
            if len(devs) > 1:
                c += ["-sm", "layer"]
        else:
            c += ["-ngl", "0"]
        if self.cfg.get("log_verbosity") is not None:
            c += ["-lv", str(self.cfg["log_verbosity"])]
        c += [str(x) for x in mc.get("extra_args", [])]
        self.cmd = c
        return c

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.build_cmd()
        exe = cmd[0]
        if not os.path.isfile(exe):
            raise LoadError("llama-server not found at %s. Restore llama.cpp with scripts/restore_kaggle.py (never recompiled automatically)." % exe)
        if not os.access(exe, os.X_OK):
            raise LoadError("llama-server at %s is not executable (permission bits are stripped when a Kaggle session is "
                            "snapshotted/restored). Run scripts/restore_kaggle.py, which re-applies the modes recorded in "
                            "MAPLE-llamacpp-cuda.tar.gz." % exe)
        logf = open(self.log_path, "wb")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=build_env(self.cfg),
                                     start_new_session=True)
        base = "http://%s:%d" % (self.host, self.port)
        deadline = t0 + self.cfg["startup_timeout_s"]
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise LoadError("llama-server exited with code %s while loading '%s'. Last log lines:\n%s"
                                % (self.proc.returncode, self.model["id"], self.log_tail(25)))
            try:
                code, _ = http_json(base + "/health", timeout=2)
                if code == 200:
                    return {"load_time_s": round(time.time() - t0, 2), "pid": self.proc.pid, "cmd": cmd}
            except (urllib.error.URLError, OSError, ValueError):
                pass
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        raise LoadError("timeout (%ss) waiting for llama-server '%s'. Last log lines:\n%s"
                        % (self.cfg["startup_timeout_s"], self.model["id"], self.log_tail(25)))

    def log_tail(self, n=25):
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def healthy(self):
        if not self.proc or self.proc.poll() is not None:
            return False
        try:
            return http_json("http://%s:%d/health" % (self.host, self.port), timeout=3)[0] == 200
        except Exception:
            return False

    def stop(self):
        p = self.proc
        if not p or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            p.terminate()
        try:
            p.wait(timeout=self.cfg["unload_timeout_s"])
        except subprocess.TimeoutExpired:
            log.warning("llama-server did not exit on SIGTERM; sending SIGKILL")
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                p.kill()
            p.wait(timeout=10)


class ModelManager:
    """Keeps at most one llama-server resident; switches only when idle."""

    def __init__(self, registry: Registry | None = None, cfg: dict | None = None, log_dir=None):
        self.registry = registry or Registry()
        self.cfg = cfg or load_runtime_config()
        self.log_dir = Path(log_dir) if log_dir else maple_home() / "logs"
        self.cond = threading.Condition()
        self.current = None          # model id
        self.server = None
        self.inflight = 0
        self.switches = 0
        self.last_load = {}
        self.kv_override = {}        # model id -> measured KV bytes/token (set by benchmark.py)

    def ctx_for(self, model_id: str) -> int:
        return int(self.cfg.get("models", {}).get(model_id, {}).get("ctx", self.cfg["default_ctx"]))

    def _load(self, model_id: str, ctx: int | None = None):
        m = self.registry.require(model_id)        # raises ModelMissing with exact path
        ctx = ctx or self.ctx_for(m["id"])
        gpus = rm.gpu_info(self.cfg)
        kv_bpt = self.kv_override.get(m["id"]) or self.registry.recorded(m["id"]).get("tested", {}).get("kv_bytes_per_token")
        kw = {"kv_bytes_per_token": int(kv_bpt)} if kv_bpt else {}
        placement = rm.plan_placement(os.path.getsize(m["model_path"]), ctx, gpus, **kw)
        log.info("loading %s ctx=%d placement=%s", m["id"], ctx, placement)
        srv = LlamaServer(m, self.cfg, placement, ctx, self.log_dir)
        info = srv.start()
        after = rm.gpu_info(self.cfg)
        self.server, self.current = srv, m["id"]
        self.switches += 1
        self.last_load = {"model": m["id"], "ctx": ctx, "placement": placement, "gpu_before": gpus,
                          "gpu_after": after, **info}
        log.info("loaded %s in %.1fs", m["id"], info["load_time_s"])
        return self.last_load

    def unload(self):
        with self.cond:
            while self.inflight:
                self.cond.wait()
            self._unload_locked()

    def _unload_locked(self):
        if self.server:
            log.info("unloading %s", self.current)
            self.server.stop()
        self.server, self.current = None, None

    @contextmanager
    def use(self, model_id: str, ctx: int | None = None):
        """Context manager yielding the base URL of a llama-server hosting model_id.
        Switching waits until in-flight requests on the other model finish."""
        mid = self.registry.get(model_id)["id"]
        with self.cond:
            while self.current != mid and self.inflight > 0:
                self.cond.wait()
            if self.current == mid and self.server and not self.server.healthy():
                log.warning("resident model %s unhealthy; reloading", mid)
                self._unload_locked()
            if self.current != mid:
                self._unload_locked()
                self._load(mid, ctx)
            self.inflight += 1
        try:
            yield "http://%s:%d" % (self.cfg["host"], self.cfg["llama_port"])
        finally:
            with self.cond:
                self.inflight -= 1
                self.cond.notify_all()

    def status(self) -> dict:
        return {"loaded": self.current, "inflight": self.inflight, "switches": self.switches,
                "healthy": bool(self.server and self.server.healthy()), "last_load": self.last_load}
