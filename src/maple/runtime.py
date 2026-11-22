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


# ---------------------------------------------------------------------------
# PROPOSAL (not applied): several resident llama-server instances.
# Enabled only when runtime.yaml has  "residency": {"mode": "multi", ...}.
# Removing that key restores the single-model ModelManager above (rollback).
# ---------------------------------------------------------------------------
class QueueTimeout(LoadError):
    pass


class MultiModelManager(ModelManager):
    """Keeps several models resident at once (one llama-server per model, one request at a time per model
    because of -np 1). Different models serve in parallel. Placement is memory-aware: a model is loaded on its
    preferred GPU only if free VRAM minus its measured size still leaves `vram_margin_mib`; otherwise the least
    recently used IDLE model on that GPU is evicted; a busy model is never evicted. If nothing fits, the request
    waits (bounded) and then fails with a clear error instead of risking a CUDA out-of-memory."""

    def __init__(self, registry=None, cfg=None, log_dir=None):
        super().__init__(registry, cfg, log_dir)
        rc = self.cfg.get("residency") or {}
        self.layout = rc.get("layout", {})                  # model -> {"gpu": 0|1, "port": 809x}
        self.vram_mib = rc.get("vram_mib", {})              # measured solo VRAM per model (MiB)
        self.margin = int(rc.get("vram_margin_mib", 1200))
        self.queue_timeout = float(rc.get("queue_timeout_s", 120))
        self.fail_cooldown = float(rc.get("fail_cooldown_s", 60))
        self.servers, self.busy, self.last_used, self.failed, self.gpu_of = {}, {}, {}, {}, {}
        self.waiting = {}                                   # model -> requests currently queued for it
        self._wlock = threading.Lock()
        self.closing = False

    # -- helpers (all called with self.cond held) ---------------------------------------------------------
    def _port(self, mid):
        return int(self.layout.get(mid, {}).get("port") or (self.cfg["llama_port"] + 1 + list(self.registry.ids()).index(mid)))

    def _need_mib(self, m, ctx):
        if m["id"] in self.vram_mib:
            return int(self.vram_mib[m["id"]])
        return rm.plan_placement(os.path.getsize(m["model_path"]), ctx, [])["estimate_mib"]

    def _prefs(self, mid, gpus):
        want = self.layout.get(mid, {}).get("gpu")
        want = [want] if isinstance(want, int) else list(want or [])
        rest = [g["index"] for g in sorted(gpus, key=lambda g: -g["mem_free_mib"]) if g["index"] not in want]
        return want + rest

    def _idle_lru_on(self, gpu, exclude):
        c = [x for x, g in self.gpu_of.items() if g == gpu and x != exclude and not self.busy.get(x)]
        return min(c, key=lambda x: self.last_used.get(x, 0)) if c else None

    def _drop(self, mid, why):
        srv = self.servers.pop(mid, None)
        self.gpu_of.pop(mid, None)
        if srv:
            log.info("stopping %s (%s)", mid, why)
            srv.stop()

    def _load_one(self, mid):
        m = self.registry.require(mid)
        ctx = self.ctx_for(mid)
        need = self._need_mib(m, ctx)
        gpus = rm.gpu_info(self.cfg)
        if not gpus:
            raise LoadError("no GPU information available; refusing to load '%s' blindly" % mid)
        for gi in self._prefs(mid, gpus):
            while True:                      # evict idle LRU models on this GPU until it fits or none are left
                free = next(g["mem_free_mib"] for g in gpus if g["index"] == gi)
                if free - need >= self.margin:
                    break
                victim = self._idle_lru_on(gi, mid)
                if victim is None:
                    break
                self._drop(victim, "evicted to fit '%s' on GPU %d" % (mid, gi))
                time.sleep(0.5)              # let the driver release memory before re-reading it
                gpus = rm.gpu_info(self.cfg)
            free = next(g["mem_free_mib"] for g in gpus if g["index"] == gi)
            if free - need < self.margin:
                continue
            placement = {"devices": [gi], "split_mode": "none", "estimate_mib": need,
                         "reason": "resident layout: %d MiB free on GPU %d, need %d + margin %d" % (free, gi, need, self.margin)}
            srv = LlamaServer(m, dict(self.cfg, llama_port=self._port(mid)), placement, ctx, self.log_dir)
            info = srv.start()
            self.servers[mid], self.gpu_of[mid] = srv, gi
            self.switches += 1
            self.last_load = {"model": mid, "ctx": ctx, "placement": placement, **info}
            return
        raise LoadError("insufficient VRAM for '%s' (needs %d MiB + %d MiB margin; all GPUs hold busy models). "
                        "Try again later." % (mid, need, self.margin))

    # -- public API (same contract as ModelManager) -------------------------------------------------------
    @contextmanager
    def use(self, model_id: str, ctx: int | None = None):
        mid = self.registry.get(model_id)["id"]
        deadline = time.time() + self.queue_timeout
        with self._wlock:
            self.waiting[mid] = self.waiting.get(mid, 0) + 1
        counted = [True]

        def _dequeue():
            if counted[0]:
                counted[0] = False
                with self._wlock:
                    self.waiting[mid] = max(0, self.waiting.get(mid, 1) - 1)
        try:
            yield from self._use_locked(mid, deadline, _dequeue)
        finally:
            _dequeue()

    def _use_locked(self, mid, deadline, _dequeue):
        with self.cond:
            while True:
                if time.time() > deadline:
                    raise QueueTimeout("timed out after %.0fs waiting for model '%s'" % (self.queue_timeout, mid))
                srv = self.servers.get(mid)
                if srv and not srv.healthy() and not self.busy.get(mid):
                    log.warning("resident model %s unhealthy; reloading", mid)
                    self._drop(mid, "unhealthy")
                    srv = None
                if srv and not self.busy.get(mid):
                    self.busy[mid] = True
                    _dequeue()
                    break
                if not srv:
                    failed = self.failed.get(mid)
                    if failed and time.time() - failed[0] < self.fail_cooldown:
                        raise LoadError("model '%s' failed to load recently (%s); retry after cooldown" % (mid, failed[1][:200]))
                    try:
                        self._load_one(mid)
                        self.failed.pop(mid, None)
                        deadline = max(deadline, time.time() + 10)   # a slow (but successful) load must not eat the queue budget
                        continue
                    except LoadError as e:
                        if "insufficient VRAM" not in str(e):
                            self.failed[mid] = (time.time(), str(e))
                            raise
                left = deadline - time.time()
                if left <= 0:
                    raise QueueTimeout("timed out after %.0fs waiting for model '%s'" % (self.queue_timeout, mid))
                self.cond.wait(timeout=min(left, 2.0))
            self.inflight += 1
            self.last_used[mid] = time.time()
        try:
            yield "http://%s:%d" % (self.cfg["host"], self._port(mid))
        finally:
            with self.cond:
                self.busy[mid] = False
                self.inflight -= 1
                self.last_used[mid] = time.time()
                self.cond.notify_all()

    def snapshot(self) -> dict:
        """Cheap routing view (no lock, no health probes): resident models with busy/queue state and models that
        are cooling down after a failed load."""
        now = time.time()
        res = {}
        for m, s in list(self.servers.items()):
            p = getattr(s, "proc", False)
            if p is False or (p is not None and p.poll() is None):     # p False: server object without a process handle
                res[m] = {"gpu": self.gpu_of.get(m), "busy": bool(self.busy.get(m)), "waiting": self.waiting.get(m, 0)}
        failed = {m: round(self.fail_cooldown - (now - t), 1) for m, (t, _) in list(self.failed.items()) if now - t < self.fail_cooldown}
        return {"resident": res, "failed": failed, "waiting": dict(self.waiting)}

    def preload(self, ids):
        with self.cond:
            for mid in ids or []:
                if self.closing:
                    return
                if mid not in self.servers:
                    try:
                        self._load_one(self.registry.get(mid)["id"])
                    except Exception as e:  # noqa - preload is best effort
                        log.error("preload of %s failed: %s", mid, e)

    def unload(self):
        self.closing = True
        with self.cond:
            while self.inflight:
                self.cond.wait()
            for mid in list(self.servers):
                self._drop(mid, "shutdown")

    def status(self) -> dict:
        res = {m: {"gpu": self.gpu_of.get(m), "port": self._port(m), "busy": bool(self.busy.get(m)),
                   "healthy": s.healthy()} for m, s in self.servers.items()}
        last = max(self.last_used, key=self.last_used.get) if self.last_used else None
        return {"loaded": last, "inflight": self.inflight, "switches": self.switches,
                "healthy": any(v["healthy"] for v in res.values()), "resident": res, "last_load": self.last_load}


def make_manager(registry, cfg):
    if (cfg.get("residency") or {}).get("mode") == "multi":
        return MultiModelManager(registry, cfg)
    return ModelManager(registry, cfg)
