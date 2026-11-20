#!/usr/bin/env python3
"""maple_heartbeat.py - health supervisor for a MAPLE-llama.cpp API.

Adapted from the "Qwen3.8-27B heartbeat & auto-recovery" notebook cell. What changed in principle:
the old script owned a single ``llama-server`` process. MAPLE is different: the MAPLE API (``python -m maple serve``,
started by ``scripts/start_maple.sh``) owns ``llama-server`` and loads/unloads models on demand. So this supervisor
monitors and recovers the MAPLE API ONLY. It never launches, signals or restarts ``llama-server`` itself.

What it does
  * every CHECK_INTERVAL s: GET /health and GET /v1/models (never an inference request), process check, backoff-limited recovery
  * every HEARTBEAT_MINUTES: status report (console + JSON lines) with GPU / RAM / disk figures
  * recovery = graceful SIGTERM of the verified MAPLE API process (its own shutdown handler unloads llama-server),
    then ``scripts/start_maple.sh``. No SIGKILL: killing the API would orphan llama-server on the GPU.
  * single-instance lock, restart limits, exponential backoff, graceful shutdown

Linux host (Kaggle) only for process control. Standard library only; nothing to install.
Usage:  python maple_heartbeat.py run | status [--json] | stop [--all]      (see README.md)
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import logging.handlers
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

__version__ = "0.1.0"

# --------------------------------------------------------------------------- defaults (override: CLI flag > env > default)
API_HOST = "127.0.0.1"
API_PORT = 8000
LLAMA_PORT = 8081            # internal llama-server port; managed by MAPLE, only observed here
CHECK_INTERVAL = 60
HEARTBEAT_MINUTES = 5
SERVER_START_TIMEOUT = 600
MAX_API_FAILURES = 3
REQUEST_TIMEOUT = 15
MAX_RESTARTS = 5             # per RESTART_WINDOW
RESTART_WINDOW = 3600
BACKOFF_BASE = 30            # 30, 60, 120, ... capped at BACKOFF_MAX between consecutive restart attempts
BACKOFF_MAX = 600
BUSY_GRACE = 180             # a slow /health right after model activity is "busy", not a crash, for this long
STABLE_AFTER = 300           # healthy this long -> backoff counter resets
MIN_DISK_FREE_GIB = 1.5
STOP_TIMEOUT = 180            # SIGTERM wait: MAPLE's shutdown waits for in-flight requests before unloading llama-server
EXPECTED_MODELS = ("deepseek", "gemma", "qwen3", "fast")
ALIASES = {"quant": "qwen3"}
AUTO_ID = "auto"
TUNNEL_STATE_FILE = "tunnel_state.json"      # written by maple_tunnel.py, read-only here
ENV_PREFIX = "MAPLE_SUP_"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


# --------------------------------------------------------------------------- small helpers
def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def tail_text(path: Optional[Path], lines: int = 20, max_chars: int = 2500) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError):
        return ""
    return "\n".join(text.splitlines()[-lines:])[-max_chars:]


# --------------------------------------------------------------------------- configuration
def find_maple_home(explicit: Optional[str] = None, start: Optional[Path] = None) -> Optional[Path]:
    """--maple-home, else $MAPLE_HOME, else a parent folder of this script that has scripts/start_maple.sh."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("MAPLE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(start or __file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "scripts" / "start_maple.sh").is_file():
            return d
    return None


@dataclasses.dataclass
class Config:
    maple_home: Optional[Path] = None
    state_dir: Path = Path("maple_supervisor")
    api_host: str = API_HOST
    api_port: int = API_PORT
    llama_port: int = LLAMA_PORT
    check_interval: float = CHECK_INTERVAL
    heartbeat_minutes: float = HEARTBEAT_MINUTES
    start_timeout: float = SERVER_START_TIMEOUT
    max_api_failures: int = MAX_API_FAILURES
    request_timeout: float = REQUEST_TIMEOUT
    max_restarts: int = MAX_RESTARTS
    restart_window: float = RESTART_WINDOW
    backoff_base: float = BACKOFF_BASE
    backoff_max: float = BACKOFF_MAX
    busy_grace: float = BUSY_GRACE
    stable_after: float = STABLE_AFTER
    min_disk_gib: float = MIN_DISK_FREE_GIB
    stop_timeout: float = STOP_TIMEOUT
    no_start: bool = False
    stop_maple_on_exit: bool = False
    nvidia_lib_dirs: tuple = ("/usr/local/nvidia/lib64",)
    disk_path: Optional[Path] = None

    @property
    def api_url(self) -> str:
        return "http://%s:%d" % (self.api_host, self.api_port)

    @property
    def start_script(self) -> Optional[Path]:
        return (self.maple_home / "scripts" / "start_maple.sh") if self.maple_home else None

    def validate(self) -> "Config":
        if self.api_host not in LOOPBACK:
            raise SystemExit("refusing api_host %r: MAPLE has no authentication, keep it on loopback (127.0.0.1)" % self.api_host)
        if self.api_port == self.llama_port:
            raise SystemExit("api_port and llama_port must differ (MAPLE API %d, internal llama-server %d)" % (self.api_port, self.llama_port))
        if self.check_interval <= 0 or self.max_api_failures < 1 or self.max_restarts < 0:
            raise SystemExit("check_interval must be > 0, max_api_failures >= 1, max_restarts >= 0")
        if self.stop_timeout < 10:
            raise SystemExit("stop_timeout must be >= 10 s (MAPLE waits for in-flight requests before it exits)")
        return self


def _pick(cli: Any, env_name: str, default: Any, cast: Callable = str) -> Any:
    if cli is not None:
        return cast(cli)
    raw = os.environ.get(ENV_PREFIX + env_name)
    if raw not in (None, ""):
        try:
            return cast(raw)
        except ValueError:
            raise SystemExit("invalid value for %s%s: %r" % (ENV_PREFIX, env_name, raw))
    return default


def config_from_args(ns: argparse.Namespace) -> Config:
    home = find_maple_home(getattr(ns, "maple_home", None))
    state = getattr(ns, "state_dir", None) or os.environ.get(ENV_PREFIX + "STATE_DIR")
    if state:
        state_dir = Path(state).expanduser().resolve()
    elif home:
        state_dir = home / "logs" / "supervisor"
    else:
        state_dir = Path.cwd() / "maple_supervisor"
    g = lambda n: getattr(ns, n, None)   # noqa: E731
    libs = tuple(g("nvidia_lib_dir") or ()) or Config.nvidia_lib_dirs
    return Config(
        maple_home=home, state_dir=state_dir,
        api_host=_pick(g("api_host"), "API_HOST", API_HOST),
        api_port=_pick(g("api_port"), "API_PORT", API_PORT, int),
        llama_port=_pick(g("llama_port"), "LLAMA_PORT", LLAMA_PORT, int),
        check_interval=_pick(g("check_interval"), "CHECK_INTERVAL", CHECK_INTERVAL, float),
        heartbeat_minutes=_pick(g("heartbeat_minutes"), "HEARTBEAT_MINUTES", HEARTBEAT_MINUTES, float),
        start_timeout=_pick(g("start_timeout"), "SERVER_START_TIMEOUT", SERVER_START_TIMEOUT, float),
        max_api_failures=_pick(g("max_api_failures"), "MAX_API_FAILURES", MAX_API_FAILURES, int),
        request_timeout=_pick(g("request_timeout"), "REQUEST_TIMEOUT", REQUEST_TIMEOUT, float),
        stop_timeout=_pick(g("stop_timeout"), "STOP_TIMEOUT", STOP_TIMEOUT, float),
        max_restarts=_pick(g("max_restarts"), "MAX_RESTARTS", MAX_RESTARTS, int),
        no_start=bool(g("no_start")), stop_maple_on_exit=bool(g("stop_maple_on_exit")),
        nvidia_lib_dirs=libs,
        disk_path=home if home else Path.cwd(),
    ).validate()


# --------------------------------------------------------------------------- HTTP (stdlib, no proxies, GET only)
class HttpResult(NamedTuple):
    status: Optional[int]
    body: Any
    error: Optional[str]      # None | refused | timeout | unreachable | invalid_json


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _classify_exc(exc: BaseException) -> str:
    if isinstance(exc, ConnectionRefusedError):
        return "refused"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    return "unreachable"


def http_get_json(url: str, timeout: float) -> HttpResult:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "maple-heartbeat/" + __version__})
    status, raw = None, b""
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            status, raw = r.status, r.read(1 << 20)
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            raw = e.read(1 << 20)
        except Exception:   # noqa: BLE001
            raw = b""
    except urllib.error.URLError as e:
        return HttpResult(None, None, _classify_exc(e.reason if isinstance(e.reason, BaseException) else e))
    except (socket.timeout, TimeoutError) as e:
        return HttpResult(None, None, _classify_exc(e))
    except OSError as e:
        return HttpResult(None, None, _classify_exc(e))
    try:
        body = json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        return HttpResult(status, None, "invalid_json")
    return HttpResult(status, body, None)


def tcp_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def interpret_health(res: HttpResult) -> tuple:
    """-> (state, detail, data). 'ok' and 'degraded' mean the API process is alive and answering."""
    if res.error in ("refused", "timeout", "unreachable"):
        return res.error, "no answer from /health (%s)" % res.error, None
    if res.error == "invalid_json":
        return "invalid", "HTTP %s with a body that is not JSON" % res.status, None
    b = res.body
    shaped = isinstance(b, dict) and isinstance(b.get("models"), dict) and isinstance(b.get("runtime"), dict)
    if res.status == 200 and shaped and b.get("status") == "ok":
        return "ok", "ok", b
    if res.status == 503 and shaped and b.get("status") == "degraded":
        missing = [m.get("id") for m in b.get("missing", []) if isinstance(m, dict)]
        return "degraded", "model file(s) missing: %s (a restart cannot fix this)" % (", ".join(map(str, missing)) or "?"), b
    if res.status not in (200, 503):
        return "http_error", "unexpected HTTP %s from /health" % res.status, None
    return "invalid", "response does not look like a MAPLE /health document", None


def interpret_models(res: HttpResult) -> tuple:
    """-> (state, detail, info). Alive states: ok, incomplete (API answers but the roster differs; a restart cannot fix it)."""
    if res.error in ("refused", "timeout", "unreachable"):
        return res.error, "no answer from /v1/models (%s)" % res.error, {}
    if res.error == "invalid_json":
        return "invalid", "/v1/models body is not JSON", {}
    b = res.body
    if res.status != 200 or not isinstance(b, dict) or not isinstance(b.get("data"), list):
        return ("http_error" if res.status != 200 else "invalid"), "unexpected /v1/models response (HTTP %s)" % res.status, {}
    ids, names = [], {}
    for m in b["data"]:
        if isinstance(m, dict) and isinstance(m.get("id"), str):
            ids.append(m["id"])
            if isinstance(m.get("name"), str):
                names[m["id"]] = m["name"]
    aliases = b.get("aliases") if isinstance(b.get("aliases"), dict) else {}
    missing = [m for m in EXPECTED_MODELS + (AUTO_ID,) if m not in ids]
    bad_alias = [a for a, t in ALIASES.items() if aliases.get(a) != t]
    info = {"ids": ids, "names": names, "aliases": aliases, "missing": missing, "bad_aliases": bad_alias}
    if missing or bad_alias:
        return "incomplete", "roster differs: missing=%s alias_problems=%s" % (missing, bad_alias), info
    return "ok", "ok", info


class ActivityTracker:
    """Remembers recent model activity so a slow /health during a model load or switch is not counted as an API crash."""

    def __init__(self, busy_grace: float):
        self.busy_grace = busy_grace
        self.last_activity: Optional[float] = None
        self.switches: Optional[int] = None
        self.loaded: Optional[str] = None
        self.seen = False

    def observe(self, runtime: dict, now: float) -> None:
        switches, loaded = runtime.get("switches"), runtime.get("loaded")
        try:
            inflight = int(runtime.get("inflight") or 0)
        except (TypeError, ValueError):
            inflight = 0
        if inflight > 0 or (self.seen and (switches != self.switches or loaded != self.loaded)):
            self.last_activity = now
        self.switches, self.loaded, self.seen = switches, loaded, True

    def is_busy(self, now: float) -> bool:
        return self.last_activity is not None and (now - self.last_activity) < self.busy_grace


# --------------------------------------------------------------------------- processes (Linux /proc only; never writes)
class ProcInfo(NamedTuple):
    pid: int
    ppid: int
    comm: str
    cmdline: tuple


def list_processes(proc_root: str = "/proc") -> Optional[list]:
    root = Path(proc_root)
    if not root.is_dir():
        return None
    out = []
    for d in root.iterdir():
        if not d.name.isdigit():
            continue
        try:
            comm = (d / "comm").read_text().strip()
            cmd = tuple(x for x in (d / "cmdline").read_bytes().decode("utf-8", "replace").split("\0") if x)
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        out.append(ProcInfo(int(d.name), ppid, comm, cmd))
    return out


def _arg_value(args: tuple, flag: str) -> Optional[str]:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def is_maple_api(p: ProcInfo, api_port: int) -> bool:
    """True for ``python3 ... -m maple serve [--port N]``: exactly what scripts/start_maple.sh execs."""
    c = p.cmdline
    for i in range(len(c) - 2):
        if c[i] == "-m" and c[i + 1] == "maple" and c[i + 2] == "serve":
            raw = _arg_value(c[i + 3:], "--port")
            try:
                port = int(raw) if raw is not None else 8000       # MAPLE's own default api_port
            except ValueError:
                return False
            return port == api_port
    return False


def is_ssh_cloudflared(p: ProcInfo) -> bool:
    """The cloudflared that carries SSH access (``tunnel --url ssh://...``). Protected: reported, never managed."""
    return "cloudflared" in p.comm and any(a.startswith("ssh://") for a in p.cmdline)


def default_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":        # os.kill() would TERMINATE the process on Windows; use a read-only query instead
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            k32.CloseHandle(h)
            return bool(ok) and code.value == 259
        except Exception:   # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:
        if Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False                      # zombie: already dead, just not reaped
    except (OSError, IndexError):
        pass
    return True


def default_send_signal(pid: int, sig: int) -> None:
    if os.name == "nt":
        raise OSError("process signalling is not supported on Windows; run the supervisor on the Linux host")
    os.kill(pid, sig)


def cmdline_of(pid: int) -> Optional[str]:
    try:
        return Path("/proc/%d/cmdline" % pid).read_bytes().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return None


def is_supervisor_pid(pid: int) -> bool:
    cmd = cmdline_of(pid)
    return True if cmd is None else ("maple_heartbeat" in cmd)     # cannot verify without /proc: assume yes


# --------------------------------------------------------------------------- GPU / RAM / disk (read-only)
def query_gpus(lib_dirs=("/usr/local/nvidia/lib64",), run=subprocess.run, which=shutil.which, exists=os.path.exists) -> dict:
    cands = [os.environ.get("NVIDIA_SMI"), which("nvidia-smi"), "/opt/bin/nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"]
    smi = next((c for c in dict.fromkeys(x for x in cands if x) if exists(c)), None)
    if not smi:
        return {"available": False, "error": "nvidia-smi not found", "devices": []}
    env = dict(os.environ)
    dirs = [d for d in lib_dirs if os.path.isdir(d)]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    try:
        r = run([smi, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"available": False, "error": "nvidia-smi failed: %s" % type(e).__name__, "devices": []}
    if r.returncode != 0:
        first = (r.stderr or r.stdout or "").strip().splitlines()[:1]
        return {"available": False, "error": "nvidia-smi exit %s: %s" % (r.returncode, (first or ["?"])[0][:160]), "devices": []}
    devs = []
    for line in r.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        try:
            devs.append({"index": int(p[0]), "name": p[1], "mem_used_mib": int(p[2]), "mem_total_mib": int(p[3]),
                         "util_pct": int(p[4]) if p[4].isdigit() else None})
        except ValueError:
            continue
    if not devs:
        return {"available": False, "error": "nvidia-smi returned no parsable GPU rows", "devices": []}
    return {"available": True, "devices": devs}


def read_ram(text: Optional[str] = None) -> Optional[dict]:
    """MemAvailable is the figure that matters (page cache counts as used in `free` but is reclaimable)."""
    try:
        text = text if text is not None else Path("/proc/meminfo").read_text()
        kv = {}
        for line in text.splitlines():
            k, _, v = line.partition(":")
            kv[k.strip()] = int(v.split()[0]) * 1024
        total, avail = kv["MemTotal"], kv["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    return {"total_mib": total // 2**20, "available_mib": avail // 2**20, "available_pct": round(100.0 * avail / total, 1) if total else None}


# --------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": utc_now_iso(), "level": record.levelname.lower(), "event": getattr(record, "event", record.getMessage())}
        d.update(getattr(record, "fields", {}))
        return json.dumps(d, default=str)


class DuplicateSupervisor(RuntimeError):
    pass


class StartResult(NamedTuple):
    ok: bool
    pid: Optional[int]
    detail: str


@dataclasses.dataclass
class Probe:
    procs_supported: bool = False
    api_proc: Optional[ProcInfo] = None
    llama_procs: list = dataclasses.field(default_factory=list)
    ssh_tunnels: list = dataclasses.field(default_factory=list)
    health_state: str = "refused"
    health_detail: str = ""
    health: Optional[dict] = None
    models_state: Optional[str] = None
    models_detail: str = ""
    models: dict = dataclasses.field(default_factory=dict)
    llama_listening: bool = False

    @property
    def alive(self) -> bool:
        return self.health_state in ("ok", "degraded") and self.models_state in (None, "ok", "incomplete")


# --------------------------------------------------------------------------- the supervisor
class Supervisor:
    def __init__(self, cfg: Config, *, http_get=http_get_json, popen=subprocess.Popen, list_procs=list_processes,
                 send_signal=default_send_signal, pid_alive=default_pid_alive, clock=time.monotonic, sleep=time.sleep,
                 gpu_query=query_gpus, ram_read=read_ram, disk_usage=shutil.disk_usage, tcp_probe=tcp_listening,
                 which=shutil.which, out=None, console: bool = True):
        self.cfg = cfg
        self.http_get, self.popen, self.list_procs = http_get, popen, list_procs
        self.send_signal, self.pid_alive, self.clock, self.sleep = send_signal, pid_alive, clock, sleep
        self.gpu_query, self.ram_read, self.disk_usage, self.tcp_probe, self.which = gpu_query, ram_read, disk_usage, tcp_probe, which
        self.out = out or (lambda s: print(s, flush=True))
        self.console = console
        self.t0 = clock()
        self.tracker = ActivityTracker(cfg.busy_grace)
        self.failures = 0
        self.attempts = 0                 # consecutive restart attempts (drives the backoff)
        self.restart_times: list = []
        self.next_restart_at = 0.0
        self.healthy_since: Optional[float] = None
        self.api_state = "UNKNOWN"
        self.blocked_reason: Optional[str] = None
        self.gave_up = False
        self.watch_only_notified = False
        self.stopping = False
        self.child = None                 # Popen of an API that THIS supervisor started
        self.names: dict = {}
        self.last_probe: Optional[Probe] = None
        self.lock_path = cfg.state_dir / "supervisor.lock"
        self._lock_held = False
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("maple_heartbeat.%x" % id(self))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        h = logging.handlers.RotatingFileHandler(cfg.state_dir / "heartbeat.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        h.setFormatter(JsonFormatter())
        self.logger.addHandler(h)

    # ---- logging helpers
    def event(self, name: str, level: str = "info", **fields) -> None:
        self.logger.log(getattr(logging, level.upper(), logging.INFO), name, extra={"event": name, "fields": fields})

    def say(self, line: str) -> None:
        if self.console:
            self.out(line)

    def close(self) -> None:
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)

    # ---- probing (no side effects on the server)
    def _find_api(self, procs: Optional[list]) -> Optional[ProcInfo]:
        me = os.getpid()
        for p in procs or []:
            if p.pid != me and is_maple_api(p, self.cfg.api_port):
                return p
        return None

    def probe_once(self) -> Probe:
        cfg = self.cfg
        procs = self.list_procs()
        pr = Probe(procs_supported=procs is not None)
        pr.api_proc = self._find_api(procs)
        pr.llama_procs = [p for p in (procs or []) if p.comm == "llama-server"]
        pr.ssh_tunnels = [p for p in (procs or []) if is_ssh_cloudflared(p)]
        res = self.http_get(cfg.api_url + "/health", cfg.request_timeout)
        pr.health_state, pr.health_detail, pr.health = interpret_health(res)
        if pr.health_state in ("ok", "degraded"):
            mres = self.http_get(cfg.api_url + "/v1/models", cfg.request_timeout)
            pr.models_state, pr.models_detail, pr.models = interpret_models(mres)
            if pr.models.get("names"):
                self.names = pr.models["names"]
        pr.llama_listening = self.tcp_probe(cfg.api_host, cfg.llama_port)
        self.last_probe = pr
        return pr

    # ---- one monitoring cycle
    def tick(self) -> str:
        cfg, now = self.cfg, self.clock()
        if self.child is not None and self.child.poll() is not None:
            self.child = None                                   # reap our own finished child
        pr = self.probe_once()
        if pr.health_state in ("ok", "degraded") and pr.health:
            self.tracker.observe(pr.health["runtime"], now)
        if pr.alive:
            self.failures = 0
            self.healthy_since = self.healthy_since if self.healthy_since is not None else now
            if now - self.healthy_since >= cfg.stable_after:
                self.attempts = 0
            if self.gave_up or self.api_state in ("GAVE_UP", "BLOCKED"):
                self.event("api_recovered_without_help", level="info")
            self.gave_up, self.blocked_reason, self.watch_only_notified = False, None, False
            self.api_state = "ONLINE" if (pr.health_state == "ok" and pr.models_state == "ok") else "DEGRADED"
            if pr.health_state == "degraded":
                self.event("api_degraded", level="warning", detail=pr.health_detail)
            if pr.models_state == "incomplete":
                self.event("model_roster_differs", level="warning", detail=pr.models_detail)
            rt = (pr.health or {}).get("runtime", {})
            if rt.get("loaded") and rt.get("healthy") is False and not rt.get("inflight"):
                self.event("llama_server_not_responding", level="warning", loaded=rt.get("loaded"),
                           note="MAPLE reloads a model whose llama-server is unhealthy on the next request; no action taken here")
            return "healthy"
        self.healthy_since = None
        process_missing = pr.procs_supported and pr.api_proc is None
        failure_kind = pr.health_state if pr.health_state not in ("ok", "degraded") else (pr.models_state or "invalid")
        if failure_kind == "timeout" and not process_missing and self.tracker.is_busy(now):
            self.api_state = "BUSY"
            self.event("api_busy", level="info", note="slow /health shortly after model activity; not counted as a failure",
                       grace_s=cfg.busy_grace)
            return "busy"
        self.failures += 1
        self.api_state = "UNHEALTHY"
        self.event("api_failure", level="warning", kind=failure_kind, detail=pr.health_detail or pr.models_detail,
                   consecutive=self.failures, limit=cfg.max_api_failures, process_found=pr.api_proc is not None)
        self.say("[%s] WARNING MAPLE API %s (failure %d/%d)" % (local_stamp(), failure_kind, self.failures, cfg.max_api_failures))
        if process_missing and failure_kind in ("refused", "unreachable"):
            return "recover:" + self.recover("process_exited")
        if self.failures >= cfg.max_api_failures:
            return "recover:" + self.recover("consecutive_api_failures")
        return "failing"

    # ---- inspection and recovery
    def inspect_failure(self, reason: str) -> dict:
        cfg = self.cfg
        procs = self.list_procs()
        api = self._find_api(procs)
        llama = [p for p in (procs or []) if p.comm == "llama-server"]
        orphans = [p.pid for p in llama if api is None or p.ppid != api.pid]
        port_busy = self.tcp_probe(cfg.api_host, cfg.api_port)
        tails = {"maple_api_log": tail_text(cfg.state_dir / "maple_api.log"),
                 "maple_log": tail_text(cfg.maple_home / "logs" / "maple.log") if cfg.maple_home else ""}
        diagnosis, block = [], None
        if api is None and port_busy:
            block = "port %d is held by a process that is not MAPLE; not starting a second server" % cfg.api_port
        elif "Address already in use" in tails["maple_api_log"] and api is None:
            block = "the last MAPLE start failed with 'Address already in use'; free port %d first" % cfg.api_port
        if "Traceback" in tails["maple_api_log"]:
            diagnosis.append("Python traceback in maple_api.log (tail recorded)")
        if orphans:
            diagnosis.append("llama-server pid(s) %s are not children of the MAPLE API; they may hold port %d/GPU memory and are NOT touched "
                             "by this tool" % (orphans, cfg.llama_port))
        info = {"reason": reason, "api_pid": api.pid if api else None, "api_cmdline": " ".join(api.cmdline) if api else None,
                "llama_server_pids": [p.pid for p in llama], "orphan_llama_server_pids": orphans, "api_port_in_use": port_busy,
                "diagnosis": diagnosis, "block": block, "log_tails": tails}
        self.event("recovery_inspection", level="warning", **info)
        return {"process": api, "block": block, **info}

    def recover(self, reason: str) -> str:
        cfg, now = self.cfg, self.clock()
        if cfg.no_start:                     # watch-only (--no-start): never stop, start or restart MAPLE; just report
            self.api_state = "UNHEALTHY"
            if not self.watch_only_notified:
                self.watch_only_notified = True
                self.event("recovery_disabled_watch_only", level="warning", reason=reason, consecutive=self.failures,
                           note="--no-start is set: no automatic stop, start or restart is performed")
                self.say("[%s] WATCH-ONLY (--no-start): MAPLE API unhealthy (%s); no automatic recovery will be attempted" % (local_stamp(), reason))
            return "disabled"
        self.restart_times = [t for t in self.restart_times if now - t < cfg.restart_window]
        if len(self.restart_times) >= cfg.max_restarts:
            if not self.gave_up:
                self.gave_up = True
                self.event("restart_limit_reached", level="critical", restarts=len(self.restart_times), window_s=cfg.restart_window,
                           note="no further automatic restarts; operator action required")
                self.say("[%s] CRITICAL restart limit reached (%d in %ds); automatic recovery stopped" % (local_stamp(), cfg.max_restarts, cfg.restart_window))
            self.api_state = "GAVE_UP"
            return "limit"
        if now < self.next_restart_at:
            self.event("recovery_backoff", wait_s=round(self.next_restart_at - now), reason=reason)
            self.api_state = "BACKOFF"
            return "backoff"
        diag = self.inspect_failure(reason)
        if diag["block"]:
            self.api_state, self.blocked_reason = "BLOCKED", diag["block"]
            self.next_restart_at = now + cfg.backoff_base
            self.event("recovery_blocked", level="critical", why=diag["block"])
            self.say("[%s] CRITICAL recovery blocked: %s" % (local_stamp(), diag["block"]))
            return "blocked"
        self.attempts += 1
        self.next_restart_at = now + min(cfg.backoff_base * 2 ** (self.attempts - 1), cfg.backoff_max)
        self.restart_times.append(now)
        self.api_state = "RECOVERING"
        self.event("recovery_attempt", level="warning", attempt=self.attempts, reason=reason, restarts_in_window=len(self.restart_times))
        self.say("[%s] Recovering MAPLE API (reason: %s, attempt %d)" % (local_stamp(), reason, self.attempts))
        if diag["process"] is not None and not self.stop_maple_api(diag["process"]):
            self.api_state, self.blocked_reason = "BLOCKED", "MAPLE API did not exit on SIGTERM; SIGKILL is deliberately not used (it would orphan llama-server)"
            self.event("recovery_blocked", level="critical", why=self.blocked_reason)
            return "blocked"
        res = self.start_maple()
        if res.ok:
            self.failures, self.blocked_reason, self.api_state = 0, None, "ONLINE"
            self.event("recovery_succeeded", pid=res.pid, detail=res.detail)
            self.say("[%s] MAPLE API recovered (pid %s)" % (local_stamp(), res.pid))
            return "recovered"
        self.event("recovery_failed", level="error", detail=res.detail, pid=res.pid)
        self.say("[%s] ERROR recovery failed: %s" % (local_stamp(), res.detail))
        return "failed"

    def stop_maple_api(self, proc: ProcInfo) -> bool:
        """SIGTERM the verified MAPLE API; its shutdown handler stops llama-server. Never SIGKILL, never touch llama-server."""
        cur = next((p for p in (self.list_procs() or []) if p.pid == proc.pid), None)
        if cur is None:
            return True
        if not is_maple_api(cur, self.cfg.api_port):
            self.event("refusing_to_signal", level="error", pid=proc.pid, why="process no longer looks like the MAPLE API")
            return False
        try:
            self.send_signal(proc.pid, signal.SIGTERM)
        except OSError as e:
            self.event("signal_failed", level="error", pid=proc.pid, error=str(e))
            return False
        self.event("sigterm_sent", pid=proc.pid)
        deadline = self.clock() + self.cfg.stop_timeout
        while self.clock() < deadline:
            if self.child is not None and self.child.pid == proc.pid:
                self.child.poll()
            if not self.pid_alive(proc.pid):
                self.event("maple_api_stopped", pid=proc.pid)
                self.child = None if (self.child is not None and self.child.pid == proc.pid) else self.child
                return True
            self.sleep(1)
        self.event("maple_api_did_not_exit", level="critical", pid=proc.pid, waited_s=self.cfg.stop_timeout)
        return False

    def start_maple(self) -> StartResult:
        cfg = self.cfg
        if cfg.no_start:                     # hard guard in front of the only Popen of MAPLE
            return StartResult(False, None, "--no-start is set: automatic start/restart of MAPLE is disabled")
        script = cfg.start_script
        if script is None or not script.is_file():
            return StartResult(False, None, "scripts/start_maple.sh not found (set --maple-home or MAPLE_HOME)")
        bash = self.which("bash")
        if not bash:
            return StartResult(False, None, "bash not found in PATH")
        existing = self._find_api(self.list_procs())
        if existing:
            return StartResult(False, existing.pid, "a MAPLE API process already exists (pid %d); not starting a second one" % existing.pid)
        if self.tcp_probe(cfg.api_host, cfg.api_port):
            return StartResult(False, None, "port %d is already in use by a process that is not MAPLE" % cfg.api_port)
        env = {k: v for k, v in os.environ.items() if k != "API_KEY"}      # the tunnel's key never reaches MAPLE
        env["MAPLE_HOME"] = str(cfg.maple_home)
        cmd = [bash, str(script), "--host", cfg.api_host, "--port", str(cfg.api_port)]
        logf = open(cfg.state_dir / "maple_api.log", "ab")
        self.event("starting_maple", cmd=cmd, cwd=str(cfg.maple_home))
        try:
            proc = self.popen(cmd, cwd=str(cfg.maple_home), stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                              env=env, start_new_session=True)
        except OSError as e:
            return StartResult(False, None, "could not start %s: %s" % (script.name, e))
        finally:
            logf.close()
        self.child = proc
        atomic_write_json(cfg.state_dir / "maple_api.pid", {"pid": proc.pid, "started_utc": utc_now_iso(), "cmd": cmd})
        t_start, deadline = self.clock(), self.clock() + cfg.start_timeout
        while self.clock() < deadline:
            if proc.poll() is not None:
                return StartResult(False, proc.pid, "MAPLE exited during startup (code %s); log tail: %s" %
                                   (proc.poll(), tail_text(cfg.state_dir / "maple_api.log", 8, 800).replace("\n", " | ")))
            state = interpret_health(self.http_get(cfg.api_url + "/health", cfg.request_timeout))[0]
            if state in ("ok", "degraded"):
                return StartResult(True, proc.pid, "healthy after %.0fs" % (self.clock() - t_start))
            self.sleep(2)
        return StartResult(False, proc.pid, "startup timeout after %ds (process left running; see %s)" % (cfg.start_timeout, cfg.state_dir / "maple_api.log"))

    # ---- status
    def read_tunnel_state(self) -> dict:
        st = read_json(self.cfg.state_dir / TUNNEL_STATE_FILE)
        if not st:
            return {"state": "not running", "url": None, "mode": "ssh", "note": "no tunnel manager state; use an SSH tunnel"}
        age = time.time() - float(st.get("updated_epoch", 0) or 0)
        if st.get("state") not in ("disabled", "stopped") and age > max(3 * self.cfg.check_interval, 180):
            return {**{k: st.get(k) for k in ("mode", "url")}, "state": "unknown", "note": "tunnel state file is stale (%.0fs old)" % age}
        return st

    def build_status(self, pr: Optional[Probe] = None) -> dict:
        cfg, pr = self.cfg, (pr or self.last_probe or Probe())
        now = self.clock()
        gpus = self.gpu_query(cfg.nvidia_lib_dirs)
        ram = self.ram_read()
        try:
            du = self.disk_usage(str(cfg.disk_path or cfg.state_dir))
            disk = {"path": str(cfg.disk_path or cfg.state_dir), "free_gib": round(du.free / 2**30, 2), "total_gib": round(du.total / 2**30, 1)}
        except OSError:
            disk = {"error": "disk usage unavailable"}
        warnings = []
        if disk.get("free_gib") is not None and disk["free_gib"] < cfg.min_disk_gib:
            warnings.append("low disk space: %.2f GiB free" % disk["free_gib"])
        if ram and ram.get("available_pct") is not None and ram["available_pct"] < 10:
            warnings.append("low available RAM: %s MiB" % ram["available_mib"])
        if gpus.get("available") and pr.procs_supported and not pr.llama_procs and any(d["mem_used_mib"] > 600 for d in gpus["devices"]):
            warnings.append("GPU memory is in use but no llama-server process exists (possible orphan; not touched by this tool)")
        if pr.models_state == "incomplete":
            warnings.append(pr.models_detail)
        rt = (pr.health or {}).get("runtime", {})
        return {
            "timestamp": utc_now_iso(), "uptime_minutes": int((now - self.t0) / 60), "version": __version__,
            "supervisor": {"pid": os.getpid(), "check_interval_s": cfg.check_interval, "heartbeat_minutes": cfg.heartbeat_minutes},
            "maple_api": {"state": self.api_state, "url": cfg.api_url, "pid": pr.api_proc.pid if pr.api_proc else None,
                          "health": pr.health_state, "health_detail": pr.health_detail, "models_endpoint": pr.models_state,
                          "consecutive_failures": self.failures, "failure_limit": cfg.max_api_failures,
                          "restarts_in_window": len([t for t in self.restart_times if now - t < cfg.restart_window]),
                          "restart_limit": cfg.max_restarts, "blocked_reason": self.blocked_reason,
                          "loaded_model": rt.get("loaded"), "inflight": rt.get("inflight"), "switches": rt.get("switches"),
                          "model_files": (pr.health or {}).get("models"), "roster": {"models": list(EXPECTED_MODELS), "aliases": ALIASES, "auto": True}},
            "llama_server": {"managed_by": "maple", "internal_port": cfg.llama_port, "port_listening": pr.llama_listening,
                             "pids": [p.pid for p in pr.llama_procs], "healthy_per_maple": rt.get("healthy")},
            "ssh_tunnel": {"protected_not_managed": True, "cloudflared_pids": [p.pid for p in pr.ssh_tunnels]},
            "tunnel": self.read_tunnel_state(), "gpu": gpus, "ram": ram, "disk": disk, "warnings": warnings,
        }

    def format_report(self, st: dict) -> str:
        api, t = st["maple_api"], st["tunnel"]
        loaded = api.get("loaded_model")
        model = ("%s (%s)" % (loaded, self.names[loaded]) if loaded in self.names else str(loaded)) if loaded else "none loaded (models load on the first request)"
        label = {"ONLINE": "ONLINE", "DEGRADED": "DEGRADED", "BUSY": "ONLINE (busy: model activity)"}.get(api["state"], api["state"])
        tstate = str(t.get("state", "unknown")).upper()
        if t.get("mode") == "ssh" or t.get("state") in ("disabled", "not running"):
            tstate = "DISABLED (SSH tunnelling; no public tunnel)"
        endpoint = t.get("url") or "none (SSH: ssh -N -L %d:127.0.0.1:%d <ssh-alias>)" % (self.cfg.api_port, self.cfg.api_port)
        if t.get("url") and t.get("auth") == "bearer-proxy-verified":
            endpoint += "  (Bearer key required)"
        lines = ["", "=" * 65, "[%s] Uptime: %d minutes" % (local_stamp(), st["uptime_minutes"]),
                 "Model       : %s" % model, "Llama API   : %s" % label, "Tunnel      : %s" % tstate, "Endpoint    : %s" % endpoint,
                 "MAPLE API   : %s pid=%s | failures %d/%d | restarts %d/%d per %ds" % (
                     api["url"], api["pid"], api["consecutive_failures"], api["failure_limit"], api["restarts_in_window"], api["restart_limit"], self.cfg.restart_window),
                 "llama-server: managed by MAPLE | internal port %d %s | pids %s" % (
                     self.cfg.llama_port, "LISTENING" if st["llama_server"]["port_listening"] else "closed (no model loaded)", st["llama_server"]["pids"] or "none")]
        g = st["gpu"]
        if g.get("available"):
            for d in g["devices"]:
                lines.append("GPU %-8d: %s %d/%d MiB, util %s%%" % (d["index"], d["name"], d["mem_used_mib"], d["mem_total_mib"], d["util_pct"]))
        else:
            lines.append("GPU         : unavailable (%s)" % g.get("error"))
        r = st["ram"]
        lines.append("RAM         : %s" % ("%d of %d MiB available (%s%%)" % (r["available_mib"], r["total_mib"], r["available_pct"]) if r else "unavailable"))
        d = st["disk"]
        lines.append("Disk        : %s" % ("%s GiB free of %s GiB" % (d["free_gib"], d["total_gib"]) if "free_gib" in d else d.get("error")))
        if st["ssh_tunnel"]["cloudflared_pids"]:
            lines.append("SSH tunnel  : cloudflared pid %s (protected, not managed by this tool)" % st["ssh_tunnel"]["cloudflared_pids"])
        if api.get("blocked_reason"):
            lines.append("BLOCKED     : %s" % api["blocked_reason"])
        lines += ["WARNING     : %s" % w for w in st["warnings"]]
        lines.append("=" * 65)
        return "\n".join(lines)

    def write_report(self, st: dict) -> None:
        atomic_write_json(self.cfg.state_dir / "status.json", st)
        with open(self.cfg.state_dir / "reports.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(st, default=str) + "\n")
        self.event("status_report", state=st["maple_api"]["state"], loaded=st["maple_api"]["loaded_model"],
                   failures=st["maple_api"]["consecutive_failures"], gpu_available=st["gpu"].get("available"))

    # ---- single-instance lock
    def acquire_lock(self) -> None:
        for _ in range(2):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                held = read_json(self.lock_path) or {}
                pid = int(held.get("pid") or 0)
                if pid and pid != os.getpid() and self.pid_alive(pid) and is_supervisor_pid(pid):
                    raise DuplicateSupervisor("another maple_heartbeat is already running (pid %d)" % pid)
                self.event("stale_lock_removed", level="warning", stale_pid=pid)
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "started_utc": utc_now_iso()}, f)
            self._lock_held = True
            return
        raise DuplicateSupervisor("could not take the supervisor lock %s" % self.lock_path)

    def release_lock(self) -> None:
        if self._lock_held and (read_json(self.lock_path) or {}).get("pid") == os.getpid():
            try:
                self.lock_path.unlink()
            except OSError:
                pass
        self._lock_held = False

    # ---- main loop
    def install_signal_handlers(self) -> None:
        def _stop(signum, frame):   # noqa: ARG001
            self.stopping = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _stop)
            except ValueError:      # not in the main thread
                pass

    def run(self) -> int:
        self.acquire_lock()
        cfg = self.cfg
        try:
            self.install_signal_handlers()
            self.event("supervisor_started", pid=os.getpid(), cfg={k: str(v) for k, v in dataclasses.asdict(cfg).items()})
            self.say("MAPLE heartbeat started | check every %ss | report every %s min | state dir %s" % (cfg.check_interval, cfg.heartbeat_minutes, cfg.state_dir))
            first = self.probe_once()
            if first.alive:
                self.api_state = "ONLINE" if first.models_state == "ok" and first.health_state == "ok" else "DEGRADED"
            elif first.procs_supported and first.api_proc is None and not cfg.no_start:
                self.say("MAPLE API is not running; starting it with scripts/start_maple.sh (timeout %ss)" % cfg.start_timeout)
                res = self.start_maple()
                self.event("initial_start", level="info" if res.ok else "error", ok=res.ok, detail=res.detail)
                self.say("  %s: %s" % ("started" if res.ok else "NOT started", res.detail))
            self.write_report(self.build_status())
            self.say(self.format_report(self.build_status()))
            next_check = self.clock() + cfg.check_interval
            next_report = self.clock() + cfg.heartbeat_minutes * 60
            while not self.stopping:
                self.sleep(1)
                if self.stopping:
                    break
                if self.clock() >= next_check:
                    self.tick()
                    next_check = self.clock() + cfg.check_interval
                if self.clock() >= next_report:
                    st = self.build_status()
                    self.write_report(st)
                    self.say(self.format_report(st))
                    next_report = self.clock() + cfg.heartbeat_minutes * 60
            return 0
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.event("supervisor_stopping", stop_maple=self.cfg.stop_maple_on_exit)
        self.say("[%s] Heartbeat stopping." % local_stamp())
        if self.cfg.stop_maple_on_exit and self.cfg.no_start:      # watch-only never stops MAPLE, even on exit
            self.say("MAPLE API left running (--no-start is watch-only; --stop-maple-on-exit ignored; use `stop --all` to stop it).")
        elif self.cfg.stop_maple_on_exit:
            api = self._find_api(self.list_procs())
            ok = self.stop_maple_api(api) if api else True
            self.say("MAPLE API %s." % ("stopped" if ok else "did NOT stop (see heartbeat.log)"))
        else:
            self.say("MAPLE API left running (use `stop --all` or --stop-maple-on-exit to stop it too).")
        try:
            st = self.build_status()
            st["supervisor"]["state"] = "stopped"
            atomic_write_json(self.cfg.state_dir / "status.json", st)
        except Exception as e:   # noqa: BLE001
            self.event("final_status_failed", level="warning", error=str(e))
        self.release_lock()
        self.event("supervisor_stopped")


# --------------------------------------------------------------------------- CLI
def cmd_status(cfg: Config, as_json: bool) -> int:
    sup = Supervisor(cfg, console=False)
    try:
        pr = sup.probe_once()
        sup.api_state = "ONLINE" if (pr.health_state == "ok" and pr.models_state == "ok") else ("DEGRADED" if pr.alive else "OFFLINE")
        st = sup.build_status(pr)
        lock = read_json(sup.lock_path) or {}
        st["supervisor"]["running"] = bool(lock.get("pid") and sup.pid_alive(int(lock["pid"])))
        st["supervisor"]["lock_pid"] = lock.get("pid")
        print(json.dumps(st, indent=2, default=str) if as_json else sup.format_report(st) + "\nSupervisor  : %s" % (
            "running (pid %s)" % lock["pid"] if st["supervisor"]["running"] else "not running"))
        return 0 if pr.alive else 2
    finally:
        sup.close()


def cmd_stop(cfg: Config, stop_all: bool) -> int:
    sup = Supervisor(cfg, console=False)
    try:
        lock = read_json(sup.lock_path) or {}
        pid = int(lock.get("pid") or 0)
        if pid and sup.pid_alive(pid) and is_supervisor_pid(pid):
            sup.send_signal(pid, signal.SIGTERM)
            print("SIGTERM sent to heartbeat supervisor pid %d; waiting for it to exit ..." % pid)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and sup.pid_alive(pid):
                time.sleep(1)
            if sup.pid_alive(pid):
                print("supervisor still running after 30 s (it may be waiting for a MAPLE start to finish); not forcing it")
                return 1
            print("supervisor stopped")
        else:
            print("no running heartbeat supervisor found")
        if stop_all:
            api = sup._find_api(sup.list_procs())
            if not api:
                print("no MAPLE API process found")
                return 0
            ok = sup.stop_maple_api(api)
            print("MAPLE API pid %d %s" % (api.pid, "stopped (its handler also stops llama-server)" if ok else "did NOT exit on SIGTERM; not using SIGKILL"))
            return 0 if ok else 1
        return 0
    finally:
        sup.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="maple_heartbeat.py", description="Health supervisor for the MAPLE API (llama-server stays managed by MAPLE).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in (("run", "supervise MAPLE until stopped"), ("status", "one live read-only status check"), ("stop", "stop the supervisor")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--maple-home", help="MAPLE project folder (default: $MAPLE_HOME or auto-detect)")
        p.add_argument("--state-dir", help="logs/state folder shared with maple_tunnel.py")
        p.add_argument("--api-host"), p.add_argument("--api-port", type=int), p.add_argument("--llama-port", type=int)
        if name == "run":
            p.add_argument("--check-interval", type=float), p.add_argument("--heartbeat-minutes", type=float)
            p.add_argument("--start-timeout", type=float), p.add_argument("--max-api-failures", type=int)
            p.add_argument("--request-timeout", type=float), p.add_argument("--max-restarts", type=int)
            p.add_argument("--stop-timeout", type=float, help="seconds to wait after SIGTERM for the MAPLE API to exit (default 180; env MAPLE_SUP_STOP_TIMEOUT)")
            p.add_argument("--no-start", action="store_true", help="WATCH-ONLY: never start, stop or restart MAPLE (not at launch, not on failures); only observe and report")
            p.add_argument("--stop-maple-on-exit", action="store_true", help="also SIGTERM the MAPLE API when this supervisor exits")
            p.add_argument("--nvidia-lib-dir", action="append", help="extra LD_LIBRARY_PATH dir for nvidia-smi only (repeatable)")
        if name == "status":
            p.add_argument("--json", action="store_true")
        if name == "stop":
            p.add_argument("--stop-timeout", type=float, help="seconds to wait after SIGTERM for the MAPLE API to exit (default 180)")
            p.add_argument("--all", action="store_true", help="after the supervisor, also stop the MAPLE API (SIGTERM)")
    return ap


def main(argv: Optional[list] = None) -> int:
    ns = build_parser().parse_args(argv)
    cfg = config_from_args(ns)
    if ns.cmd == "status":
        return cmd_status(cfg, ns.json)
    if ns.cmd == "stop":
        return cmd_stop(cfg, ns.all)
    sup = Supervisor(cfg)
    try:
        return sup.run()
    except DuplicateSupervisor as e:
        print("not started: %s" % e, file=sys.stderr)
        return 3
    finally:
        sup.close()


if __name__ == "__main__":
    sys.exit(main())
