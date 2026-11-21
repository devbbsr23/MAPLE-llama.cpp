#!/usr/bin/env python3
"""maple_tunnel.py - remote connectivity manager for a MAPLE-llama.cpp API (separate from maple_heartbeat.py).

Adapted from the Cloudflare Quick Tunnel helper (``start_tunnel`` / URL detection / ``tunnel.log``).

SECURITY MODEL (read this first)
  MAPLE has NO built-in authentication. Therefore:
  * default mode ``ssh``: nothing public is started. The tool prints the SSH-tunnel command and exits. (Recommended.)
  * mode ``--public``: a Cloudflare Quick Tunnel is started, but it points at a small local authenticating reverse proxy
    (127.0.0.1:AUTH_PORT), NEVER directly at the MAPLE API. The proxy demands ``Authorization: Bearer <API_KEY>``,
    allows only the five MAPLE endpoints, and cannot reach the internal llama-server port. The public URL is printed
    only after a live check proves that an unauthenticated request through the public URL is rejected (401).
  * mode ``--auth-only``: NO Cloudflare and nothing public. The same Bearer-key proxy listens on 127.0.0.1:AUTH_PORT only and
    forwards to the MAPLE API on loopback. $API_KEY (from the process environment, or `start --key-stdin`) is REQUIRED; a key is
    never generated, printed, logged or saved. Reach it from Windows with `ssh -N -L 8088:127.0.0.1:8088 <alias>`.
  * Cloudflare Access (account-level SSO / service tokens) is NOT implemented here; it needs your Cloudflare account
    and cannot be verified from this code. See README.md for how to set it up yourself.
  Quick Tunnels are free, unauthenticated at Cloudflare's edge, and their URL changes whenever the tunnel restarts.

Process safety: this tool only ever signals the cloudflared child it started itself. An existing
``cloudflared ... ssh://...`` process (your SSH access) is detected, reported as protected, and never touched.
It never starts, stops or signals MAPLE or llama-server. Standard library only.

Usage:  python maple_tunnel.py run [--public | --auth-only] | start [--key-stdin] | status [--json] [--auth-only] | stop   (see README.md)
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hmac
import http.client
import json
import logging
import logging.handlers
import os
import re
import secrets
import shutil
import signal
import ipaddress
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

__version__ = "0.1.0"

# --------------------------------------------------------------------------- defaults (CLI flag > env MAPLE_TUN_<NAME> > default)
API_HOST = "127.0.0.1"
API_PORT = 8000
LLAMA_PORT = 8081               # never exposed; the proxy refuses to point at it
AUTH_PORT = 8088                # local authenticating proxy (this is what cloudflared exposes)
CHECK_INTERVAL = 60
HEARTBEAT_MINUTES = 5
URL_TIMEOUT = 60                # seconds to wait for the trycloudflare URL
VERIFY_TIMEOUT = 90             # seconds to wait until the public URL answers correctly
MAX_TUNNEL_FAILURES = 3
REQUEST_TIMEOUT = 15
MAX_RESTARTS = 5
RESTART_WINDOW = 3600
BACKOFF_BASE = 15
BACKOFF_MAX = 300
STABLE_AFTER = 300
UPSTREAM_TIMEOUT = 900          # long generations through the proxy
MAX_BODY = 8 * 1024 * 1024
LINGER_MAX = 1024 * 1024        # most request-body bytes discarded after an early rejection (bounded: no unauthenticated slow-loris sink)
MIN_KEY_LEN = 24
ENV_PREFIX = "MAPLE_TUN_"
STATE_FILE = "tunnel_state.json"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
ALLOWED = {("GET", "/health"), ("GET", "/healthz"), ("GET", "/v1/models"),
           ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"), ("POST", "/v1/route")}
ALLOWED_AUTH_ONLY = {("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"),
                     ("POST", "/v1/completions"), ("POST", "/v1/route")}      # exactly the five approved endpoints (no /healthz)
KEY_RE = re.compile(r"[\x21-\x7e]{%d,256}\Z" % MIN_KEY_LEN)                # printable ASCII, no whitespace
AUTH_STATE = "AUTH_PROXY"
CONSOLE_LOG = "auth_proxy_console.log"
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
       "content-length", "authorization", "host"}


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
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def tail_text(path: Optional[Path], chars: int = 1500) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-chars:]
    except (OSError, TypeError):
        return ""


def find_maple_home(explicit: Optional[str] = None, start: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("MAPLE_HOME"):
        return Path(os.environ["MAPLE_HOME"]).expanduser().resolve()
    here = Path(start or __file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "scripts" / "start_maple.sh").is_file():
            return d
    return None


# --------------------------------------------------------------------------- configuration
@dataclasses.dataclass
class TunnelConfig:
    state_dir: Path = Path("maple_supervisor")
    public: bool = False
    auth_only: bool = False
    api_host: str = API_HOST
    api_port: int = API_PORT
    llama_port: int = LLAMA_PORT
    auth_port: int = AUTH_PORT
    check_interval: float = CHECK_INTERVAL
    heartbeat_minutes: float = HEARTBEAT_MINUTES
    url_timeout: float = URL_TIMEOUT
    verify_timeout: float = VERIFY_TIMEOUT
    max_failures: int = MAX_TUNNEL_FAILURES
    request_timeout: float = REQUEST_TIMEOUT
    max_restarts: int = MAX_RESTARTS
    restart_window: float = RESTART_WINDOW
    backoff_base: float = BACKOFF_BASE
    backoff_max: float = BACKOFF_MAX
    stable_after: float = STABLE_AFTER
    wait_for_api: float = 600
    ssh_alias: str = "kaggle2"

    @property
    def api_url(self) -> str:
        return "http://%s:%d" % (self.api_host, self.api_port)

    @property
    def tunnel_log(self) -> Path:
        return self.state_dir / "tunnel.log"

    def validate(self) -> "TunnelConfig":
        if self.api_host not in LOOPBACK:
            raise SystemExit("refusing api_host %r: keep the MAPLE API on loopback" % self.api_host)
        if self.llama_port in (self.api_port, self.auth_port):
            raise SystemExit("llama_port %d must differ from api_port and auth_port; it is never exposed" % self.llama_port)
        if self.auth_port == self.api_port:
            raise SystemExit("auth_port must differ from api_port (the tunnel must not point straight at MAPLE)")
        if self.auth_only and self.public:
            raise SystemExit("--auth-only and --public are mutually exclusive (auth-only never starts Cloudflare)")
        if self.max_failures < 1 or self.check_interval <= 0:
            raise SystemExit("max_failures must be >= 1 and check_interval > 0")
        return self


def _pick(cli: Any, name: str, default: Any, cast: Callable = str) -> Any:
    if cli is not None:
        return cast(cli)
    raw = os.environ.get(ENV_PREFIX + name)
    if raw not in (None, ""):
        try:
            return cast(raw)
        except ValueError:
            raise SystemExit("invalid value for %s%s: %r" % (ENV_PREFIX, name, raw))
    return default


def config_from_args(ns: argparse.Namespace) -> TunnelConfig:
    g = lambda n: getattr(ns, n, None)   # noqa: E731
    home = find_maple_home(g("maple_home"))
    st = g("state_dir") or os.environ.get("MAPLE_SUP_STATE_DIR")       # same variable the heartbeat uses
    state_dir = Path(st).expanduser().resolve() if st else ((home / "logs" / "supervisor") if home else Path.cwd() / "maple_supervisor")
    return TunnelConfig(
        state_dir=state_dir, public=bool(g("public")), auth_only=bool(g("auth_only")),
        api_host=_pick(g("api_host"), "API_HOST", API_HOST), api_port=_pick(g("api_port"), "API_PORT", API_PORT, int),
        llama_port=_pick(g("llama_port"), "LLAMA_PORT", LLAMA_PORT, int), auth_port=_pick(g("auth_port"), "AUTH_PORT", AUTH_PORT, int),
        check_interval=_pick(g("check_interval"), "CHECK_INTERVAL", CHECK_INTERVAL, float),
        heartbeat_minutes=_pick(g("heartbeat_minutes"), "HEARTBEAT_MINUTES", HEARTBEAT_MINUTES, float),
        max_failures=_pick(g("max_failures"), "MAX_FAILURES", MAX_TUNNEL_FAILURES, int),
        max_restarts=_pick(g("max_restarts"), "MAX_RESTARTS", MAX_RESTARTS, int),
        ssh_alias=_pick(g("ssh_alias"), "SSH_ALIAS", "kaggle2"),
    ).validate()


def get_api_key(env: Optional[dict] = None) -> tuple:
    """-> (key, generated). Uses $API_KEY (as set by the notebook cell) or generates one. Never logged."""
    env = os.environ if env is None else env
    k = env.get("API_KEY")
    if k:
        if len(k) < MIN_KEY_LEN:
            raise SystemExit("API_KEY is shorter than %d characters; use secrets.token_urlsafe(32)" % MIN_KEY_LEN)
        return k, False
    return secrets.token_urlsafe(32), True


def load_auth_only_key(env: Optional[dict] = None) -> str:
    """$API_KEY for auth-only mode. Never generates a key (it would have to be shown) and never echoes the value."""
    env = os.environ if env is None else env
    k = env.get("API_KEY")
    if not k:
        raise SystemExit("API_KEY is not set. Auth-only mode never generates or prints a key: export API_KEY first, "
                         "or use `maple_tunnel.py start --key-stdin`.")
    if not KEY_RE.match(k):
        raise SystemExit("API_KEY is not acceptable: it must be %d-256 printable ASCII characters without spaces "
                         "(secrets.token_urlsafe(32) is ideal)." % MIN_KEY_LEN)
    return k


# --------------------------------------------------------------------------- HTTP helpers
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Reply(NamedTuple):
    status: Optional[int]
    body: bytes
    error: Optional[str]


def http_get(url: str, timeout: float, headers: Optional[dict] = None) -> Reply:
    req = urllib.request.Request(url, headers={"User-Agent": "maple-tunnel/" + __version__, **(headers or {})})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return Reply(r.status, r.read(1 << 20), None)
    except urllib.error.HTTPError as e:
        try:
            return Reply(e.code, e.read(1 << 16), None)
        except Exception:   # noqa: BLE001
            return Reply(e.code, b"", None)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        why = getattr(e, "reason", e)
        return Reply(None, b"", "timeout" if isinstance(why, (socket.timeout, TimeoutError)) else "unreachable")


def tcp_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_loopback_ip(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    mapped = getattr(a, "ipv4_mapped", None)
    return bool((mapped or a).is_loopback)


def listening_addresses(port: int, proc_net: str = "/proc/net") -> Optional[list]:
    """Local addresses of LISTEN sockets on ``port`` (from /proc/net/tcp and tcp6); None when /proc is unavailable."""
    found, seen = [], False
    for name, v6 in (("tcp", False), ("tcp6", True)):
        try:
            rows = Path(proc_net, name).read_text().splitlines()[1:]
        except OSError:
            continue
        seen = True
        for row in rows:
            f = row.split()
            if len(f) < 4 or f[3] != "0A":                 # 0A = LISTEN
                continue
            addr, _, p = f[1].partition(":")
            try:
                if int(p, 16) != port:
                    continue
                if v6:
                    raw = bytes.fromhex(addr)
                    ip = socket.inet_ntop(socket.AF_INET6, b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4)))
                else:
                    ip = socket.inet_ntoa(struct.pack("<I", int(addr, 16)))
            except (ValueError, struct.error):
                ip = "?"
            found.append(ip)
    return sorted(set(found)) if seen else None


def detect_url(log_text: str) -> Optional[str]:
    m = URL_RE.search(log_text or "")
    return m.group(0) if m else None


# --------------------------------------------------------------------------- the authenticating reverse proxy
def make_proxy_handler(key: str, upstream_host: str, upstream_port: int, on_event: Callable = lambda *a, **k: None, allowed=ALLOWED):
    key_b = key.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "maple-auth-proxy"
        sys_version = ""
        protocol_version = "HTTP/1.0"       # response ends with close: simple, and correct for SSE streaming
        timeout = 300                       # idle limit while READING the client request (not while a generation runs)

        def log_message(self, *a):          # never log request lines (no headers, no key)
            pass

        def _json(self, code: int, msg: str, typ: str, extra: Optional[dict] = None) -> None:
            body = json.dumps({"error": {"message": msg, "type": typ}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            self._linger()

        def _linger(self) -> None:
            """Half-close, then discard a bounded amount of whatever the client is still sending, so the status line is not lost to a reset."""
            try:
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                self.connection.settimeout(2.0)
                left, end = LINGER_MAX, time.monotonic() + 5.0
                while left > 0 and time.monotonic() < end:
                    data = self.connection.recv(min(65536, left))
                    if not data:
                        break
                    left -= len(data)
            except OSError:
                pass

        def _authorized(self) -> bool:
            h = self.headers.get("Authorization", "")
            if not h.lower().startswith("bearer "):
                return False
            return hmac.compare_digest(h[7:].strip().encode("utf-8"), key_b)

        def _handle(self, method: str) -> None:
            if not self._authorized():                                  # auth BEFORE anything else (no path probing)
                on_event("auth_rejected")
                return self._json(401, "missing or invalid API key", "invalid_api_key", {"WWW-Authenticate": "Bearer"})
            path = self.path.split("?")[0]
            if (method, path) not in allowed:
                return self._json(404, "not found: %s" % path, "not_found")
            body = None
            if method == "POST":
                try:
                    n = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    return self._json(411, "Content-Length required", "length_required")
                if n < 0 or n > MAX_BODY:
                    return self._json(413, "request body too large", "too_large")
                body = self.rfile.read(n)
            fwd = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=UPSTREAM_TIMEOUT)
            try:
                conn.request(method, self.path, body=body, headers=fwd)
                resp = conn.getresponse()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = resp.read1(8192)          # forwards SSE chunks as they arrive
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                on_event("proxied", method=method, path=path, status=resp.status)
            except (OSError, http.client.HTTPException) as e:
                on_event("upstream_error", path=path, error=type(e).__name__)
                try:
                    self._json(502, "MAPLE API unreachable", "bad_gateway")
                except OSError:
                    pass
            finally:
                conn.close()

        def do_GET(self):    # noqa: N802
            self._handle("GET")

        def do_POST(self):   # noqa: N802
            self._handle("POST")

        def do_PUT(self):    # noqa: N802
            self._handle("PUT")

        do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_PUT   # authenticated but never allowed -> 404; unauthenticated -> 401 (never the stdlib's 501)

    return Handler


class ProxyBindError(OSError):
    pass


class AuthProxy:
    def __init__(self, key: str, listen_port: int, upstream_host: str, upstream_port: int, on_event: Callable = lambda *a, **k: None,
                 allowed=ALLOWED):
        try:
            self.httpd = ThreadingHTTPServer(("127.0.0.1", listen_port), make_proxy_handler(key, upstream_host, upstream_port, on_event, allowed))
        except OSError as e:
            raise ProxyBindError("cannot bind 127.0.0.1:%d (%s)" % (listen_port, e.strerror or e)) from e
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="maple-auth-proxy", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self) -> None:
        try:
            self.httpd.shutdown()
        finally:
            self.httpd.server_close()


# --------------------------------------------------------------------------- process helpers (read-only inspection)
class ProcInfo(NamedTuple):
    pid: int
    comm: str
    cmdline: tuple


def list_processes(proc_root: str = "/proc") -> Optional[list]:
    root = Path(proc_root)
    if not root.is_dir():
        return None
    out = []
    for d in root.iterdir():
        if d.name.isdigit():
            try:
                comm = (d / "comm").read_text().strip()
                cmd = tuple(x for x in (d / "cmdline").read_bytes().decode("utf-8", "replace").split("\0") if x)
            except OSError:
                continue
            out.append(ProcInfo(int(d.name), comm, cmd))
    return out


def is_ssh_cloudflared(p: ProcInfo) -> bool:
    return "cloudflared" in p.comm and any(a.startswith("ssh://") for a in p.cmdline)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
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
            return False
    except (OSError, IndexError):
        pass
    return True


def cmdline_of(pid: int) -> Optional[str]:
    try:
        return Path("/proc/%d/cmdline" % pid).read_bytes().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": utc_now_iso(), "level": record.levelname.lower(), "event": getattr(record, "event", record.getMessage())}
        d.update(getattr(record, "fields", {}))
        return json.dumps(d, default=str)


class DuplicateTunnelManager(RuntimeError):
    pass


# --------------------------------------------------------------------------- the manager
class TunnelManager:
    def __init__(self, cfg: TunnelConfig, key: Optional[str] = None, *, http=http_get, popen=subprocess.Popen, which=shutil.which,
                 clock=time.monotonic, sleep=time.sleep, list_procs=list_processes, alive=pid_alive, cmdline=cmdline_of,
                 proxy_factory=AuthProxy, tcp_probe=tcp_listening, out=None, console: bool = True):
        self.cfg, self.key = cfg, key
        self.http, self.popen, self.which, self.clock, self.sleep = http, popen, which, clock, sleep
        self.list_procs, self.alive, self.cmdline, self.proxy_factory, self.tcp_probe = list_procs, alive, cmdline, proxy_factory, tcp_probe
        self.out = out or (lambda s: print(s, flush=True))
        self.console = console
        self.t0 = clock()
        self.proc = None
        self.proxy: Optional[AuthProxy] = None
        self.url: Optional[str] = None
        self.state = "DISABLED"
        self.detail = ""
        self.failures = 0
        self.attempts = 0
        self.restart_times: list = []
        self.next_restart_at = 0.0
        self.healthy_since: Optional[float] = None
        self.gave_up = False
        self.key_shown = False
        self.stopping = False
        self.insecure = False
        self.auth_verified = False
        self.upstream_ok: Optional[bool] = None
        self.rejected = 0
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = cfg.state_dir / "tunnel.lock"
        self._lock_held = False
        self.logger = logging.getLogger("maple_tunnel.%x" % id(self))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        h = logging.handlers.RotatingFileHandler(cfg.state_dir / "tunnel_manager.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        h.setFormatter(JsonFormatter())
        self.logger.addHandler(h)

    # ---- output / logging (the key is never passed to event() or say() except the one banner)
    def event(self, name: str, level: str = "info", **f) -> None:
        self.logger.log(getattr(logging, level.upper(), logging.INFO), name, extra={"event": name, "fields": f})

    def say(self, line: str) -> None:
        if self.console:
            self.out(line)

    def close(self) -> None:
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)

    # ---- state file (read by the heartbeat). Contains NO key.
    def write_state(self, state: Optional[str] = None) -> None:
        if state:
            self.state = state
        c = self.cfg
        mode = "auth-only" if c.auth_only else ("public" if c.public else "ssh")
        verified = bool((c.public and self.url and self.state == "RUNNING") or (c.auth_only and self.auth_verified and self.state == AUTH_STATE))
        doc = {
            "mode": mode, "state": self.state.lower(), "url": self.url, "detail": self.detail,
            "auth": "bearer-proxy-verified" if verified else "none",
            "cloudflared_pid": getattr(self.proc, "pid", None) if self.proc is not None else None,
            "auth_proxy_port": c.auth_port if (c.public or c.auth_only) else None,
            "updated_utc": utc_now_iso(), "updated_epoch": time.time(), "manager_pid": os.getpid()}
        if c.auth_only:
            # "url" is the LOOPBACK endpoint: the unmodified heartbeat prints it (plus "Bearer key required") in its 5-minute report
            doc.update(url=("http://127.0.0.1:%d/v1" % c.auth_port) if verified else None,
                       listen="127.0.0.1:%d" % c.auth_port, upstream="%s:%d" % (c.api_host, c.api_port), cloudflare="not used")
        atomic_write_json(c.state_dir / STATE_FILE, doc)

    # ---- ssh mode
    def ssh_instructions(self) -> str:
        c = self.cfg
        return ("SSH access (recommended; nothing public is exposed):\n"
                "  Windows PowerShell : ssh -N -L %d:127.0.0.1:%d %s\n"
                "  then use           : http://127.0.0.1:%d/v1  (any api_key string; MAPLE itself has no authentication)\n"
                "A public tunnel is OFF by default because MAPLE has no built-in authentication. "
                "Run with --public to expose it through the Bearer-key proxy.\n"
                "For a REAL API key over SSH (no Cloudflare): pipe the key into `maple_tunnel.py start --key-stdin` (or set $API_KEY and run `maple_tunnel.py start`), then forward port %d instead."
                % (c.api_port, c.api_port, c.ssh_alias, c.api_port, c.auth_port))

    def run_ssh_mode(self) -> int:
        self.detail = "public tunnel disabled by default (MAPLE has no authentication); use SSH tunnelling"
        self.write_state("DISABLED")
        procs = self.list_procs() or []
        protected = [p.pid for p in procs if is_ssh_cloudflared(p)]
        self.event("ssh_mode", protected_ssh_cloudflared=protected)
        self.say(self.ssh_instructions())
        if protected:
            self.say("Detected existing SSH cloudflared pid(s) %s: left untouched." % protected)
        return 0

    # ---- lock + orphan handling
    def acquire_lock(self) -> None:
        for _ in range(2):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                held = read_json(self.lock_path) or {}
                pid = int(held.get("pid") or 0)
                cmd = self.cmdline(pid) if pid else None
                if pid and pid != os.getpid() and self.alive(pid) and (cmd is None or "maple_tunnel" in cmd):
                    raise DuplicateTunnelManager("another maple_tunnel is already running (pid %d)" % pid)
                self.event("stale_lock_removed", level="warning", stale_pid=pid)
                self.cleanup_orphan()
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "started_utc": utc_now_iso()}, f)
            self._lock_held = True
            return
        raise DuplicateTunnelManager("could not take lock %s" % self.lock_path)

    def release_lock(self) -> None:
        if self._lock_held and (read_json(self.lock_path) or {}).get("pid") == os.getpid():
            try:
                self.lock_path.unlink()
            except OSError:
                pass
        self._lock_held = False

    def cleanup_orphan(self) -> None:
        """A crashed previous manager may have left ITS cloudflared behind. Stop it only if its command line proves it is ours
        (a quick tunnel to our auth proxy) - never an ssh:// tunnel or any other cloudflared."""
        if self.cfg.auth_only:                      # auth-only never touches any cloudflared, not even a leftover of ours
            self.event("orphan_cleanup_skipped_auth_only")
            return
        st = read_json(self.cfg.state_dir / STATE_FILE) or {}
        pid = int(st.get("cloudflared_pid") or 0)
        if not pid or not self.alive(pid):
            return
        cmd = self.cmdline(pid) or ""
        want = "--url http://127.0.0.1:%d" % self.cfg.auth_port
        if "cloudflared" in cmd and want in cmd and "ssh://" not in cmd:
            try:
                os.kill(pid, signal.SIGTERM)
                self.event("orphan_cloudflared_stopped", pid=pid)
            except OSError as e:
                self.event("orphan_stop_failed", level="warning", pid=pid, error=str(e))
        else:
            self.event("orphan_not_ours_left_alone", level="warning", pid=pid)

    # ---- start / verify / stop
    def local_api_ok(self) -> bool:
        r = self.http(self.cfg.api_url + "/health", self.cfg.request_timeout)
        if r.status not in (200, 503) or r.error:
            return False
        try:
            d = json.loads(r.body.decode("utf-8"))
        except ValueError:
            return False
        return isinstance(d, dict) and d.get("status") in ("ok", "degraded")

    def _bearer(self) -> dict:
        return {"Authorization": "Bearer " + (self.key or "")}

    def verify_proxy_local(self) -> tuple:
        """Before cloudflared starts: the proxy must reject no-key and wrong-key requests and accept the right one."""
        base = "http://127.0.0.1:%d" % self.proxy.port
        no_key = self.http(base + "/health", 5)
        bad = self.http(base + "/health", 5, {"Authorization": "Bearer wrong-" + secrets.token_hex(4)})
        good = self.http(base + "/health", 5, self._bearer())
        ok = no_key.status == 401 and bad.status == 401 and good.status in (200, 503)
        return ok, {"no_key": no_key.status, "wrong_key": bad.status, "right_key": good.status}

    def stop_cloudflared(self) -> None:
        p = self.proc
        self.proc = None
        if p is None or p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()                       # our own child only
            p.wait()

    def stop_proxy(self) -> None:
        if self.proxy:
            try:
                self.proxy.stop()
            except Exception as e:   # noqa: BLE001
                self.event("proxy_stop_error", level="warning", error=str(e))
            self.proxy = None

    def start_public(self) -> tuple:
        """-> (ok, detail). Brings up proxy -> cloudflared -> URL -> public verification. Prints nothing secret."""
        cfg = self.cfg
        cf = self.which("cloudflared")
        if not cf:
            return False, "cloudflared not found in PATH (install it, or use SSH tunnelling)"
        if self.proxy is None or not self.proxy.alive():
            self.stop_proxy()
            try:
                self.proxy = self.proxy_factory(self.key, cfg.auth_port, cfg.api_host, cfg.api_port,
                                                lambda n, **k: self.event("proxy_" + n, **k) if n != "proxied" else None)
                self.proxy.start()
            except OSError as e:
                return False, "could not bind the auth proxy on 127.0.0.1:%d (%s)" % (cfg.auth_port, e)
        ok, codes = self.verify_proxy_local()
        self.event("proxy_selftest", ok=ok, **codes)
        if not ok:
            self.stop_proxy()
            return False, "auth proxy self-test failed %s; not starting the tunnel" % codes
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        cfg.tunnel_log.write_text("", encoding="utf-8")                    # clear the old URL (as in the original script)
        env = {k: v for k, v in os.environ.items() if k != "API_KEY"}     # the key never reaches cloudflared
        cmd = [cf, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:%d" % cfg.auth_port]
        self.event("starting_cloudflared", cmd=cmd)
        with open(cfg.tunnel_log, "ab") as lf:
            try:
                self.proc = self.popen(cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            except OSError as e:
                return False, "could not start cloudflared: %s" % e
        self.write_state("STARTING")
        deadline, url = self.clock() + cfg.url_timeout, None
        while self.clock() < deadline:
            if self.proc.poll() is not None:
                return False, "cloudflared exited during startup (code %s): %s" % (self.proc.poll(), tail_text(cfg.tunnel_log, 400).replace("\n", " | "))
            url = detect_url(tail_text(cfg.tunnel_log, 20000))
            if url:
                break
            self.sleep(1)
        if not url:
            self.stop_cloudflared()
            return False, "tunnel URL not detected within %ds (see tunnel.log)" % cfg.url_timeout
        self.event("tunnel_url_detected", url=url)
        v = self.verify_public(url)
        if v["insecure"]:
            self.insecure = True
            self.stop_cloudflared()
            return False, "SECURITY: the public URL answered an unauthenticated request with HTTP %s; tunnel stopped" % v["no_key"]
        if not v["ok"]:
            self.stop_cloudflared()
            return False, "public URL did not verify (unauthenticated=%s, authenticated=%s)" % (v["no_key"], v["with_key"])
        self.url = url
        self.healthy_since = None
        return True, "verified: unauthenticated -> %s, authenticated -> %s" % (v["no_key"], v["with_key"])

    def verify_public(self, url: str) -> dict:
        """Poll until the edge answers: unauthenticated must be 401 and the right key must reach MAPLE (200)."""
        end = self.clock() + self.cfg.verify_timeout
        last = {"ok": False, "insecure": False, "no_key": None, "with_key": None}
        while True:
            a = self.http(url + "/v1/models", self.cfg.request_timeout)
            if a.status == 200:                                 # reachable WITHOUT a key: never acceptable
                return {"ok": False, "insecure": True, "no_key": a.status, "with_key": None}
            b = self.http(url + "/v1/models", self.cfg.request_timeout, self._bearer())
            last = {"ok": a.status == 401 and b.status == 200, "insecure": False, "no_key": a.status, "with_key": b.status}
            if last["ok"] or self.clock() >= end or self.proc is None or self.proc.poll() is not None:
                return last
            self.sleep(3)

    def banner(self) -> None:
        """The ONE place the key is printed (console only; never written to any file)."""
        if self.key_shown:
            self.say("Endpoint changed -> %s  (same API key as before)" % self.url)
            return
        self.key_shown = True
        self.say("\n".join(["", "=" * 65, "MAPLE PUBLIC ENDPOINT (Bearer key required; shown once)", "=" * 65,
                            "URL      : %s" % self.url, "API Base : %s/v1" % self.url, "API key  : %s" % self.key,
                            "Use      : OpenAI(base_url=<API Base>, api_key=<API key>)",
                            "Keep the key private: notebook output is saved with the notebook.", "=" * 65]))

    # ---- supervision
    def backoff_ok(self) -> str:
        cfg, now = self.cfg, self.clock()
        self.restart_times = [t for t in self.restart_times if now - t < cfg.restart_window]
        if len(self.restart_times) >= cfg.max_restarts:
            if not self.gave_up:
                self.gave_up = True
                self.event("restart_limit_reached", level="critical", restarts=len(self.restart_times))
                self.say("[%s] CRITICAL tunnel restart limit reached; no further automatic restarts" % local_stamp())
            return "limit"
        if now < self.next_restart_at:
            return "backoff"
        return "go"

    def restart(self, reason: str) -> str:
        gate = self.backoff_ok()
        if gate != "go":
            self.write_state("GAVE_UP" if gate == "limit" else "BACKOFF")
            return gate
        now = self.clock()
        self.attempts += 1
        self.next_restart_at = now + min(self.cfg.backoff_base * 2 ** (self.attempts - 1), self.cfg.backoff_max)
        self.restart_times.append(now)
        self.event("tunnel_restart", level="warning", reason=reason, attempt=self.attempts)
        self.say("[%s] Restarting Cloudflare tunnel (%s)" % (local_stamp(), reason))
        self.stop_cloudflared()
        self.url = None
        ok, detail = self.start_public()
        self.detail = detail
        if ok:
            self.failures, self.state = 0, "RUNNING"
            self.write_state()
            self.banner()
            return "recovered"
        self.write_state("FAILED_INSECURE" if self.insecure else "FAILED")
        self.event("tunnel_start_failed", level="error", detail=detail)
        self.say("[%s] ERROR tunnel start failed: %s" % (local_stamp(), detail))
        return "failed"

    def tick(self) -> str:
        cfg, now = self.cfg, self.clock()
        if self.insecure:
            return "insecure"
        if self.proxy is not None and not self.proxy.alive():
            self.event("proxy_died", level="error")
            self.stop_proxy()
            return "recover:" + self.restart("auth proxy stopped")
        if self.proc is None or self.proc.poll() is not None:
            self.event("cloudflared_exited", level="warning", code=getattr(self.proc, "returncode", None))
            self.url = None
            self.healthy_since = None
            return "recover:" + self.restart("cloudflared process exited")
        if not self.local_api_ok():
            self.state, self.detail = "WAITING_FOR_API", "MAPLE API is not answering; the tunnel is left alone (the heartbeat recovers MAPLE)"
            self.write_state()
            return "waiting_for_api"
        r = self.http(self.url + "/v1/models", cfg.request_timeout, self._bearer()) if self.url else Reply(None, b"", "no_url")
        if r.status == 200:
            self.failures = 0
            self.healthy_since = self.healthy_since if self.healthy_since is not None else now
            if now - self.healthy_since >= cfg.stable_after:
                self.attempts = 0
            self.gave_up = False
            self.state, self.detail = "RUNNING", "ok"
            self.write_state()
            return "healthy"
        self.failures += 1
        self.healthy_since = None
        self.event("public_check_failed", level="warning", status=r.status, error=r.error, consecutive=self.failures, limit=cfg.max_failures)
        self.say("[%s] WARNING public endpoint check failed (%d/%d)" % (local_stamp(), self.failures, cfg.max_failures))
        if self.failures >= cfg.max_failures:
            return "recover:" + self.restart("public endpoint unreachable")
        return "failing"

    def format_report(self) -> str:
        up = int((self.clock() - self.t0) / 60)
        return "\n".join(["", "=" * 65, "[%s] Tunnel manager uptime: %d minutes" % (local_stamp(), up),
                          "Tunnel      : %s" % ("RUNNING" if self.state == "RUNNING" else self.state),
                          "Endpoint    : %s" % (self.url or "none"), "Auth        : %s" % (
                              "Bearer key enforced (verified through the public URL)" if self.state == "RUNNING" else "n/a"),
                          "Exposed port: %d (auth proxy) -> MAPLE API %d; llama-server %d is never exposed" % (
                              self.cfg.auth_port, self.cfg.api_port, self.cfg.llama_port),
                          "Failures    : %d/%d | restarts %d/%d per %ds" % (self.failures, self.cfg.max_failures, len(self.restart_times),
                                                                          self.cfg.max_restarts, self.cfg.restart_window), "=" * 65])

    def install_signal_handlers(self) -> None:
        def _stop(signum, frame):   # noqa: ARG001
            self.stopping = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _stop)
            except ValueError:
                pass

    def run_public(self) -> int:
        cfg = self.cfg
        self.acquire_lock()
        try:
            self.install_signal_handlers()
            self.event("tunnel_manager_started", pid=os.getpid(), mode="public", auth_port=cfg.auth_port, api_port=cfg.api_port)
            self.say("MAPLE tunnel manager (PUBLIC mode, Bearer-key proxy on 127.0.0.1:%d -> MAPLE %d)" % (cfg.auth_port, cfg.api_port))
            protected = [p.pid for p in (self.list_procs() or []) if is_ssh_cloudflared(p)]
            if protected:
                self.say("Existing SSH cloudflared pid(s) %s detected: protected, left untouched." % protected)
            waited = 0.0
            while not self.local_api_ok() and not self.stopping:
                if waited >= cfg.wait_for_api:
                    self.detail = "MAPLE API never became reachable on %s" % cfg.api_url
                    self.write_state("FAILED")
                    self.say("ERROR: %s; not starting a tunnel" % self.detail)
                    return 5
                self.state = "WAITING_FOR_API"
                self.write_state()
                self.sleep(5)
                waited += 5
            if self.stopping:
                return 0
            ok, detail = self.start_public()
            self.detail = detail
            if not ok:
                self.write_state("FAILED_INSECURE" if self.insecure else "FAILED")
                self.say("ERROR: %s" % detail)
                if self.insecure:
                    return 4
                self.restart_times.append(self.clock())
                self.attempts = 1
                self.next_restart_at = self.clock() + cfg.backoff_base
            else:
                self.state = "RUNNING"
                self.write_state()
                self.banner()
            next_check = self.clock() + cfg.check_interval
            next_report = self.clock() + cfg.heartbeat_minutes * 60
            while not self.stopping:
                self.sleep(1)
                if self.stopping:
                    break
                if self.clock() >= next_check:
                    self.tick()
                    next_check = self.clock() + cfg.check_interval
                    if self.insecure:
                        return 4
                if self.clock() >= next_report:
                    self.say(self.format_report())
                    next_report = self.clock() + cfg.heartbeat_minutes * 60
            return 0
        finally:
            self.shutdown()

    # ---- AUTH-ONLY mode: a local Bearer-key proxy in front of MAPLE. No cloudflared, no public endpoint, no key output.
    def _on_proxy_event(self, name: str, **f) -> None:
        if name == "proxied":
            return
        if name == "auth_rejected":
            self.rejected += 1
            if self.rejected in (1, 10, 100, 1000, 10000, 100000):          # avoid log floods from scanners
                self.event("proxy_auth_rejected", total_rejected=self.rejected)
            return
        self.event("proxy_" + name, **f)

    def start_proxy_auth_only(self) -> None:
        cfg = self.cfg
        self.stop_proxy()
        self.proxy = self.proxy_factory(self.key, cfg.auth_port, cfg.api_host, cfg.api_port, self._on_proxy_event, allowed=ALLOWED_AUTH_ONLY)
        self.proxy.start()

    def verify_auth_only(self) -> tuple:
        """Self-test through the proxy's own port: no key / wrong key / wrong scheme must be 401; the right key must be accepted
        (any status except 401/403, because MAPLE itself may be down -> 502 is still 'accepted')."""
        base = "http://127.0.0.1:%d/health" % self.cfg.auth_port
        no_key = self.http(base, 5)
        bad = self.http(base, 5, {"Authorization": "Bearer wrong-" + secrets.token_hex(8)})
        scheme = self.http(base, 5, {"Authorization": "Basic " + secrets.token_hex(8)})
        good = self.http(base, 5, self._bearer())
        ok = no_key.status == 401 and bad.status == 401 and scheme.status == 401 and good.status not in (None, 401, 403)
        return ok, {"no_key": no_key.status, "wrong_key": bad.status, "wrong_scheme": scheme.status, "right_key": good.status}

    def _fail_insecure(self, why: str) -> None:
        self.insecure, self.auth_verified = True, False
        self.stop_proxy()
        self.detail = why
        self.event("auth_not_enforced", level="critical", why=why)
        self.write_state("FAILED_INSECURE")
        self.say("ERROR: %s" % why)

    def restart_proxy(self, reason: str) -> str:
        cfg, now = self.cfg, self.clock()
        self.restart_times = [t for t in self.restart_times if now - t < cfg.restart_window]
        if len(self.restart_times) >= cfg.max_restarts:
            self.gave_up, self.auth_verified = True, False
            self.detail = "restart limit reached (%d in %ds); operator action required" % (cfg.max_restarts, int(cfg.restart_window))
            self.event("restart_limit_reached", level="critical", reason=reason)
            self.write_state("FAILED")
            return "limit"
        if now < self.next_restart_at:
            self.auth_verified = False
            self.write_state("BACKOFF")
            return "backoff"
        self.attempts += 1
        self.next_restart_at = now + min(cfg.backoff_base * 2 ** (self.attempts - 1), cfg.backoff_max)
        self.restart_times.append(now)
        self.auth_verified = False
        self.event("proxy_restart_attempt", level="warning", attempt=self.attempts, reason=reason)
        try:
            self.start_proxy_auth_only()
        except OSError as e:
            self.detail = "could not restart the auth proxy: %s" % e
            self.event("proxy_restart_failed", level="error", error=str(e))
            self.write_state("FAILED_RESTARTING")
            return "failed"
        ok, codes = self.verify_auth_only()
        if not ok:
            self._fail_insecure("self-test failed after restart (%s); proxy stopped" % ", ".join("%s=%s" % kv for kv in codes.items()))
            return "insecure"
        self.auth_verified = True
        self.detail = "auth proxy restarted (%s)" % reason
        self.event("proxy_restarted", attempt=self.attempts)
        self.write_state(AUTH_STATE)
        return "recovered"

    def tick_auth_only(self) -> str:
        cfg, now = self.cfg, self.clock()
        if self.insecure:
            return "insecure"
        if self.proxy is None or not self.proxy.alive():
            self.event("proxy_died", level="error")
            return "recover:" + self.restart_proxy("auth proxy stopped")
        probe = self.http("http://127.0.0.1:%d/health" % cfg.auth_port, 5)      # unauthenticated on purpose: must be rejected
        if probe.status is None:
            self.event("proxy_not_answering", level="error", error=probe.error)
            return "recover:" + self.restart_proxy("auth proxy not answering")
        if probe.status != 401:
            self._fail_insecure("an unauthenticated request was NOT rejected (HTTP %s); proxy stopped" % probe.status)
            return "insecure"
        up = self.local_api_ok()
        if up != self.upstream_ok:
            self.upstream_ok = up
            self.event("upstream_reachable" if up else "upstream_unreachable", level="info" if up else "warning")
        self.failures = 0
        self.healthy_since = self.healthy_since if self.healthy_since is not None else now
        if now - self.healthy_since >= cfg.stable_after:
            self.attempts = 0
        self.gave_up = False
        self.auth_verified = True
        self.detail = ("Bearer key enforced; MAPLE reachable" if up else
                       "Bearer key enforced; MAPLE not answering (the proxy returns 502; the heartbeat recovers MAPLE)")
        self.write_state(AUTH_STATE)
        return "healthy"

    def banner_auth_only(self) -> str:
        c = self.cfg
        return "\n".join(["", "=" * 65, "MAPLE AUTH PROXY (auth-only: no Cloudflare, nothing public)",
                          "Listen      : 127.0.0.1:%d (loopback only)" % c.auth_port,
                          "Forwards to : %s:%d (MAPLE API); llama-server %d is never reachable through the proxy" % (c.api_host, c.api_port, c.llama_port),
                          "Auth        : Bearer key required on every request (401 otherwise); startup self-test passed",
                          "Key         : read from the environment; never printed, logged or saved",
                          "Windows     : ssh -N -L %d:127.0.0.1:%d %s   ->  base_url http://127.0.0.1:%d/v1" % (c.auth_port, c.auth_port, c.ssh_alias, c.auth_port),
                          "=" * 65])

    def format_report_auth_only(self) -> str:
        c = self.cfg
        up = int((self.clock() - self.t0) / 60)
        mp = "reachable" if self.upstream_ok else ("not answering" if self.upstream_ok is False else "unknown")
        return "\n".join(["", "=" * 65, "[%s] Auth proxy uptime: %d minutes" % (local_stamp(), up),
                          "Proxy       : %s" % self.state,
                          "Listen      : 127.0.0.1:%d (loopback only) -> MAPLE %s:%d" % (c.auth_port, c.api_host, c.api_port),
                          "Auth        : %s" % ("Bearer key enforced (unauthenticated probe rejected with 401)" if self.auth_verified else "NOT VERIFIED"),
                          "MAPLE API   : %s" % mp,
                          "Rejected    : %d request(s) without a valid key since start" % self.rejected,
                          "Restarts    : %d/%d per %ds" % (len(self.restart_times), c.max_restarts, c.restart_window), "=" * 65])

    def run_auth_only(self) -> int:
        cfg = self.cfg
        self.acquire_lock()
        try:
            self.install_signal_handlers()
            self.event("tunnel_manager_started", pid=os.getpid(), mode="auth-only", listen_port=cfg.auth_port, upstream_port=cfg.api_port)
            self.say("MAPLE auth proxy (AUTH-ONLY mode; Cloudflare is not used)")
            protected = [p.pid for p in (self.list_procs() or []) if is_ssh_cloudflared(p)]
            if protected:
                self.say("Existing SSH cloudflared pid(s) %s detected: protected, left untouched." % protected)
            self.detail = "starting"
            self.write_state("STARTING")
            try:
                self.start_proxy_auth_only()
            except OSError as e:
                self.detail = "could not bind the auth proxy on 127.0.0.1:%d (%s)" % (cfg.auth_port, e)
                self.write_state("FAILED")
                self.say("ERROR: %s" % self.detail)
                return 6
            ok, codes = self.verify_auth_only()
            self.event("auth_self_test", ok=ok, **codes)
            if not ok:
                self._fail_insecure("startup self-test failed (%s); proxy stopped" % ", ".join("%s=%s" % kv for kv in codes.items()))
                return 4
            self.auth_verified = True
            self.upstream_ok = self.local_api_ok()
            self.detail = ("Bearer key enforced; MAPLE reachable" if self.upstream_ok else
                           "Bearer key enforced; MAPLE not reachable yet (requests get 502 until the heartbeat brings it up)")
            self.write_state(AUTH_STATE)
            self.say(self.banner_auth_only())
            next_check = self.clock() + cfg.check_interval
            next_report = self.clock() + cfg.heartbeat_minutes * 60
            while not self.stopping:
                self.sleep(1)
                if self.stopping:
                    break
                if self.clock() >= next_check:
                    res = self.tick_auth_only()
                    next_check = self.clock() + cfg.check_interval
                    if self.insecure:
                        return 4
                    if res == "recover:limit":
                        return 6
                if self.clock() >= next_report:
                    self.say(self.format_report_auth_only())
                    next_report = self.clock() + cfg.heartbeat_minutes * 60
            return 0
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.event("tunnel_manager_stopping")
        self.say("[%s] %s" % (local_stamp(), "Auth proxy stopping: only the proxy is stopped (MAPLE, llama-server, the heartbeat and SSH access are untouched)."
                             if self.cfg.auth_only else "Tunnel manager stopping: stopping only the cloudflared this tool started."))
        self.auth_verified = False
        self.stop_cloudflared()
        self.stop_proxy()
        self.url = None
        self.detail = "stopped"
        try:
            self.write_state("STOPPED")
        except OSError:
            pass
        self.release_lock()
        self.event("tunnel_manager_stopped")


# --------------------------------------------------------------------------- CLI
def cmd_status(cfg: TunnelConfig, as_json: bool, *, http=http_get, addresses=listening_addresses) -> int:
    st = read_json(cfg.state_dir / STATE_FILE)
    lock = read_json(cfg.state_dir / "tunnel.lock") or {}
    running = bool(lock.get("pid") and pid_alive(int(lock["pid"])))
    procs = list_processes() or []
    out = {"manager_running": running, "manager_pid": lock.get("pid") if running else None, "state": st or {"state": "no state file", "mode": "ssh"},
           "protected_ssh_cloudflared_pids": [p.pid for p in procs if is_ssh_cloudflared(p)],
           "llama_port_listening_locally": tcp_listening(cfg.api_host, cfg.llama_port),
           "note": "llama-server's port is internal to MAPLE; it is never exposed by this tool"}
    if cfg.auth_only or (st or {}).get("mode") == "auth-only":
        addrs = addresses(cfg.auth_port)
        unauth = http("http://127.0.0.1:%d/health" % cfg.auth_port, 5)             # no key on purpose
        upstream = http(cfg.api_url + "/health", 5)
        out["auth_proxy"] = {
            "port": cfg.auth_port, "listen_addresses": addrs,
            "loopback_only": None if addrs is None else bool(addrs) and all(is_loopback_ip(a) for a in addrs),
            "unauthenticated_request_status": unauth.status, "auth_enforced": unauth.status == 401,
            "maple_api_status": upstream.status, "cloudflare": "not used"}
    if as_json:
        print(json.dumps(out, indent=2, default=str))
    else:
        s = out["state"]
        print("Tunnel manager: %s" % ("running (pid %s)" % out["manager_pid"] if running else "not running"))
        print("Mode          : %s" % s.get("mode"))
        print("Tunnel        : %s" % str(s.get("state", "?")).upper())
        print("Endpoint      : %s" % (s.get("url") or "none (use SSH tunnelling)"))
        print("Auth          : %s" % s.get("auth", "none"))
        ap = out.get("auth_proxy")
        if ap:
            lo = ap["loopback_only"]
            print("Auth proxy    : 127.0.0.1:%d | listening on %s | loopback only: %s%s" % (
                ap["port"], ap["listen_addresses"] if ap["listen_addresses"] is not None else "unknown",
                {True: "yes", False: "NO - EXPOSED, stop it", None: "unknown"}[lo], "" if lo is not False else " <<<"))
            print("Key enforced  : %s (request without a key -> HTTP %s)" % ("yes" if ap["auth_enforced"] else "NO / not answering", ap["unauthenticated_request_status"]))
            print("MAPLE API     : HTTP %s (direct, loopback)" % ap["maple_api_status"])
        print("SSH cloudflared (protected, not managed): %s" % (out["protected_ssh_cloudflared_pids"] or "none seen"))
    return 0


def cmd_stop(cfg: TunnelConfig) -> int:
    lock = read_json(cfg.state_dir / "tunnel.lock") or {}
    pid = int(lock.get("pid") or 0)
    cmd = cmdline_of(pid) if pid else None
    if not (pid and pid_alive(pid) and (cmd is None or "maple_tunnel" in cmd)):
        print("no running tunnel manager found (nothing was signalled)")
        return 0
    mode = (read_json(cfg.state_dir / STATE_FILE) or {}).get("mode")
    os.kill(pid, signal.SIGTERM)
    print("SIGTERM sent to the %s manager pid %d; it stops only %s" % (
        mode or "tunnel", pid, "its own auth proxy (MAPLE, llama-server, the heartbeat and SSH access are not touched)" if mode == "auth-only" else "its own cloudflared"))
    end = time.monotonic() + 30
    while time.monotonic() < end and pid_alive(pid):
        time.sleep(1)
    print("tunnel manager stopped" if not pid_alive(pid) else "still running after 30 s; not forcing it")
    if mode == "auth-only" and not pid_alive(pid):
        print("port %d listening: %s" % (cfg.auth_port, tcp_listening("127.0.0.1", cfg.auth_port)))
    return 0 if not pid_alive(pid) else 1


def cmd_start(cfg: TunnelConfig, key_stdin: bool = False, wait: float = 20.0, *, popen=subprocess.Popen, stdin=None,
              sleep=time.sleep, clock=time.monotonic) -> int:
    """Detached launcher for `run --auth-only`. The key goes to the child ONLY through its environment (never argv, never a file)."""
    key = load_auth_only_key({"API_KEY": (stdin or sys.stdin).readline().rstrip("\r\n")} if key_stdin else os.environ)
    lock = read_json(cfg.state_dir / "tunnel.lock") or {}
    pid = int(lock.get("pid") or 0)
    if pid and pid_alive(pid) and "maple_tunnel" in (cmdline_of(pid) or "maple_tunnel"):
        print("not started: a maple_tunnel manager is already running (pid %d)" % pid)
        return 3
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--auth-only", "--state-dir", str(cfg.state_dir),
           "--api-host", cfg.api_host, "--api-port", str(cfg.api_port), "--llama-port", str(cfg.llama_port), "--auth-port", str(cfg.auth_port)]
    env = dict(os.environ)
    env["API_KEY"] = key
    log_path = cfg.state_dir / CONSOLE_LOG
    with open(log_path, "ab") as logf:
        proc = popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    end = clock() + wait
    while clock() < end:
        rc = proc.poll()
        if rc is not None:
            print("auth proxy did NOT start (exit code %s). Last console lines:\n%s" % (rc, tail_text(log_path, 800)))
            return 5
        st = read_json(cfg.state_dir / STATE_FILE) or {}
        if st.get("mode") == "auth-only" and st.get("state") == AUTH_STATE.lower() and st.get("manager_pid") == proc.pid:
            print("auth proxy running: pid %d | listening 127.0.0.1:%d | forwards to %s:%d | key enforced (self-test passed) | Cloudflare not used"
                  % (proc.pid, cfg.auth_port, cfg.api_host, cfg.api_port))
            print("console log: %s (contains no key)" % log_path)
            return 0
        sleep(0.5)
    print("auth proxy pid %s is still starting after %.0f s; check `maple_tunnel.py status --auth-only`" % (proc.pid, wait))
    return 5


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="maple_tunnel.py", description="Remote connectivity manager for MAPLE (SSH by default; --auth-only = local Bearer-key proxy; --public = protected Cloudflare tunnel).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in (("run", "start in the foreground (ssh mode prints instructions; --auth-only starts the key proxy; --public the protected tunnel)"),
                    ("start", "start the --auth-only key proxy detached (key from $API_KEY, or --key-stdin)"),
                    ("status", "show tunnel / proxy state"), ("stop", "stop the manager")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--maple-home"), p.add_argument("--state-dir"), p.add_argument("--api-host"), p.add_argument("--api-port", type=int)
        p.add_argument("--llama-port", type=int), p.add_argument("--auth-port", type=int)
        if name == "run":
            g = p.add_mutually_exclusive_group()
            g.add_argument("--public", action="store_true", help="start a Cloudflare Quick Tunnel behind the Bearer-key proxy (off by default)")
            g.add_argument("--auth-only", action="store_true", help="Bearer-key proxy on 127.0.0.1 only; NO Cloudflare; $API_KEY required")
            p.add_argument("--check-interval", type=float), p.add_argument("--heartbeat-minutes", type=float)
            p.add_argument("--max-failures", type=int), p.add_argument("--max-restarts", type=int)
            p.add_argument("--ssh-alias", help="alias printed in the SSH instructions")
        if name == "start":
            p.add_argument("--key-stdin", action="store_true", help="read the API key from the first line of stdin (keeps it off the command line)")
            p.add_argument("--wait", type=float, default=20.0, help="seconds to wait for the proxy to report ready")
            p.add_argument("--ssh-alias", help="alias printed in the SSH instructions")
        if name == "status":
            p.add_argument("--json", action="store_true")
            p.add_argument("--auth-only", action="store_true", help="also probe the auth proxy (shown automatically when it is the active mode)")
    return ap


def main(argv: Optional[list] = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.cmd == "start":
        ns.auth_only = True
    cfg = config_from_args(ns)
    if ns.cmd == "status":
        return cmd_status(cfg, ns.json)
    if ns.cmd == "stop":
        return cmd_stop(cfg)
    if ns.cmd == "start":
        return cmd_start(cfg, ns.key_stdin, ns.wait)
    if cfg.auth_only:
        key = load_auth_only_key(os.environ)
        os.environ.pop("API_KEY", None)                 # nothing this process starts later can inherit it
        mgr = TunnelManager(cfg, key)
        try:
            return mgr.run_auth_only()
        except DuplicateTunnelManager as e:
            print("not started: %s" % e, file=sys.stderr)
            return 3
        finally:
            mgr.close()
    if not cfg.public:
        mgr = TunnelManager(cfg)
        try:
            return mgr.run_ssh_mode()
        finally:
            mgr.close()
    key, generated = get_api_key()
    mgr = TunnelManager(cfg, key)
    if generated:
        mgr.say("API_KEY was not set; generated a random key for this run (it is shown once when the tunnel is up).")
    try:
        return mgr.run_public()
    except DuplicateTunnelManager as e:
        print("not started: %s" % e, file=sys.stderr)
        return 3
    finally:
        mgr.close()


if __name__ == "__main__":
    sys.exit(main())
