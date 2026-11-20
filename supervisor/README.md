# MAPLE-compatible heartbeat and tunnel manager

Two standard-library Python files (no packages to install), adapted from your original notebook scripts:

| File | Role |
|---|---|
| `maple_heartbeat.py` | Supervises the **MAPLE API** (`127.0.0.1:8000`): health checks, status reports, recovery. |
| `maple_tunnel.py` | Remote connectivity, **separate process**. SSH instructions by default; a protected public tunnel only with `--public`. |

**Status: written and unit-tested locally with mocks. Not deployed, not run against a live MAPLE.**
Nothing here starts MAPLE, llama-server or cloudflared until *you* run it.

## Architecture (verified from the MAPLE repository)

```
Windows PC --(SSH)--> Kaggle
                      maple_heartbeat.py ──watches/recovers──> MAPLE API  127.0.0.1:8000   (python -m maple serve)
                      maple_tunnel.py   ──(optional)──> auth proxy 127.0.0.1:8088 ──> MAPLE API :8000
                                                         MAPLE API ──manages──> llama-server 127.0.0.1:8081 (internal)
```

* MAPLE starts, switches and stops `llama-server` itself (one model at a time). **Neither file ever launches, signals or
  restarts llama-server.** Port 8081 is only observed, and is never exposed.
* The heartbeat recovers MAPLE with the official `scripts/start_maple.sh`. To stop a hung API it sends **SIGTERM only** (MAPLE's
  handler then unloads llama-server). It never uses SIGKILL, because that would orphan llama-server on the GPU.
* The existing SSH-carrying `cloudflared ... ssh://localhost:2222` process (for example pid 1971) is **protected**: both
  tools only report it and never signal it. `maple_tunnel.py` only ever stops the cloudflared child *it* started.

## How the two files work together

They share one state folder (default `<MAPLE_HOME>/logs/supervisor/`, override with `--state-dir` or `MAPLE_SUP_STATE_DIR`):

* the tunnel manager writes `tunnel_state.json` (no key inside); the heartbeat reads it for the `Tunnel :` and `Endpoint :` lines;
* each has its own lock file (`supervisor.lock`, `tunnel.lock`), its own log and its own process, so stopping one cannot stop the other;
* the tunnel manager waits while MAPLE is down and never restarts MAPLE; the heartbeat never touches the tunnel.

## Requirements

* Python 3.9+ (written for 3.11/3.13), Linux for process control (Kaggle). Windows can run the tests and `--help` only.
* `bash` (for `start_maple.sh`), `nvidia-smi` optional (GPU lines show "unavailable" without it).
* `cloudflared` only for `--public`. Nothing is installed by these scripts.

## Quick start (on the Kaggle host, after you approve deployment)

```bash
cd supervisor                                  # this folder of the repository (deployed on Kaggle as /kaggle/working/MAPLE_compatible)
export MAPLE_HOME=/kaggle/working/MAPLE-llama.cpp

# 1) heartbeat (starts MAPLE through start_maple.sh if it is not running, then supervises it)
nohup setsid python3 maple_heartbeat.py run > /tmp/heartbeat.out 2>&1 < /dev/null &

# 2) optional: SSH-mode tunnel helper (prints the SSH command; starts nothing public)
python3 maple_tunnel.py run
```

Use `--no-start` for **watch-only** mode: the heartbeat never starts, restarts or signals MAPLE - not at launch, not after 3 failed checks, not when the API process disappears, and `--stop-maple-on-exit` is ignored. It still counts failures, logs, and writes status reports (state `UNHEALTHY`, one `WATCH-ONLY` notice). Only an explicit `stop --all` stops MAPLE.

### Public tunnel (opt-in, protected by a Bearer key)

```bash
export API_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"   # >= 24 chars; or omit to auto-generate
nohup setsid python3 maple_tunnel.py run --public > /tmp/tunnel.out 2>&1 < /dev/null &
cat /tmp/tunnel.out        # the URL and API key appear ONCE here
```

* If `API_KEY` is not set, a random key is generated for that run. The key is printed **once, on the console only**, together
  with the URL. It is never written to `tunnel.log`, the state file or any log. A notebook saves cell output, so treat that output as secret.
* In a notebook, `os.environ["API_KEY"] = secrets.token_urlsafe(32)` only reaches subprocesses started from the same kernel.
  A separate SSH shell has its own environment, so `export API_KEY=...` there, or let the tool generate one.
* The tool starts a small authenticating reverse proxy on `127.0.0.1:8088` and points cloudflared at **the proxy**, never at
  MAPLE. The proxy requires `Authorization: Bearer <key>` (constant-time compare, checked before any routing), allows only
  `/health`, `/healthz`, `/v1/models`, `/v1/chat/completions`, `/v1/completions` and `/v1/route`, strips the key before forwarding, and cannot reach port 8081.
* The URL is announced only after live checks through the **public** URL: no key -> HTTP 401 and correct key -> HTTP 200.
  If the public URL ever answers without a key the tunnel is stopped and reported as insecure.
* Client use: `OpenAI(base_url="<URL>/v1", api_key="<the key>")`.
* Quick Tunnel URLs change on every restart and are not an SLA. The key is the only protection, so anyone with the URL **and** key can use your GPU.

#### Cloudflare Access (not implemented here, on purpose)

Cloudflare Access (SSO or service-token policy in front of the hostname) needs a Cloudflare account, a *named* tunnel and a
domain. Quick Tunnels cannot sit behind Access. This code cannot create or verify that policy, so it does not claim it. If you want it:
create a named tunnel plus a self-hosted Access application and a service-token policy in the Zero Trust dashboard, point the
tunnel at `http://127.0.0.1:8088` (the proxy), and keep the Bearer key as a second layer. Verify an unauthenticated request is blocked at Cloudflare.

## Recommended: SSH tunnelling from Windows (nothing public)

```powershell
ssh -N -L 8000:127.0.0.1:8000 kaggle2
```
then `http://127.0.0.1:8000/v1` with any placeholder `api_key`. MAPLE has **no built-in authentication**, so keep it on loopback and use SSH.

## Commands

```bash
python3 maple_heartbeat.py run [--maple-home P] [--state-dir P] [--no-start] [--stop-maple-on-exit]
python3 maple_heartbeat.py status [--json]      # one live read-only check; exit 0 alive, 2 not
python3 maple_heartbeat.py stop [--all]         # --all also SIGTERMs the MAPLE API
python3 maple_tunnel.py run [--public]
python3 maple_tunnel.py status [--json]
python3 maple_tunnel.py stop                    # stops only the tunnel manager and ITS cloudflared
```

## Configuration

Order: command-line flag > environment variable > default.

| Setting | Default | Heartbeat env | Tunnel env |
|---|---|---|---|
| API host / port | `127.0.0.1` / `8000` | `MAPLE_SUP_API_HOST/PORT` | `MAPLE_TUN_API_HOST/PORT` |
| llama-server port (observed only) | `8081` | `MAPLE_SUP_LLAMA_PORT` | `MAPLE_TUN_LLAMA_PORT` |
| Check interval | `60` s | `MAPLE_SUP_CHECK_INTERVAL` | `MAPLE_TUN_CHECK_INTERVAL` |
| Report interval | `5` min | `MAPLE_SUP_HEARTBEAT_MINUTES` | `MAPLE_TUN_HEARTBEAT_MINUTES` |
| Startup timeout | `600` s | `MAPLE_SUP_SERVER_START_TIMEOUT` | - |
| Failures before recovery | `3` | `MAPLE_SUP_MAX_API_FAILURES` | `MAPLE_TUN_MAX_FAILURES` |
| SIGTERM wait before BLOCKED | `180` s (MAPLE waits for in-flight requests before it exits) | `MAPLE_SUP_STOP_TIMEOUT` / `--stop-timeout` | - |
| Restart limit | `5` per hour, exponential backoff 30 s -> 600 s | `MAPLE_SUP_MAX_RESTARTS` | `MAPLE_TUN_MAX_RESTARTS` (15 s -> 300 s) |
| Auth proxy port | - | - | `MAPLE_TUN_AUTH_PORT` (`8088`) |
| MAPLE project | auto-detect or `$MAPLE_HOME` | `--maple-home` | `--maple-home` |
| State and log folder | `<MAPLE_HOME>/logs/supervisor` | `MAPLE_SUP_STATE_DIR` | `MAPLE_SUP_STATE_DIR` |

The API host must stay on loopback (other values are refused). No Kaggle path is hard-coded.

## What the heartbeat does

* Every 60 s: process check, `GET /health`, `GET /v1/models`. It **never sends inference requests**.
* Every 5 min: report to the console, `status.json`, `reports.jsonl` (JSON lines) and `heartbeat.log` (structured JSON events).
* **3 consecutive failures** before recovery. Stopping a hung API waits up to 180 s for it to exit (it finishes in-flight requests first); after that it is reported BLOCKED. A missing MAPLE process with a refused connection recovers at once.
* A slow `/health` right after a model load or switch (inflight request, switch counter or loaded model changed, 180 s grace) is reported as **busy**, never counted as a crash.
* Missing model files (HTTP 503 `degraded`) and a differing model roster are reported as warnings; a restart cannot fix them, so none is attempted.
* Before recovery it inspects the process, the port and the log tails, and records orphan llama-server processes (reported, never touched).
  It refuses to start a second server if port 8000 is held by a non-MAPLE process.
* Recognised IDs: `deepseek`, `gemma`, `qwen3`, `fast`, alias `quant` -> `qwen3`, and `auto`. Models are never force-loaded.
* By default, stopping the heartbeat **leaves MAPLE running**. Add `--stop-maple-on-exit` to stop it as well.

Report layout (from your example):
```
[2026-11-20 12:31:12] Uptime: 205 minutes
Model       : qwen3 (Name)  | none loaded (models load on the first request)
Llama API   : ONLINE
Tunnel      : RUNNING | DISABLED (SSH tunnelling; no public tunnel)
Endpoint    : https://<name>.trycloudflare.com  (Bearer key required) | none (SSH: ssh -N -L ...)
```
"Model" is whatever MAPLE reports as loaded. MAPLE loads models on demand, so *none loaded* right after startup is normal.

## Stopping everything cleanly

```bash
python3 maple_tunnel.py stop                # only the tunnel manager + its own cloudflared
python3 maple_heartbeat.py stop             # supervisor only; MAPLE keeps running
python3 maple_heartbeat.py stop --all       # supervisor, then SIGTERM to the MAPLE API (it stops llama-server)
```
Never use `pkill cloudflared` or `pkill python`: that would also kill your SSH tunnel (pid 1971).

## Kaggle limitations

* Sessions end (about 12 h, idle limits); everything running stops. These tools cannot extend a session.
* Quick Tunnel URLs change each time; the SSH `cloudflared` URL also changes per session.
* `/kaggle/working` persists only as the platform saves it; process state and locks do not survive a restart (stale locks are detected and replaced).
* Two T4s are independent 15 GiB devices; this tool does not change MAPLE's placement.

## Tests (local only, all mocked)

```bash
# run from inside the supervisor/ folder
python -m unittest discover -s tests -v
```
Three tests in `tests/test_maple_tunnel_authonly.py` (class `RealProcess`) start a throw-away `maple_tunnel.py --auth-only` process on random loopback ports against a fake upstream; everything else is fully mocked. They never contact MAPLE, Kaggle or Cloudflare.

See `docs/TEST_RESULTS.md`, `docs/COMPARISON.md`, `SHA256SUMS.txt` and `UPLOAD_MANIFEST.sha256`.


## `--auth-only` mode (real Bearer-key proxy, loopback only)
`maple_tunnel.py run|start|status|stop --auth-only` puts a key-checking proxy on 127.0.0.1:8088 in front of MAPLE (127.0.0.1:8000). No Cloudflare is started. See **docs/AUTH_ONLY_GUIDE.md** for design, limits, key handling and the Kaggle/Windows commands. Unit-tested (mocks and loopback); uploaded to the Kaggle host but not yet activated or tested live against real MAPLE.
