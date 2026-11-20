# Comparison: original scripts -> MAPLE-compatible files

Originals (untouched, not part of this repository): `model_hearbeat_example.py` and `cloudflare_tunnel.py`.

## model_hearbeat_example.py -> maple_heartbeat.py

| Original | Result | In the new file |
|---|---|---|
| `HEARTBEAT_MINUTES=5`, `CHECK_INTERVAL=60`, `SERVER_START_TIMEOUT=600`, `MAX_API_FAILURES=3`, `REQUEST_TIMEOUT=15` | **Preserved** (same values) | module constants + `Config`, overridable by flag / env |
| `API_URL`, `HEALTH_URL`, `MODELS_URL` (127.0.0.1:PORT) | **Modified** | port 8000 (MAPLE API) instead of the llama-server port; `Config.api_url` |
| `is_llama_ready()` (checks the server) | **Modified** | `interpret_health()`, `interpret_models()`: validate the *MAPLE* `/health` and `/v1/models` shape, never trust a bare 200 |
| `start_llama_server()` (launched llama-server directly with model, GPU and context flags) | **Replaced** | `start_maple()`: runs the official `scripts/start_maple.sh --host --port`; MAPLE owns llama-server |
| `stop_llama_server()` (killed the server) | **Replaced** | `stop_maple_api()`: SIGTERM of the verified MAPLE API only (its handler unloads llama-server); no SIGKILL, never llama-server |
| heartbeat loop with 3 failures -> restart | **Preserved, extended** | `Supervisor.tick()` + `recover()`: 3 failures, restart limit, exponential backoff, pre-recovery inspection |
| `start_tunnel()` / `display_connection_info()` | **Moved** | tunnel is now `maple_tunnel.py`; the heartbeat only reads its state file |
| periodic report (Uptime / Model / Llama API / Tunnel / Endpoint) | **Preserved, extended** | `format_report()`: same five lines, plus GPU, RAM, disk, llama-server (managed by MAPLE), warnings |
| model name from the GGUF file name / `MODEL_ALIAS` / vision `mmproj` | **Removed** | MAPLE has 4 models + `auto`; the *loaded* model comes from `/health` (`runtime.loaded`), the display name from `/v1/models`. No vision/mmproj in MAPLE |
| `HEADERS` / API-key use for the llama API | **Removed** | MAPLE ignores `Authorization`; the heartbeat sends only unauthenticated GETs to loopback |
| (not in original) | **New** | `/v1/models` roster check (deepseek, gemma, qwen3, fast, `quant`->qwen3, `auto`); busy-vs-crash logic for model switching; nvidia-smi GPU memory/utilisation (graceful when missing); RAM (`MemAvailable`); disk; structured JSON log + `status.json` + `reports.jsonl`; single-instance lock with stale-lock takeover; `status` and `stop` commands; orphan llama-server reporting; protected SSH-cloudflared reporting |

## cloudflare_tunnel.py -> maple_tunnel.py

| Original | Result | In the new file |
|---|---|---|
| `shutil.which("cloudflared")` + `FileNotFoundError` | **Preserved** | `which=shutil.which`; a clear message and a hint to use SSH |
| `Popen([cloudflared, "tunnel", "--url", ...])` | **Modified** | targets `http://127.0.0.1:8088` (auth proxy), not `localhost:8080`; adds `--no-autoupdate`; removed `--http-host-header` (not needed for the proxy) |
| output to `tunnel.log` | **Preserved** | `<state-dir>/tunnel.log`, cleared at each start so an old URL is never reused |
| regex `https://[a-zA-Z0-9-]+\.trycloudflare\.com` and 60 s wait | **Preserved** | `detect_url()`, `URL_TIMEOUT=60` |
| "process exited" detection | **Preserved, extended** | exits during startup and while running both detected, tail of `tunnel.log` reported |
| returns `(proc, url)` | **Modified** | `start_public()` returns `(ok, detail)`; the URL is only set after verification |
| (not in original) | **New** | auth proxy (Bearer, endpoint allow-list, no key forwarded); public-URL verification (401 without key, 200 with key); automatic stop if unauthenticated access works; recovery with backoff and limit; lock; orphan cleanup only when the command line proves it is ours; SSH mode as the default; protected SSH-cloudflared detection; the URL and key printed once |

## Your API-key snippet
`os.environ["API_KEY"] = secrets.token_urlsafe(32)` is honoured: `maple_tunnel.py --public` uses `$API_KEY` if set (min 24 chars), otherwise
generates one the same way. Difference from your snippet: the key is shown **once**, in the connection banner with the URL, not repeatedly,
and it is stripped from the environment of cloudflared and of MAPLE.

## Compatibility problems and honest limits
1. **No MAPLE authentication.** A public tunnel is therefore off by default; `--public` adds a Bearer proxy that I tested only locally with a fake upstream. Not tested against real Cloudflare.
2. **Cloudflare Access is not implemented** (needs your account/domain and a named tunnel; Quick Tunnels cannot use it). Documented in the README, not claimed.
3. **Unverified against live MAPLE/cloudflared.** The HTTP shapes come from `api.py`, process detection from `start_maple.sh`/`__main__.py`, and all of it is unit-tested with mocks. Real processes (`/proc` parsing, `start_new_session`, SIGTERM handling) are not exercised here: first live run needs your approval.
4. **Linux only for process control.** The code runs on Windows for tests and `--help`, but cannot supervise there.
5. **Notebook key scope.** An `API_KEY` set inside a notebook kernel reaches only subprocesses of that kernel.
6. **Key in notebook output.** Printing the key once makes it part of saved notebook output: treat that as a secret.
7. **Model line.** MAPLE loads on demand, so "Model" is often `none loaded` right after start. The original's constant model name (Qwen3.8-27B) does not apply: MAPLE has no such model.
8. **Recovery limits.** A hung API that ignores SIGTERM is reported as BLOCKED, never SIGKILLed (that would orphan llama-server); a port held by a non-MAPLE process also blocks.
9. **Port 8088** is my choice for the auth proxy (configurable); it is not a MAPLE default.
10. **Test authoring errors** (found and fixed before the final run): three of my tests had wrong timing/over-broad regexes; the code under test did not change for them.


## Addendum: maple_tunnel.py before and after `--auth-only`
The earlier script (SHA256 `abdae690fd7b0106...`) is not shipped in this repository. Difference to the current one: +376 / -19 lines, 22 hunks. No version number changed.

| Area | Before `--auth-only` | With `--auth-only` |
|---|---|---|
| Modes | ssh (default), `--public` | ssh (default), **`--auth-only`**, `--public` (unchanged behaviour) |
| Key check | only inside `--public` | same proxy, now also `--auth-only`; `hmac.compare_digest`; 401 before path handling |
| Allowed endpoints | 5 | same 5, passed as a parameter; auth-only list asserted exactly by a test |
| Key source | `API_KEY` or generated | auth-only: `API_KEY` or `--key-stdin` only; never generated, never shown |
| Commands | run, status, stop | + `start` (detached, ready-wait), `status --auth-only`, `stop` unchanged semantics |
| Bind | 127.0.0.1 | 127.0.0.1 only (explicit bind error message, test for non-loopback refusal) |
| Reliability fix | - | lingering close: early 401/404 sent before the request body is read could be lost to a TCP reset (measured ~1 in 40 on loopback); fixed and regression-tested |
| Header handling | Authorization stripped | unchanged; also asserted that MAPLE never receives it |
| Processes | may start cloudflared (`--public`) | auth-only never starts or signals cloudflared/MAPLE/llama-server/heartbeat |
| State file | url = public URL | auth-only: url = loopback `http://127.0.0.1:8088/v1` only after the self-test passed |

Not changed: `maple_heartbeat.py` (8af6007b...), the two original source scripts, MAPLE.
