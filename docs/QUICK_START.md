# MAPLE Quick Start (daily commands only)

The full explanations, expected output, troubleshooting and security notes are in **MAPLE_COMPLETE_OPERATING_GUIDE.md**; section numbers below refer to it. Every command here is copied from the guide. Project version 0.1.0.

Labels: **[KAGGLE NOTEBOOK]** or **[WINDOWS PowerShell]** says where to run it. Tags say how much evidence exists (definitions in guide section 0): **[LIVE-VERIFIED]** ran for real, **[UNIT-TESTED]** covered by automated tests only, **[MOCKED]** run with a stand-in, **[UNVERIFIED]** never run.

> Ground rules: never print or paste the API key; never forward port 8000 or 8081; never `kill -9` anything; never stop Kaggle pid 1971 or any cloudflared you did not start.
> **Not yet run live end to end:** the proxy against real MAPLE, the forward to 8088, `maple_tunnel.py start`/`stop`, and `maple_heartbeat.py stop`/`stop --all` (guide section 14.4).

## 1. Start on Kaggle
1. New Kaggle session: **Accelerator = GPU T4 x2**, Internet on.
2. Run **your existing SSH-setup cell**, unchanged. It is not part of this documentation and must not be recreated from it (guide 3.2).
3. Get the hostname if your cell did not print it. **[KAGGLE NOTEBOOK]** **[LIVE-VERIFIED]** (equivalent check):
```python
import re
t = open("/tmp/tunnel.log", errors="replace").read()
hosts = re.findall(r"https://([a-zA-Z0-9-]+\.trycloudflare\.com)", t)
print(hosts[-1] if hosts else "no hostname yet - has your SSH-setup cell finished?")
```
4. **[WINDOWS]** put it in `%USERPROFILE%\.ssh\config` under `Host kaggle2` as `HostName <host>` (bare name, no `https://`; change nothing else). Test **[LIVE-VERIFIED]**: `ssh -v kaggle2 "echo connected; hostname"`.
5. **Cell 1: Preflight** (read-only, **[LIVE-VERIFIED]**): copy it from guide section 2.3. Every line must be `OK`.
6. **Cell 2: Start heartbeat** (starts MAPLE for you; safe to run twice). Launch branch **[MOCKED]**, command itself **[LIVE-VERIFIED]** in earlier stages. **[KAGGLE NOTEBOOK]**, identical to guide section 4.5:

```python
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

D = Path("/kaggle/working/MAPLE_compatible")        # heartbeat + tunnel scripts, state/ folder
P = Path("/kaggle/working/MAPLE-llama.cpp")         # MAPLE project (scripts/start_maple.sh)
S = D / "state"                                      # logs, lock files, status.json
ARGS = ["--maple-home", str(P), "--state-dir", str(S)]
NV = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"}


def cmdline(pid):
    try:
        return Path("/proc/%d/cmdline" % pid).read_bytes().decode("utf-8", "replace").replace("\0", " ").strip()
    except OSError:
        return None


def http_status(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def listeners():
    """{port: [addresses]} of TCP listeners; 127.0.0.1 shows as 127.0.0.1, anything else is flagged by the caller."""
    out = {}
    for f, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            p = ln.split()
            if p[3] != "0A":
                continue
            addr, port = p[1].rsplit(":", 1)
            if v6:
                ip = "::1" if addr == "0" * 31 + "1" else ("::" if set(addr) == {"0"} else "ipv6:" + addr)
            else:
                ip = ".".join(str(int(addr[i:i + 2], 16)) for i in (6, 4, 2, 0))
            out.setdefault(int(port, 16), set()).add(ip)
    return {k: sorted(v) for k, v in out.items()}


def lock_pid(name):
    try:
        return int(json.loads((S / name).read_text()).get("pid") or 0)
    except Exception:
        return 0


# ---------------------------------------------------------------- start the MAPLE heartbeat supervisor (detached)
# The supervisor starts MAPLE for you (scripts/start_maple.sh) and keeps it alive. Do NOT start MAPLE by hand.
# Safe to run twice: if a supervisor already holds the lock, nothing is started.
S.mkdir(parents=True, exist_ok=True)
running = lock_pid("supervisor.lock")
if running and "maple_heartbeat" in (cmdline(running) or ""):
    print("The heartbeat is ALREADY RUNNING (pid %d). Nothing was started." % running)
else:
    env = {k: v for k, v in os.environ.items() if k != "API_KEY"}     # neither the heartbeat nor MAPLE ever receives the API key
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = ["python3", "-B", str(D / "maple_heartbeat.py"), "run"] + ARGS
    log = S / "heartbeat_console.log"
    with open(log, "ab") as console:                                   # console output only; the real log is state/heartbeat.log
        sup = subprocess.Popen(cmd, cwd=str(D), stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)       # detached: survives this cell and a kernel restart
    print("heartbeat supervisor started, pid %d" % sup.pid)
    print("console log:", log)
    t0 = time.time()
    while time.time() - t0 < 600:                                      # MAPLE's startup timeout in the heartbeat is also 600 s
        rc = sup.poll()
        if rc is not None:
            print("The supervisor EXITED with code %s (3 = another heartbeat already holds the lock)." % rc)
            print((log.read_text(errors="replace")[-800:]))
            break
        if http_status("http://127.0.0.1:8000/health") == 200:
            print("MAPLE API is healthy after %d s (no model is loaded yet; models load on the first request)." % (time.time() - t0))
            break
        time.sleep(3)
    else:
        print("MAPLE is not healthy after 600 s. The supervisor is still running; read state/heartbeat.log and state/maple_api.log.")
    time.sleep(5)
    print("supervisor still alive:", sup.poll() is None, "| lock file pid:", lock_pid("supervisor.lock"))
```

## 2. Authentication proxy (key made on Windows, sent over SSH stdin)
The command is `maple_tunnel.py start --key-stdin`. **[WINDOWS PowerShell]**, same commands as guide 7.3 (key generation **[LIVE-VERIFIED]**; the live start **[UNVERIFIED]**, its logic **[UNIT-TESTED]**). Keep this window open; the key lives only here:

```powershell
# Create a real random API key in THIS PowerShell window only. Nothing is printed or saved.
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:MAPLE_API_KEY = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
"Key created for this window only: $($env:MAPLE_API_KEY.Length) characters (value not shown)."
```
```powershell
$S = "/kaggle/working/MAPLE_compatible"
$env:MAPLE_API_KEY | ssh kaggle2 "python3 -B $S/maple_tunnel.py start --key-stdin --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"
```
Expected (from the script's print format, not captured live): `auth proxy running: pid … | listening 127.0.0.1:8088 | … key enforced (self-test passed)`.

A key made in a notebook cell (guide 7.4) cannot be used from Windows without moving it out of Kaggle, which the guide forbids: use this Windows method.

## 3. Connect (Windows)
Window 1, leave it open (**[UNVERIFIED]** for 8088; same shape **[LIVE-VERIFIED]** for 8000):
```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 8088:127.0.0.1:8088 kaggle2
```

## 4. Test (Windows, window 2, the window that holds `$env:MAPLE_API_KEY`)
**[MOCKED]** (local stub; unverified against the real proxy), identical to guide section 9 "Short form":
```powershell
$h = @{ Authorization = "Bearer $env:MAPLE_API_KEY" }
(Invoke-RestMethod http://127.0.0.1:8088/v1/models -Headers $h).data.id          # model list
try { Invoke-WebRequest http://127.0.0.1:8088/v1/models -UseBasicParsing } catch { "no key -> HTTP " + [int]$_.Exception.Response.StatusCode }   # 401
```
Full six-check script: guide section 9. OpenAI SDK (`[MOCKED]`): guide section 10:
```powershell
python -m pip install openai
python maple_client.py "Hello"
```
`base_url = http://127.0.0.1:8088/v1`, `api_key` from `$env:MAPLE_API_KEY`; models `fast`, `gemma`, `deepseek`, `qwen3` (alias `quant`), `auto`; use a 600 s or longer timeout.

## 5. Status (read-only, **[LIVE-VERIFIED]**)
```powershell
$S = "/kaggle/working/MAPLE_compatible"
$A = "--maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"
ssh kaggle2 "python3 -B $S/maple_heartbeat.py status $A"
ssh kaggle2 "ss -ltn" | Select-String ':(8000|8081|8088)\s'
```
Is the proxy up? Do **not** rely on `maple_tunnel.py status --auth-only` when it may be stopped: it then prints a misleading "EXPOSED" (guide section 0, item 3). Use the keyless check (guide 7.5, **[LIVE-VERIFIED]** for the stopped state):
```powershell
ssh kaggle2 "curl -s -o /dev/null -m 5 -w 'proxy 8088 keyless request -> HTTP %{http_code}\n' http://127.0.0.1:8088/v1/models"
```
`HTTP 000` = nothing running; `HTTP 401` = running and enforcing the key; `HTTP 200` = **stop the proxy and report it**.

## 6. Shut down (in this order)
1. **Windows forward:** click its window, press **Ctrl+C**.
2. **Proxy only** (uses `$S` and `$A` from step 5; **[UNIT-TESTED]**, not run live):
```powershell
ssh kaggle2 "python3 -B $S/maple_tunnel.py stop $A"
```
3. *Only if you also want MAPLE off* (otherwise stop after step 2; heartbeat and MAPLE keep running). **[UNIT-TESTED]**, not run live:
```powershell
ssh kaggle2 "python3 -B $S/maple_heartbeat.py stop --all $A"
```
It stops the supervisor first, then sends SIGTERM to the MAPLE API (up to about 3.5 minutes). Never use `kill -9`.
4. **Verify** (**[LIVE-VERIFIED]** commands): only pid 1971 (cloudflared) remains, nothing on 8000/8081/8088, GPUs near 0 MiB:
```powershell
ssh kaggle2 "ps -eo pid,ppid,etime,args | grep -E 'maple|llama-server|cloudflared' | grep -v grep"
ssh kaggle2 "ss -ltn" | Select-String ':(8000|8081|8088)\s'
ssh kaggle2 "LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"
```
`stop` = supervisor only (MAPLE keeps running, unsupervised). `stop --all` = supervisor, then MAPLE. Stopping only the API while the heartbeat runs makes the heartbeat start it again at its next check.

## 7. Quick fixes
| Symptom | Do |
|---|---|
| `bad handshake` / ssh fails | New hostname (steps 1.3 and 1.4) or the session expired: restart from step 1 |
| `Address already in use` on 8088 | An old forward: Ctrl+C it (guide 8.5), or use `-L 18088:127.0.0.1:8088` |
| 401 | The key differs from the proxy's: restart the proxy with this window's key (step 2) |
| 404 | The path is not one of `/health /v1/models /v1/chat/completions /v1/completions /v1/route` |
| 502 | MAPLE is down: the heartbeat recovers it; check status |
| Timeout on the first request | A model is loading: wait or raise the timeout |
| Key exposed | Stop the proxy, make a new key, start the proxy (guide 7.7) |
