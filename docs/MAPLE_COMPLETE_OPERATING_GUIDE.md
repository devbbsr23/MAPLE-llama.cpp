# MAPLE Complete Operating Guide

Run the MAPLE multi-model AI server on Kaggle (2× Tesla T4) under heartbeat supervision, protect it with a real API key, reach it from your Windows PC through an SSH forward, and shut it down safely, independently.

- Project version: **0.1.0** (unchanged by this guide).
- Scripts described here: `maple_heartbeat.py` and `maple_tunnel.py` (repository folder `supervisor/`, commit `1653815`, pushed to `main` of `devbbsr23/MAPLE-llama.cpp`).
- The commands were derived from the actual source of `supervisor/` and checked read-only against a live Kaggle installation (2026-11-20). Evidence levels are given per command by the labels in section 0.

> **Where each command runs.** Every code block is labelled **[KAGGLE NOTEBOOK]**, **[KAGGLE via SSH]** or **[WINDOWS PowerShell]**. Do not mix them up.

## 0. How to read the verification labels

Every command block in this guide carries exactly one of four tags. Do not treat them as equivalent.

| Tag | Meaning |
|---|---|
| **[LIVE-VERIFIED]** | Executed for real against the actual Kaggle server or your Windows PC, either read-only while this guide was prepared or in an earlier approved live stage (Stages 3, 4, 5 and 6a of the heartbeat validation). |
| **[UNIT-TESTED]** | The behaviour of the real script is covered by the project's automated tests (real code, simulated or loopback I/O, a fake upstream), but the command was **not** executed on the live server. |
| **[MOCKED]** | The guide's own cell or script was executed with the risky part replaced by a stand-in: process launches mocked, a local stub web server instead of MAPLE, or a synthetic process list. The wrapper logic is checked; the real action is not. |
| **[UNVERIFIED]** | Never executed in any form. Described from the source code only. |

A block can mix tags when its parts have different evidence; the tag then names the part. The complete per-command list is in [section 14](#14-verification-ledger-what-is-and-is-not-proven).

### Where the files are
- The heartbeat, tunnel, tests and docs of the supervisor are in `supervisor/` of this repository (added in commit `1653815`).
- This guide and `QUICK_START.md` live in `docs/`. They describe project version **0.1.0**; no version was changed by adding them.
- The deployed Kaggle copy of `maple_tunnel.py` (`b255bce5…`) differs from the repository copy (`supervisor/maple_tunnel.py`) only by the `__version__` string (the Kaggle copy says `0.2.0`, the repository copy `0.1.0`). `maple_heartbeat.py` is identical everywhere (`8af6007b…`).
- The verification tags below describe the state when the guide was written (2026-11-20). Anything tagged **[UNVERIFIED]** may have been verified since; check the validation records before relying on that tag.

### Things that differ from what you may expect (read this once)
1. **The proxy is started with `maple_tunnel.py start --key-stdin`** (key from stdin) or plain `maple_tunnel.py start` (key from the `API_KEY` environment of the process that runs it). The `start` command takes no mode flag, because it is *always* the authentication proxy. The `--auth-only` flag belongs to `run` and `status` only. **[LIVE-VERIFIED]** from the `--help` output of every sub-command.
2. **Heartbeat API states** are exactly `UNKNOWN`, `ONLINE`, `DEGRADED`, `UNHEALTHY`, `RECOVERING`, `OFFLINE`, `BUSY`, `GAVE_UP` and `BLOCKED` (section 6.4). Whether the *supervisor* is up is a separate fact, shown as `Supervisor : running (pid N)` or `Supervisor : not running`; it is not an API state. `WATCH-ONLY` is a launch mode (`--no-start`), not a state either.
3. **Known misleading status output (not fixed in this version).** When the proxy is **not running**, `maple_tunnel.py status --auth-only` prints `listening on []  | loopback only: NO - EXPOSED, stop it <<<` and `Key enforced : NO / not answering`. Nothing is exposed: the empty listener list is simply mis-evaluated as "not loopback-only". Observed live on 2026-11-20 with port 8088 closed. **Do not use that line to decide whether the proxy is exposed.** Use the port-listener check in section 7.5 instead.
4. **A key generated in the notebook (Option B) cannot conveniently be used from Windows**: the scripts deliberately never print or store it, so the only way to move it is a secure transfer you would have to build yourself. Section 7.2 explains why. For a Windows client use **Option A**.
5. The proxy against real MAPLE, the Windows forward to 8088 and the real `stop` commands have **never been run live**. Section 14 lists exactly what is proven and by which method.

---

## 1. Architecture

```
Windows PC
   |
   |  SSH connection  (ProxyCommand: cloudflared access ssh --hostname <host>.trycloudflare.com)
   v
Kaggle SSH access connection   cloudflared pid 1971: "tunnel --url ssh://localhost:2222"  -> sshd :2222
   |
   +-- MAPLE heartbeat supervisor   maple_heartbeat.py run              (pid 4044 in the last session)
   |       |
   |       +-- MAPLE API            python -m maple serve  127.0.0.1:8000     (pid 4627, parent = heartbeat)
   |               |
   |               +-- llama-server 127.0.0.1:8081   (child of the API, only while a model is loaded)
   |
   +-- Authentication proxy         maple_tunnel.py run --auth-only  127.0.0.1:8088
           |                         (started by `maple_tunnel.py start`; NOT a child of the heartbeat)
           +-- authorised requests are forwarded to the MAPLE API at 127.0.0.1:8000
```

**Does the diagram match the real implementation?** Yes, with these precise corrections, all checked against the live process list and the source:

- The heartbeat **is** the parent of the MAPLE API (`ppid 4044` for pid 4627). The API manages `llama-server` itself; the heartbeat never starts or signals it.
- The authentication proxy is **not** under the heartbeat. It has its own manager process, lock file (`state/tunnel.lock`) and state file (`state/tunnel_state.json`). The heartbeat only *reads* that state file for its 5-minute report.
- The Kaggle SSH access connection (cloudflared pid 1971) belongs to you. No MAPLE script starts, stops or signals it. Both scripts only *report* it ("protected").
- The Windows SSH forward (section 8) is a fourth, separate thing that exists only on your PC.

### What each component does
| Component | Port | Started by | Must keep running? |
|---|---|---|---|
| Kaggle SSH access (cloudflared + sshd) | 2222 (local to Kaggle) | your SSH-setup notebook cell | **Yes** while you want to connect. If it dies you cannot reach Kaggle at all. |
| Heartbeat supervisor | none | you (section 4) | Yes, if you want automatic recovery. |
| MAPLE API (OpenAI-compatible, model router) | 127.0.0.1:8000 | the heartbeat, via `scripts/start_maple.sh --port 8000` | Yes while serving. |
| llama-server (the actual model process) | 127.0.0.1:8081 | the MAPLE API, on the first request for a model | Managed by MAPLE; appears only while a model is loaded. |
| Authentication proxy | 127.0.0.1:8088 | you (section 7) | Only while clients are connected. |
| Windows SSH forward | 127.0.0.1:8088 on your PC | you (section 8) | Only while you use the API. |

### Which ports are internal
All of 8000, 8081 and 8088 listen on **127.0.0.1 only**. Nothing is reachable from the internet except through your authenticated SSH connection. Only **8088** is ever forwarded to your PC. Never forward 8000 or 8081.

### Why MAPLE must not be exposed directly
The MAPLE API on 8000 has **no authentication at all**. It does not read or check `API_KEY` (the heartbeat even removes `API_KEY` from the environment it passes on, and MAPLE contains no key-checking code). Anyone who can reach port 8000 can use your GPUs, load models and fill the disk-backed logs. Any "key" sent to port 8000 is simply ignored, so a forward to 8000 looks authenticated and is not.

### Why the heartbeat and the proxy are separate processes
- They have different jobs and failure modes: the heartbeat repairs MAPLE, the proxy only filters requests.
- The proxy can be stopped and started (for example to rotate the key) without touching MAPLE, the loaded model or the heartbeat.
- The heartbeat is deliberately forbidden from handling the key: it never receives `API_KEY`. A bug or log line in the supervisor therefore can never leak it.
- Stopping the proxy never stops MAPLE; stopping the heartbeat never stops the proxy.

---

## 2. Fresh Kaggle session: preparation

### 2.1 Open the session
1. Open your Kaggle notebook. In the right-hand **Settings** panel choose **Accelerator → GPU T4 x2** (the exact wording is Kaggle's and may change). MAPLE's placement and the tested context sizes assume two T4 GPUs.
2. Make sure **Internet** is on (cloudflared needs it).
3. Start the session and wait until it is running.

### 2.2 Understand what a session expiry means
A Kaggle session has a time limit and ends when stopped, idle, or out of quota. **Everything running inside it dies with it**: the heartbeat, MAPLE, llama-server, the authentication proxy and the SSH access tunnel. After that:
- your Windows SSH sessions and forwards stop working (`Connection closed`, `websocket: bad handshake`);
- the API key held only in a process or kernel environment is gone (this is by design);
- whether files in `/kaggle/working` survive depends on how the session ended and was restored. **Do not assume they do.** Run the preflight below.

The preflight reports machine uptime and whether pid 1971 exists. A very small uptime means a new machine.

### 2.3 Run the preflight check
It is **read-only**: it changes and starts nothing, downloads nothing, and does not hash the 17 GiB of models (it compares file sizes with `manifests/models_manifest.json`).

It checks: project and script presence, importing MAPLE, SHA-256 of both scripts against `SHA256SUMS.txt`, model files and sizes, two T4 GPUs and their memory, free disk, listening ports, and whether the heartbeat or the proxy is already running.

**[KAGGLE NOTEBOOK]**, **Cell 1: Preflight.** **[LIVE-VERIFIED]** (read-only; real output below):

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


# ---------------------------------------------------------------- MAPLE preflight (READ-ONLY)
# Run this first in every fresh Kaggle session. It changes nothing and starts nothing.
problems = []


def check(label, ok, detail=""):
    print(("OK       " if ok else "PROBLEM  ") + label + ((" - " + detail) if detail else ""))
    if not ok:
        problems.append(label)


print("== session")
up = float(open("/proc/uptime").read().split()[0])
print("Kaggle machine uptime: %.1f hours (a Kaggle session ends after its time limit; everything running dies with it)" % (up / 3600))
ssh = cmdline(1971) or ""
print("SSH access tunnel pid 1971: %s" % ("ALIVE - " + ssh if "ssh://" in ssh else "not found (normal if your SSH cell has not been run in this session, or it runs under another pid)"))

print("\n== installation")
check("python3 >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])
for label, path in (("MAPLE project", P), ("start script", P / "scripts" / "start_maple.sh"), ("MAPLE_compatible folder", D),
                    ("maple_heartbeat.py", D / "maple_heartbeat.py"), ("maple_tunnel.py", D / "maple_tunnel.py")):
    check(label, path.exists(), str(path))
r = subprocess.run(["python3", "-s", "-c", "import maple; print(maple.__file__)"], capture_output=True, text=True,
                   env=dict(os.environ, PYTHONPATH=str(P / "src")))
check("MAPLE python package imports", r.returncode == 0, (r.stdout or r.stderr).strip()[:100])
try:
    import hashlib
    for ln in (D / "SHA256SUMS.txt").read_text().splitlines():
        want, name = ln.split(" *", 1)
        got = hashlib.sha256((D / name).read_bytes()).hexdigest()
        check("checksum " + name, got == want, got[:12])
except OSError as e:
    check("SHA256SUMS.txt readable", False, str(e))

print("\n== models (existence + size only; no download, no hashing)")
try:
    manifest = json.loads((P / "manifests" / "models_manifest.json").read_text())["models"]
except Exception as e:                                           # noqa: BLE001
    manifest = {}
    check("models_manifest.json readable", False, str(e))
FILES = {"deepseek": "deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf", "gemma": "gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf",
         "qwen3": "quant/Qwen3-8B-Q4_K_M.gguf", "fast": "fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"}
for name, rel in FILES.items():
    f = Path("/kaggle/working/models") / rel
    want = (manifest.get(name) or {}).get("size_bytes") or (manifest.get(name) or {}).get("size")
    ok = f.exists() and (want is None or f.stat().st_size == want)
    check("model %-8s" % name, ok, "%s (%.2f GiB)" % (f, f.stat().st_size / 2**30) if f.exists() else "MISSING " + str(f))

print("\n== GPU, disk, ports")
r = subprocess.run(["/opt/bin/nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                   capture_output=True, text=True, env=dict(os.environ, **NV))
gpus = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
check("two Tesla T4 GPUs visible", len(gpus) == 2 and all("T4" in g for g in gpus), "; ".join(gpus) or r.stderr.strip()[:100])
free = __import__("shutil").disk_usage("/kaggle/working").free / 2**30
check("disk free in /kaggle/working >= 1.5 GiB", free >= 1.5, "%.1f GiB free" % free)
L = listeners()
for port, what in ((8000, "MAPLE API"), (8081, "llama-server (internal, only while a model is loaded)"), (8088, "authentication proxy")):
    print("         port %d %-55s %s" % (port, what, ("LISTENING on %s" % L[port]) if port in L else "closed"))

print("\n== what is already running")
hb, tn = lock_pid("supervisor.lock"), lock_pid("tunnel.lock")
print("heartbeat supervisor:", ("pid %d ALIVE" % hb) if hb and "maple_heartbeat" in (cmdline(hb) or "") else "not running")
print("auth proxy manager  :", ("pid %d ALIVE" % tn) if tn and "maple_tunnel" in (cmdline(tn) or "") else "not running")
print("\nRESULT:", "all checks passed" if not problems else "%d problem(s): %s" % (len(problems), "; ".join(problems)))
```

Expected output when everything is healthy and MAPLE is already running (real output, 2026-11-20):

```text
== session
Kaggle machine uptime: 5.2 hours (a Kaggle session ends after its time limit; everything running dies with it)
SSH access tunnel pid 1971: ALIVE - /usr/local/bin/cloudflared tunnel --url ssh://localhost:2222

== installation
OK       python3 >= 3.9 - 3.13.16
OK       MAPLE project - /kaggle/working/MAPLE-llama.cpp
OK       start script - /kaggle/working/MAPLE-llama.cpp/scripts/start_maple.sh
OK       MAPLE_compatible folder - /kaggle/working/MAPLE_compatible
OK       maple_heartbeat.py - /kaggle/working/MAPLE_compatible/maple_heartbeat.py
OK       maple_tunnel.py - /kaggle/working/MAPLE_compatible/maple_tunnel.py
OK       MAPLE python package imports - /kaggle/working/MAPLE-llama.cpp/src/maple/__init__.py
OK       checksum maple_heartbeat.py - 8af6007bba5d
OK       checksum maple_tunnel.py - b255bce5d806

== models (existence + size only; no download, no hashing)
OK       model deepseek - /kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf (4.50 GiB)
OK       model gemma    - /kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf (4.42 GiB)
OK       model qwen3    - /kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf (4.68 GiB)
OK       model fast     - /kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (1.04 GiB)

== GPU, disk, ports
OK       two Tesla T4 GPUs visible - 0, Tesla T4, 0 MiB, 15360 MiB; 1, Tesla T4, 0 MiB, 15360 MiB
OK       disk free in /kaggle/working >= 1.5 GiB - 4.7 GiB free
         port 8000 MAPLE API                                               LISTENING on ['127.0.0.1']
         port 8081 llama-server (internal, only while a model is loaded)   closed
         port 8088 authentication proxy                                    closed

== what is already running
heartbeat supervisor: pid 4044 ALIVE
auth proxy manager  : not running

RESULT: all checks passed
```

Meaning of the checks: any line that starts with `PROBLEM` must be resolved before you continue. A `closed` port is normal before you start things. A `PROBLEM two Tesla T4 GPUs visible` means the accelerator is not set to T4 x2: stop the session, change the setting, and start again.

### 2.4 If directories or models are missing
**Stop. Do not recreate or overwrite anything on a guess.** The installation is restored by the public restore procedure, which verifies checksums and never builds llama.cpp. Read these project documents first:
- `docs/RESTORATION.md` and `docs/RECOVERY_GUIDE.md` (restore procedure and recovery)
- `docs/TROUBLESHOOTING.md` (exit codes of the restore script)
- `docs/KAGGLE_DEPLOYMENT.md`

Models are external files: `/kaggle/working/models/{deepseek,gemma,quant,fast}/…gguf`. They are restored from your own source, never bundled in Git or the archives. Do not download or copy them unless the preflight says they are missing. Missing `MAPLE_compatible/` (the two supervisor scripts) is restored by re-uploading the approved files from your PC; ask before doing that.

---

## 3. Kaggle SSH connection

### 3.1 Which thing is which
| Name | What it is | Lives on |
|---|---|---|
| **Kaggle SSH access connection** | `cloudflared tunnel --url ssh://localhost:2222` (a Cloudflare *Quick Tunnel*) in front of `sshd`. It gives your PC a shell on Kaggle. | Kaggle (pid 1971 last session) |
| **Windows SSH forwarding tunnel** | `ssh -N -L 8088:127.0.0.1:8088 kaggle2` | Your PC |
| **MAPLE API** | the model server on 127.0.0.1:8000 | Kaggle |
| **Authentication proxy** | the key-checking front door on 127.0.0.1:8088 | Kaggle |

**Never stop pid 1971, or any cloudflared you did not start, to "fix" something.** Doing so cuts off your only way into the machine. Closing a Windows forward (Ctrl+C) never touches it.

### 3.2 Using your existing SSH setup
This guide **does not contain an SSH-setup cell, and none should be copied from anywhere else**. The cell that creates your SSH access is the one you already use in your Kaggle notebook; it is deliberately neither reproduced nor invented here.

What is known, **[LIVE-VERIFIED]** by read-only inspection of the running session: `sshd` is listening, and pid 1971 runs `/usr/local/bin/cloudflared tunnel --url ssh://localhost:2222` with its output written to `/tmp/tunnel.log`.

Procedure each session:
1. Run **your existing SSH-setup cell**, unchanged, and wait until it finishes.
2. If that cell already prints the hostname, use that. Otherwise read it from `/tmp/tunnel.log` with the cell in 3.3.
3. Put the hostname into your Windows SSH config (3.4) and test (3.5).

If the SSH-setup cell is missing or no longer works, stop and rebuild it deliberately from your own notes; do not improvise one in the middle of a session.

### 3.3 Find the hostname
The hostname is a random `<words-and-dashes>.trycloudflare.com` name that **changes every time the tunnel is created**. It appears in `/tmp/tunnel.log` in the banner box as `https://<hostname>.trycloudflare.com`.

**[KAGGLE NOTEBOOK]** **[LIVE-VERIFIED]**: read-only; prints the last hostname in the log. An equivalent check ran live with the value masked (it found exactly one well-formed hostname in the log):

```python
import re
t = open("/tmp/tunnel.log", errors="replace").read()
hosts = re.findall(r"https://([a-zA-Z0-9-]+\.trycloudflare\.com)", t)
print(hosts[-1] if hosts else "no hostname yet - has your SSH-setup cell finished?")
```

If it prints several different names over time, the **last** one is current. Treat the hostname as private; anyone who has it and a way to authenticate to sshd could try to connect.

### 3.4 Update the Windows SSH config
Edit `%USERPROFILE%\.ssh\config` (Notepad is fine). Your `kaggle2` entry has this shape (**[LIVE-VERIFIED]**: read from your PC, hostname masked):

```text
Host kaggle2
    HostName <NEW-HOSTNAME>.trycloudflare.com
    User root
    ProxyCommand cloudflared access ssh --hostname %h
```

Change **only the `HostName` value**: bare hostname, **no** `https://`, no trailing slash. Leave `ProxyCommand` alone (`%h` expands to the HostName). `cloudflared` must be on `PATH`: it is (`C:\cloudflared\cloudflared.exe`). In Git Bash run `export PATH="/c/cloudflared:$PATH"` first.

### 3.5 Test the connection
**[WINDOWS PowerShell]** **[LIVE-VERIFIED]** for the `ssh kaggle2 "<command>"` form (every live check in this guide used it); `-v` only adds debug output:

```powershell
ssh -v kaggle2 "echo connected; hostname"
```

Success: the last lines show `connected` and the Kaggle machine name. Without `-v` for normal use. In the verbose output look for lines mentioning the ProxyCommand and `Authenticated to`. Common failures are in section 12 (`websocket: bad handshake` = hostname changed or tunnel down; `Connection refused`; `Permission denied`).

Do not edit your keys or `known_hosts` unless you know why. The scripts in this guide use no key flags of their own.

### 3.6 Running notebook cells from Windows instead
Every **[KAGGLE NOTEBOOK]** cell in this guide is plain Python and can also be run from PowerShell without opening the notebook:

**[WINDOWS PowerShell]**
```powershell
Get-Content .\cell_status.py -Raw | ssh kaggle2 "python3 -I -"
```
(Save the cell as a file first.) The cells use only absolute paths and `/proc`, so they behave the same both ways. **[LIVE-VERIFIED]** with `cell_status.py` through this exact PowerShell pipe.

---

## 4. Start the heartbeat supervisor

### 4.1 Why start the heartbeat instead of MAPLE
- It **starts MAPLE for you** through the official `scripts/start_maple.sh --port 8000`. Do not start MAPLE by hand; a hand-started MAPLE is not supervised and the heartbeat then treats it as an existing process to watch.
- It checks MAPLE every 60 s and recovers it if it stops answering.
- It writes the logs and the 5-minute status report you use to know it is healthy.

### 4.2 How detection and recovery work (from the source, defaults)
- **Detection:** every `--check-interval` (60 s) it requests `/health` and `/v1/models` from `127.0.0.1:8000` with a 15 s timeout. The state becomes `ONLINE` only if both are `ok`.
- **Busy grace (by design, not observed live):** a *timeout* of `/health` within 180 s (`BUSY_GRACE`) after model activity (a load, a switch or an in-flight request) is shown as `BUSY` and is **not** counted as a failure. In Stage 5/6a no health check happened to overlap a model load, so this has not been seen live.
- **Failure counter:** other failed checks are counted (`failures n/3`). At `--max-api-failures` (3) recovery starts. **If the API process itself has disappeared** (connection refused and no `maple serve` process), recovery starts **immediately** without waiting for 3 failures (this is the Stage 6a case, recovered in about 47 s).
- **Recovery:** state `RECOVERING`. It sends **SIGTERM** to the MAPLE API process only (never `SIGKILL`), waits up to `--stop-timeout` (180 s) for it to exit, then starts MAPLE again with `scripts/start_maple.sh`, waiting up to `--start-timeout` (600 s). MAPLE's own handler stops `llama-server` cleanly.
- **Limits:** at most `--max-restarts` (5) restarts per hour with exponential backoff (15 s doubling up to 300 s). When exhausted it stops restarting and reports it instead of looping.
- **[LIVE-VERIFIED]** in Stage 6a: SIGTERM to the API was followed by recovery within about 47 s (the next check), new pid, parent still the heartbeat, no orphan `llama-server`.
- It never starts, signals or restarts `llama-server`, the proxy or cloudflared.

### 4.3 `--no-start` (watch-only) versus normal operation
- **Normal:** `run` starts MAPLE if it is not running and repairs it when it fails.
- **`run --no-start` (WATCH-ONLY):** the supervisor *only observes and reports*. It never starts, stops or restarts MAPLE: not at launch, not on failures, not on exit. A failing MAPLE shows as unhealthy with the message `WATCH-ONLY (--no-start): … no automatic recovery will be attempted`. The `start` path is blocked at four places in the code (lines 654, 730, 906, 934) and covered by tests. Use it when you started MAPLE yourself or want to watch without any risk.
- `--stop-maple-on-exit` (off by default) also stops MAPLE when the supervisor exits. Do not use it for normal operation.

### 4.4 Duplicate protection
The supervisor takes a lock file `state/supervisor.lock` (`{"pid": 4044, "started_utc": "…"}`). If you run it again while the first is alive, the second one prints `not started: another maple_heartbeat is already running (pid N)` and exits with **code 3**, changing nothing (**[LIVE-VERIFIED]** in Stage 4). A lock left behind by a crashed supervisor is detected as stale (the pid is gone, or it is not a `maple_heartbeat` process) and replaced; it never signals an unrelated process.

### 4.5 The notebook cell
It launches the tested command

`python3 -B /kaggle/working/MAPLE_compatible/maple_heartbeat.py run --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir /kaggle/working/MAPLE_compatible/state`

detached (own session: it survives the cell and a kernel restart), with `PYTHONDONTWRITEBYTECODE=1`, working directory `MAPLE_compatible`, console output appended to `state/heartbeat_console.log`, `API_KEY` removed from its environment. It then prints the supervisor PID, waits until MAPLE answers `/health` (up to the 600 s startup timeout), and confirms the supervisor is still alive. It is safe to run twice.

**[KAGGLE NOTEBOOK]**, **Cell 2: Start heartbeat.** Run this **before** you create any API key in the notebook kernel:

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

**Verification of Cell 2, by part:**
- The command it launches: **[LIVE-VERIFIED]** in Stages 3, 4 and 6a.
- The "already running" branch: **[LIVE-VERIFIED]** (the cell ran against the real lock of pid 4044 and launched nothing).
- The launch branch: **[MOCKED]**. With the launch replaced by a stand-in in a simulated fresh state, it built the command above with `cwd=MAPLE_compatible`, `PYTHONDONTWRITEBYTECODE=1`, `start_new_session=True`, stdin from `/dev/null`, output to the console log, and **no `API_KEY` in the child environment** even when the kernel had one.
- A real first launch from this notebook cell: **[UNVERIFIED]**.

Expected output on a fresh start (**illustrative**: built from the cell's print statements, not captured from a real fresh launch):
```text
heartbeat supervisor started, pid 4044
console log: /kaggle/working/MAPLE_compatible/state/heartbeat_console.log
MAPLE API is healthy after 25 s (no model is loaded yet; models load on the first request).
supervisor still alive: True | lock file pid: 4044
```
(pids and seconds will differ). If it says `The supervisor EXITED with code 3`, one is already running.

> If you prefer not to use the notebook: **[WINDOWS PowerShell]** `Get-Content .\cell_start_heartbeat.py -Raw | ssh kaggle2 "python3 -I -"` runs the same cell over SSH. **[UNVERIFIED]** for this particular cell (the same pipe form was verified with the read-only status cell, and the earlier stages launched the heartbeat over SSH).

---

## 5. Models and how MAPLE loads them
Verified live via `GET /v1/models`:

| id | Model | Role |
|---|---|---|
| `deepseek` | DeepSeek-Coder-V2-Lite MAPLE | Systems engineering and algorithm specialist |
| `gemma` | Gemma 4 26B MAPLE | General software and code review specialist |
| `qwen3` | Qwen3 8B Q4_K_M | Strategic reasoning and architecture specialist |
| `fast` | Qwen2.5 Coder 1.5B Q4_K_M | Ultra-fast triage and terminal copilot |
| `auto` | automatic specialist routing | picks one of the four |

`quant` is an **alias of `qwen3`** (`aliases: {"quant":"qwen3"}`). Only **one model is loaded at a time**; asking for a different model unloads the current one and loads the new one (a "switch"). In Stage 5 on kaggle2 the first request per model took 2.5 s (fast), 6.2 s (gemma) and 4.4 s (deepseek) with warm disk cache; a cold disk or a bigger switch can be much longer, so use generous timeouts (section 10).

Context sizes validated in earlier tests: fast 32 768, qwen3 32 768, deepseek 65 536, gemma 131 072. **DeepSeek 131 072 is not validated.** Known issue recorded in validation record B: `gemma` sometimes returns an empty reply on open-ended prompts; `qwen3` is a thinking model and can return empty `content` unless `max_tokens` is large (1024+).

---

## 6. Check MAPLE status

All **read-only**.

### 6.1 One cell for everything
**[KAGGLE NOTEBOOK]**, **Cell 4: Status.** **[LIVE-VERIFIED]** (read-only):

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


# ---------------------------------------------------------------- MAPLE status (READ-ONLY)
def run(title, cmd):
    print("\n$ " + title)
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **NV))
    print((r.stdout + r.stderr).rstrip())


run("maple_heartbeat.py status", ["python3", "-B", str(D / "maple_heartbeat.py"), "status"] + ARGS)
run("maple_tunnel.py status --auth-only", ["python3", "-B", str(D / "maple_tunnel.py"), "status", "--auth-only"] + ARGS)
print("\n$ ports (127.0.0.1 only is correct for all of them; 8081 only while a model is loaded; 8088 only while the proxy runs)")
L = listeners()
for port in (8000, 8081, 8088):
    addrs = L.get(port)
    print("  %d: %s" % (port, "closed" if not addrs else "%s%s" % (addrs, "   <<< NOT loopback-only, investigate" if any(a not in ("127.0.0.1", "::1") for a in addrs) else "")))
print("\n$ processes")
r = subprocess.run("ps -eo pid,ppid,etime,args | grep -E 'maple|llama-server|cloudflared' | grep -v grep", shell=True, capture_output=True, text=True)
print("\n".join(x[:150] for x in r.stdout.splitlines()))
print("\n$ GPU memory (MiB)")
r = subprocess.run(["/opt/bin/nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, env=dict(os.environ, **NV))
print(r.stdout.strip() or r.stderr.strip())
print("\n$ last 6 heartbeat log events (%s)" % (S / "heartbeat.log"))
for ln in (S / "heartbeat.log").read_text().splitlines()[-6:]:
    try:
        e = json.loads(ln)
        print("  %s %-8s %-26s %s" % (e.get("ts", "")[11:19], e.get("level", ""), e.get("event", ""), {k: v for k, v in e.items() if k not in ("ts", "level", "event")}))
    except ValueError:
        print("  " + ln[:150])
```

Real output (2026-11-20, MAPLE running, no model loaded, proxy not running):

```text

$ maple_heartbeat.py status

=================================================================
[2026-11-20 14:06:20] Uptime: 0 minutes
Model       : none loaded (models load on the first request)
Llama API   : ONLINE
Tunnel      : DISABLED (SSH tunnelling; no public tunnel)
Endpoint    : none (SSH: ssh -N -L 8000:127.0.0.1:8000 <ssh-alias>)
MAPLE API   : http://127.0.0.1:8000 pid=4627 | failures 0/3 | restarts 0/5 per 3600s
llama-server: managed by MAPLE | internal port 8081 closed (no model loaded) | pids none
GPU 0       : Tesla T4 0/15360 MiB, util 0%
GPU 1       : Tesla T4 0/15360 MiB, util 0%
RAM         : 30725 of 32098 MiB available (95.7%)
Disk        : 4.7 GiB free of 19.5 GiB
SSH tunnel  : cloudflared pid [1971] (protected, not managed by this tool)
=================================================================
Supervisor  : running (pid 4044)

$ maple_tunnel.py status --auth-only
Tunnel manager: not running
Mode          : ssh
Tunnel        : DISABLED
Endpoint      : none (use SSH tunnelling)
Auth          : none
Auth proxy    : 127.0.0.1:8088 | listening on [] | loopback only: NO - EXPOSED, stop it <<<
Key enforced  : NO / not answering (request without a key -> HTTP None)
MAPLE API     : HTTP 200 (direct, loopback)
SSH cloudflared (protected, not managed): [1971]

$ ports (127.0.0.1 only is correct for all of them; 8081 only while a model is loaded; 8088 only while the proxy runs)
  8000: ['127.0.0.1']
  8081: closed
  8088: closed

$ processes
   1971      50    04:17:26 /usr/local/bin/cloudflared tunnel --url ssh://localhost:2222
   4044       1    01:51:22 python3 -B /kaggle/working/MAPLE_compatible/maple_heartbeat.py run --maple-home /kaggle/working/MAPLE-llama.cpp --state-di
   4627    4044    01:39:20 python3 -s -m maple serve --host 127.0.0.1 --port 8000

$ GPU memory (MiB)
0, 0 MiB, 15360 MiB
1, 0 MiB, 15360 MiB

$ last 6 heartbeat log events (/kaggle/working/MAPLE_compatible/state/heartbeat.log)
  13:40:01 info     status_report              {'state': 'ONLINE', 'loaded': None, 'failures': 0, 'gpu_available': True}
  13:45:01 info     status_report              {'state': 'ONLINE', 'loaded': None, 'failures': 0, 'gpu_available': True}
  13:50:01 info     status_report              {'state': 'ONLINE', 'loaded': None, 'failures': 0, 'gpu_available': True}
  13:55:01 info     status_report              {'state': 'ONLINE', 'loaded': None, 'failures': 0, 'gpu_available': True}
  14:00:02 info     status_report              {'state': 'ONLINE', 'loaded': None, 'failures': 0, 'gpu_available': True}
  14:05:02 info     status_report              {'state': 'ONLINE', 'loaded': None, 'failures': 0, 'gpu_available': True}
```

Note the lines caused by the proxy being stopped: `Auth proxy … listening on [] | loopback only: NO - EXPOSED …` and `Key enforced : NO / not answering` are the **misleading output** described in section 0 item 3. The `ports` section further down (`8088: closed`) is the truth. Use the port-listener check in 7.5.

### 6.2 Individual commands from Windows
**[LIVE-VERIFIED]** from PowerShell 5.1 (real output in section 6.3). The `tunnel status --auth-only` line in this block shows the misleading output of section 0 item 3 when the proxy is stopped.

**[WINDOWS PowerShell]**
```powershell
$S = "/kaggle/working/MAPLE_compatible"
$A = "--maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"

# Heartbeat status (exit code 0 = MAPLE API alive, 2 = not alive)
ssh kaggle2 "python3 -B $S/maple_heartbeat.py status $A"

# JSON status, selected fields
(ssh kaggle2 "python3 -B $S/maple_heartbeat.py status --json $A" | ConvertFrom-Json).maple_api

# Heartbeat log (one JSON event per line) - last lines
ssh kaggle2 "tail -n 20 $S/state/heartbeat.log"

# Latest 5-minute status report
ssh kaggle2 "cat $S/state/status.json"

# Supervisor PID (from its lock) and MAPLE API PID
ssh kaggle2 "cat $S/state/supervisor.lock; echo; cat $S/state/maple_api.pid"
ssh kaggle2 "ps -eo pid,ppid,etime,args | grep -E 'maple|llama-server|cloudflared' | grep -v grep"

# Is port 8000 listening? Are 8081 / 8088 exposed? (only 127.0.0.1 addresses are acceptable)
ssh kaggle2 "ss -ltn" | Select-String ':(8000|8081|8088)\s'

# GPU memory
ssh kaggle2 "LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"

# Proxy status
ssh kaggle2 "python3 -B $S/maple_tunnel.py status --auth-only $A"
```

### 6.3 Expected output
```text

===== heartbeat status (first 6 lines + last line)

=================================================================
[2026-11-20 14:10:11] Uptime: 0 minutes
Model       : none loaded (models load on the first request)
Llama API   : ONLINE
Tunnel      : DISABLED (SSH tunnelling; no public tunnel)
Supervisor  : running (pid 4044)
exit code: 0 (0 = MAPLE API alive, 2 = not alive)

===== heartbeat status --json (selected fields)
state=ONLINE pid=4627 failures=0/3 restarts=0/5 supervisor_running=True lock_pid=4044

===== heartbeat log, last 2 lines
{"ts": "2026-11-20T14:05:02Z", "level": "info", "event": "status_report", "state": "ONLINE", "loaded": null, "failures": 0, "gpu_available": true}
{"ts": "2026-11-20T14:10:02Z", "level": "info", "event": "status_report", "state": "ONLINE", "loaded": null, "failures": 0, "gpu_available": true}

===== latest status report (status.json), selected
written=2026-11-20T14:10:02Z api=ONLINE tunnel=disabled/ssh

===== supervisor lock
{"pid": 4044, "started_utc": "2026-11-20T12:14:57Z"}
===== processes
   1971      50    04:21:39 /usr/local/bin/cloudflared tunnel --url ssh://localhost:2222
   4044       1    01:55:35 python3 -B /kaggle/working/MAPLE_compatible/maple_heartbeat.py run --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir /kaggle/working/MAPLE_compatible/state
   4627    4044    01:43:33 python3 -s -m maple serve --host 127.0.0.1 --port 8000

===== listening TCP ports 8000/8081/8088 (empty line = closed)

LISTEN 0      5          127.0.0.1:8000       0.0.0.0:*          

===== GPU memory
0, 0 MiB, 15360 MiB
1, 0 MiB, 15360 MiB

===== proxy status
Tunnel manager: not running
Mode          : ssh
Tunnel        : DISABLED
Endpoint      : none (use SSH tunnelling)

===== models and roles straight from MAPLE (GET /v1/models on the server)
deepseek  DeepSeek-Coder-V2-Lite MAPLE | Systems Engineering and Algorithm Specialist
gemma     Gemma 4 26B MAPLE | General Software and Code Review Specialist
qwen3     Qwen3 8B Q4_K_M | Strategic Reasoning and Architecture Specialist
fast      Qwen2.5 Coder 1.5B Q4_K_M | Ultra-Fast Triage and Terminal Copilot
auto      automatic specialist routing | 
aliases: {"quant":"qwen3"}

===== hostname-extraction command from the notebook cell (count only, value hidden)
1
```

Healthy `ss` line: `LISTEN … 127.0.0.1:8000`. If you ever see `0.0.0.0:8000`, `*:8000`, `[::]:8000` or any LAN address for 8000, 8081 or 8088, **stop and investigate**; do not use the API until it is fixed.

### 6.4 What the words mean
| Term | Meaning |
|---|---|
| `ONLINE` | `/health` and `/v1/models` both answered correctly. Normal. `ONLINE (busy: model activity)` = a model is loading or generating; still fine. |
| `DEGRADED` | The API process answers but `/health` or the model list is not OK. |
| `UNHEALTHY` | A check failed. Recovery starts **at the next check** if the MAPLE process has exited, or after 3 consecutive failed checks if the process is still there but not answering. In watch-only mode it stays here and nothing is done. |
| `RECOVERING` | The supervisor is restarting MAPLE (SIGTERM, wait, start). |
| `OFFLINE` | What the one-shot `status` command reports when the API does not answer. Inside a running supervisor a vanished process goes `UNHEALTHY`, then `RECOVERING`. |
| `BUSY` | Shown as `ONLINE (busy: model activity)`: `/health` was slow within 180 s of a model load/switch/request. Not a failure. Designed and unit-tested; not seen live. |
| `GAVE_UP` / `BLOCKED` | Recovery was abandoned (restart limit reached) or blocked by a safety check (for example the port is held by something that is not MAPLE). The supervisor stops acting and reports why; read `heartbeat.log`. |
| `UNKNOWN` | Before the first check. A known defect: the *first* report after start can still say `UNKNOWN` for a few seconds (not fixed). |
| `Supervisor : not running` | The **supervisor** is not running (nobody is watching MAPLE). This is a fact about the supervisor, not an API state. |
| `WATCH-ONLY` | The supervisor was started with `--no-start`: observe only, never touch MAPLE. |
| `failures n/3` | Consecutive failed checks of a *running but not answering* API; reaching 3 triggers recovery. Resets to 0 after a success. A process that has **exited** does not wait for the counter: it is recovered at the next check. |
| `restarts n/5 per 3600s` | Recoveries in the last hour; at 5 the supervisor stops restarting and reports it. |
| `Tunnel : DISABLED (SSH tunnelling; no public tunnel)` | Normal. Public Cloudflare mode is not used. |
| `SSH tunnel : cloudflared pid [1971] (protected…)` | Your access tunnel, seen but never managed. |
| `Model : none loaded` | Normal until the first request. |

Normal healthy line in the log: `{"event": "status_report", "state": "ONLINE", "loaded": null, "failures": 0, "gpu_available": true}` every 5 minutes.

---

## 7. Start genuine API-key authentication

### 7.1 What the verified `--auth-only` mode does
Source: `maple_tunnel.py` (`supervisor/`, lines cited). **[UNIT-TESTED]**: 96 dedicated tests, a mutation check (10 of 10 deliberately broken copies caught), a run of the **real OpenAI SDK through a real proxy process to a fake upstream** (18/18), and 242 tests passing on Kaggle's Linux (3 process-launching tests excluded). **[UNVERIFIED]**: it has **not** been run live against the real MAPLE.

- Binds **only `127.0.0.1:8088`** (line 417) and forwards to **`127.0.0.1:8000`**. It refuses a non-loopback MAPLE host.
- Requires `Authorization: Bearer <key>`. Missing, wrong, truncated, over-long or `Basic`-scheme credentials → **HTTP 401**. The comparison uses `hmac.compare_digest` (constant time).
- The `Authorization` header is **removed** before the request is forwarded; MAPLE never sees the key.
- Allows exactly five routes and returns **404** for everything else (for example `/props`, `/healthz`, `/slots`):
  `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/route`.
- Request bodies up to 8 MiB; the upstream timeout is 900 s (long generations); streaming is supported.
- Runs a **startup self-test** (a no-key request must get 401, a keyed request must pass) and refuses to report ready otherwise.
- The key is **never printed, logged or written to a file**; it exists only in the proxy's process environment. The state file records `cloudflare: "not used"`.
- If MAPLE is down, the proxy stays up and answers **502** until the heartbeat brings MAPLE back.
- It is mutually exclusive with `--public`; no Cloudflare is involved.
- The MAPLE API on 8000 itself **still has no authentication**. Anyone with a shell on the Kaggle machine can bypass the proxy. The proxy protects the *forwarded* endpoint only.

Key requirements: 24–256 printable ASCII characters, no spaces. `secrets.token_urlsafe(32)` gives 43 characters.

### 7.2 Two ways to provide the key

| | **Option A: key made on Windows (recommended)** | **Option B: key made in the Kaggle notebook** |
|---|---|---|
| Where the key exists | Your PowerShell window (`$env:MAPLE_API_KEY`), and the proxy's environment | The notebook kernel, and the proxy's environment |
| How it reaches the proxy | Piped over SSH stdin (`--key-stdin`): never in command arguments, shell history, logs or notebook output | Inherited from the kernel (`API_KEY`) |
| Can your Windows client use it? | **Yes**: it is already in your PowerShell window | **No**, not without printing or copying it out of Kaggle, which you must not do |
| Good for | Normal use: SDK / scripts on your PC | Only clients that run inside Kaggle |

**Recommendation: Option A.** Option B is provided for completeness, but it cannot conveniently serve a Windows client.

**Why a notebook-generated key cannot conveniently be used on Windows.**
- The key is created inside the notebook kernel and handed to the proxy through its environment. By design the scripts **never print, log or save it**, and nothing sends it back to you. After the cell finishes, the only copies are in kernel memory and in the proxy's process environment.
- Your Windows client needs the same value. To get it there you must move it **out of Kaggle**, and every available route is exactly what this guide forbids or discourages: printing it in a cell (notebook outputs are saved with the notebook and published if you share or "Save & Run" it; the value also appears on screen and in screenshots), writing it to a file (which can end up in a dataset, a version snapshot or Git), or building your own secure channel for it (not provided, not tested here).
- A different SSH session cannot simply read it either: environment variables of the notebook kernel are not part of your SSH shell's environment.
- Option A avoids the problem by making the key on the machine that needs it and sending it **one way**, Windows → Kaggle, over the encrypted SSH stdin. Nothing has to come back.
- Same-user caveat for both options: any process running as the same user on the Kaggle machine can read the proxy's environment through `/proc`. The key protects the forwarded endpoint, not the machine itself.

### 7.3 Option A: generate on Windows, send through SSH stdin (recommended)

**Step 1 [WINDOWS PowerShell]** create the key in this window. Nothing is printed. **[LIVE-VERIFIED]**: the same generator ran on your PC and produced 43 URL-safe characters that the deployed `load_auth_only_key` accepted.

```powershell
# Create a real random API key in THIS PowerShell window only. Nothing is printed or saved.
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:MAPLE_API_KEY = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
"Key created for this window only: $($env:MAPLE_API_KEY.Length) characters (value not shown)."
```

Keep this window (or any window where you run the client). The variable lives only in this window's process; closing the window destroys it. **Do not use `setx`, do not write the key to a file, and do not paste it anywhere.**

**Step 2 [WINDOWS PowerShell]** start the proxy on Kaggle, with the key sent as the first line of SSH stdin:

```powershell
$S = "/kaggle/working/MAPLE_compatible"
$env:MAPLE_API_KEY | ssh kaggle2 "python3 -B $S/maple_tunnel.py start --key-stdin --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"
```

Expected output (from the script's print format; **not captured live**):
```text
auth proxy running: pid 5012 | listening 127.0.0.1:8088 | forwards to 127.0.0.1:8000 | key enforced (self-test passed) | Cloudflare not used
console log: /kaggle/working/MAPLE_compatible/state/auth_proxy_console.log (contains no key)
```
**Verification of Step 2, by part** (the whole command is **[UNVERIFIED]** as a live start): the key transport is **[LIVE-VERIFIED]** (below); the `start` logic (argument parsing, key validation, lock check, detached launch, readiness wait, exit codes) is **[UNIT-TESTED]**; the real launch of the proxy on Kaggle was not run.

Exit codes of `start`: **0** running and self-tested; **3** a proxy manager is already running (nothing changed); **5** it did not become ready within `--wait` (default 20 s) or exited: the message includes the last console lines. A key that is too short or contains spaces is rejected before anything starts.

**Is this pipeline really safe and does it work? What is verified:**
- **[LIVE-VERIFIED]** (validation only): the Windows PowerShell 5.1 → `ssh kaggle2` → `python3` pipe delivers the key intact. On 2026-11-20 a fresh random key was piped through the real SSH connection into the *deployed* `maple_tunnel.py` on Kaggle, and its real `load_auth_only_key` **accepted** it (length 43; the SSH hop added a CRLF line ending, which is stripped), while a too-short key was **rejected**. That test only validated the key; it started nothing and printed nothing, and the key never appeared in any output.
- **[UNIT-TESTED]**: the script reads the key with `readline()` from stdin and passes it to the proxy child **only through its environment** (the child's stdin is `/dev/null`, its output goes to a log that never contains the key).
- **[UNVERIFIED]**: the complete `start` launching the real proxy on Kaggle, and whether the `ssh` call returns promptly after the detached launch. Starting the proxy needs your separate approval.

**If the pipe misbehaves:** do **not** work around it by putting the key in a command line, a file or a printed output. Note that `maple_tunnel.py run` has no `--key-stdin` option, so a foreground run cannot receive the key from stdin either. The only other supported path is Option B (7.4), whose launch is **[MOCKED]**/**[UNVERIFIED]** in the same way. Report the exact error text instead of improvising.

### 7.4 Option B: generate inside the Kaggle notebook (inherits `API_KEY`)

Use only if the client runs on Kaggle. The key exists only in this kernel and the proxy it starts. It is never printed.

> **Start the heartbeat first (section 4).** The heartbeat cell removes `API_KEY` from its own environment, but if you create a key in this kernel first and *then* start other things from the same kernel, they inherit it.

**[KAGGLE NOTEBOOK]**, **Cell 3: Start proxy with a notebook-made key (Option B only).**
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


# ---------------------------------------------------------------- start the authentication proxy with a key made in THIS notebook (Option B)
# The key exists only in this kernel and in the proxy process it starts. It is never printed or written anywhere.
import secrets

if os.environ.get("API_KEY"):
    print("Using the API_KEY already present in this kernel (value not shown).")
else:
    os.environ["API_KEY"] = secrets.token_urlsafe(32)                   # 43 characters, cryptographically random
    print("API key generated and kept in this kernel's environment (value not shown).")
r = subprocess.run(["python3", "-B", str(D / "maple_tunnel.py"), "start"] + ARGS, env=os.environ,
                   capture_output=True, text=True, timeout=120)        # `start` reads API_KEY from the environment it inherits
print(r.stdout.strip() or "(no output)")
if r.stderr.strip():
    print(r.stderr.strip())
print("exit code:", r.returncode, "(0 = running and self-tested, 3 = already running, 5 = did not become ready)")
```
**[MOCKED]**: the child process environment contains `API_KEY`; the key never appears in the cell output; the command line contains no key. The real launch is **[UNVERIFIED]**.

**Never** add `print(os.environ["API_KEY"])`, `!echo $API_KEY`, `%env`, or put the key in a cell *output*: Kaggle notebooks save outputs, and **saving or sharing the notebook would publish the key**.

### 7.5 Proxy status, and the safer port-listener verification

**Do not trust the `Auth proxy …` line of `status --auth-only` when the proxy may be stopped** (section 0 item 3). Verify with these keyless, read-only checks instead; they need no API key and send none.

**[WINDOWS PowerShell]** **[LIVE-VERIFIED]** in the *stopped* state (output below); the *running* output is **[UNVERIFIED]**:
```powershell
# 1. Who listens on 8088 (and on what address)? Only 127.0.0.1 is acceptable.
ssh kaggle2 "ss -ltn" | Select-String ':(8000|8081|8088)\s'

# 2. Does the proxy answer, and does it enforce a key? (no key sent)
ssh kaggle2 "curl -s -o /dev/null -m 5 -w 'proxy 8088 keyless request -> HTTP %{http_code}\n' http://127.0.0.1:8088/v1/models"
```

| Situation | `ss` shows for 8088 | Keyless request |
|---|---|---|
| Proxy **stopped** (verified live 2026-11-20) | no `8088` line | `HTTP 000` (nothing answered) |
| Proxy **running and enforcing the key** | `LISTEN … 127.0.0.1:8088` | `HTTP 401` |
| **Problem:** `8088` on `0.0.0.0`, `*`, `[::]` or a LAN address | stop and investigate | any |
| **Problem:** running but the keyless request returns `200` | stop the proxy immediately and report | 200 |

The full status command, for the other fields (manager pid, MAPLE reachability):
```powershell
$S = "/kaggle/working/MAPLE_compatible"
ssh kaggle2 "python3 -B $S/maple_tunnel.py status --auth-only --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"
```
**[LIVE-VERIFIED]** in the stopped state (with the misleading lines). Expected when running, **[UNIT-TESTED]** only: `Mode : auth-only`, `Auth proxy : 127.0.0.1:8088 | listening on ['127.0.0.1'] | loopback only: yes`, `Key enforced : yes (request without a key -> HTTP 401)`, `MAPLE API : HTTP 200`. Add `--json` for machine-readable output.

### 7.6 Stop only the proxy
```powershell
ssh kaggle2 "python3 -B $S/maple_tunnel.py stop --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"
```
**[UNIT-TESTED]**, not run live. It sends SIGTERM to the proxy **manager** only, waits up to 30 s, then prints `tunnel manager stopped` and `port 8088 listening: False`. It does not touch MAPLE, llama-server, the heartbeat or pid 1971. If nothing runs it prints `no running tunnel manager found (nothing was signalled)` and exits 0. If the proxy has not exited after 30 s it says so and does **not** force it (exit 1).

### 7.7 Key rotation and restarts
- The key is **not stored anywhere**. Restarting the proxy needs the key again.
- To rotate: stop the proxy (7.6) → create a new key (7.3 step 1) → start again (7.3 step 2). Old clients then get **401** until they use the new key. There is no grace period and no second key.
- A Kaggle session restart wipes the proxy and the key: make a new key.
- If a key may have been exposed, rotate immediately (section 13).
- The key is one shared secret with **no rate limiting**.

---

## 8. Start Windows SSH forwarding

### 8.1 The command
**[WINDOWS PowerShell]** (window 1):

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 8088:127.0.0.1:8088 kaggle2
```

It prints nothing when it works and **the window stays busy: leave it open**. The command shape is **[LIVE-VERIFIED]** for port 8000 in Stages 5 and 6a; the forward to **8088** itself is **[UNVERIFIED]** (the proxy has not been started live).

- `-N`: forward only, no shell. `ExitOnForwardFailure=yes`: if your PC's port 8088 is already taken the command **fails immediately** instead of pretending to work. `ServerAliveInterval=20`: detects a dead link within about a minute.
- `-L 8088:127.0.0.1:8088`: *your PC's* 127.0.0.1:8088 → *Kaggle's* 127.0.0.1:8088, i.e. the authentication proxy.

### 8.2 Why 8088, not 8000
Port 8000 is MAPLE itself and has **no authentication**. If you forwarded it, every request would be accepted whatever key it carries. Forward 8088 so all traffic must pass the key check. **Never forward 8081 or 8000.**

### 8.3 Why it is not a public endpoint
`-L` binds to your PC's **loopback** (127.0.0.1) by default, so only programs on your PC can use it, and the data travels inside your authenticated SSH session. Nothing is published on the internet. Do not add `-g`, and do not write `-L 0.0.0.0:8088:…`.

### 8.4 A second window for an interactive Kaggle shell
The forward window is busy, so open a **new** PowerShell window for anything else:

```powershell
ssh kaggle2
```
(Or run single commands: `ssh kaggle2 "python3 -B …"`.) Each ssh command makes its own connection; they do not interfere with the forward.

### 8.5 Stopping only the forward
- **Preferred:** click the forward window and press **Ctrl+C**. That ends only that `ssh.exe`.
- **If it runs in the background** and you must find it, use this script. It **lists every ssh.exe first** and, **only with `-Stop`**, stops only processes that have `-N` and `-L 8088:127.0.0.1:8088` in their command line. It never matches your interactive `ssh kaggle2`, an 8000 forward, or `cloudflared.exe`.

**[WINDOWS PowerShell]** **[MOCKED]** on synthetic process lists (picks only the intended forward; rejects an interactive session, an 8000 forward, a forward without `-N`, and `18088`); the dry run (no `-Stop`) was also run for real on your PC and found no ssh processes. The `-Stop` action itself is **[UNVERIFIED]**:

```powershell
# List (and optionally stop) ONLY the Windows port-forwarding tunnel "ssh -N ... -L 8088:127.0.0.1:8088 ...".
# Without -Stop this is a dry run: it changes nothing. It never touches an ssh.exe that has no -L forward (your interactive
# "ssh kaggle2" session) and never touches cloudflared.exe.
param([switch]$Stop, [int]$Port = 8088)

function Select-Forward {
    param($Procs, [int]$Port)
    $want = [regex]::Escape("-L ${Port}:127.0.0.1:${Port}")
    @($Procs | Where-Object { $_.CommandLine -match '\s-N(\s|$)' -and $_.CommandLine -match $want })
}

$ssh = @(Get-CimInstance Win32_Process -Filter "Name='ssh.exe'")
"All ssh.exe processes on this PC:"
foreach ($p in $ssh) { "  PID {0,-6} {1}" -f $p.ProcessId, $p.CommandLine }
if (-not $ssh) { "  (none)" }

$fwd = Select-Forward $ssh $Port
if (-not $fwd) { "`nNo forwarding tunnel for port $Port found. Nothing to stop."; return }
"`nForwarding tunnel(s) for port ${Port}: PID " + (($fwd | ForEach-Object { $_.ProcessId }) -join ", ")
$owner = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($owner) { "Port $Port is listened on by PID $($owner.OwningProcess)" }
if (-not $Stop) { "Dry run only. To stop exactly these: re-run with -Stop (or press Ctrl+C in the window that runs the tunnel)."; return }
foreach ($p in $fwd) { Stop-Process -Id $p.ProcessId; "Stopped ssh.exe PID $($p.ProcessId)" }
Start-Sleep -Seconds 1
"Port $Port listening now: " + [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
```
Dry run: `.\win_stop_forward.ps1`. Stop: `.\win_stop_forward.ps1 -Stop`.

**To avoid killing the main SSH session:** never use `Stop-Process -Name ssh` or `taskkill /IM ssh.exe`; they kill *every* ssh. Stop by exact PID only, and check the command line first.

### 8.6 Never touch pid 1971
Pid 1971 is **on Kaggle**, not on Windows. Stopping a Windows forward can never require touching it.

---

## 9. Verify the authenticated API

Base URL on Windows: `http://127.0.0.1:8088/v1`. Endpoints: `/health`, `/v1/models`, `/v1/chat/completions`.

Run this in the **same PowerShell window** that holds `$env:MAPLE_API_KEY`, with the forward (section 8) open. The key is used only inside the request header and is never printed.

**[WINDOWS PowerShell]** **[MOCKED]**: tested against a local stub that imitates the proxy's answers (not the real proxy). All 6 checks passed, a wrong key makes every "correct key" line show 401, and with nothing listening it shows status 0 without crashing. Against the real proxy it is **[UNVERIFIED]**.

```powershell
# Check the authenticated MAPLE API from Windows PowerShell 5.1. The key is only used inside the request header; it is never printed.
$base = "http://127.0.0.1:8088"          # the forwarded authentication proxy (NOT 8000)

function Invoke-Maple {
    param([string]$Path, [string]$Key, [string]$Method = "GET", [string]$Body = "", [int]$TimeoutSec = 30)
    $p = @{ Uri = "$base$Path"; Method = $Method; TimeoutSec = $TimeoutSec; UseBasicParsing = $true; Headers = @{} }
    if ($Key)  { $p.Headers["Authorization"] = "Bearer $Key" }
    if ($Body) { $p.Body = $Body; $p.ContentType = "application/json" }
    try {
        $r = Invoke-WebRequest @p
        return [pscustomobject]@{ Status = [int]$r.StatusCode; Body = $r.Content }
    } catch [System.Net.WebException] {
        if ($_.Exception.Response) {
            $text = (New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())).ReadToEnd()
            return [pscustomobject]@{ Status = [int]$_.Exception.Response.StatusCode; Body = $text }
        }
        return [pscustomobject]@{ Status = 0; Body = $_.Exception.Message }      # 0 = no HTTP answer at all (tunnel/proxy down)
    }
}

$wrong = "wrong-" + [guid]::NewGuid().ToString("N")
$r = Invoke-Maple "/health" $env:MAPLE_API_KEY;                  "1. GET  /health            correct key : $($r.Status)   (expect 200)"
$r = Invoke-Maple "/v1/models" $env:MAPLE_API_KEY;               "2. GET  /v1/models         correct key : $($r.Status)   (expect 200)"
if ($r.Status -eq 200) { "   models: " + ((($r.Body | ConvertFrom-Json).data | ForEach-Object { $_.id }) -join ", ") }
$r = Invoke-Maple "/v1/models" "";                               "3. GET  /v1/models         no key      : $($r.Status)   (expect 401)"
$r = Invoke-Maple "/v1/models" $wrong;                           "4. GET  /v1/models         wrong key   : $($r.Status)   (expect 401)"
$r = Invoke-Maple "/props" $env:MAPLE_API_KEY;                   "5. GET  /props             correct key : $($r.Status)   (expect 404: not an approved route)"

$body = @{ model = "fast"; messages = @(@{ role = "user"; content = "Reply with exactly: OK" }); max_tokens = 20; temperature = 0 } | ConvertTo-Json -Depth 5
$r = Invoke-Maple "/v1/chat/completions" $env:MAPLE_API_KEY "POST" $body 600
"6. POST /v1/chat/completions correct key : $($r.Status)   (expect 200; the first request loads the model, allow up to a few minutes)"
if ($r.Status -eq 200) { "   reply: " + (($r.Body | ConvertFrom-Json).choices[0].message.content) } elseif ($r.Body) { "   " + $r.Body.Substring(0, [Math]::Min(200, $r.Body.Length)) }
```

### Short form (two requests, same window)
**[WINDOWS PowerShell]** **[MOCKED]** (run against the same local stub):
```powershell
$h = @{ Authorization = "Bearer $env:MAPLE_API_KEY" }
(Invoke-RestMethod http://127.0.0.1:8088/v1/models -Headers $h).data.id          # model list
try { Invoke-WebRequest http://127.0.0.1:8088/v1/models -UseBasicParsing } catch { "no key -> HTTP " + [int]$_.Exception.Response.StatusCode }   # 401
```

### Expected result
```text
1. GET  /health            correct key : 200   (expect 200)
2. GET  /v1/models         correct key : 200   (expect 200)
   models: deepseek, gemma, qwen3, fast, auto
3. GET  /v1/models         no key      : 401   (expect 401)
4. GET  /v1/models         wrong key   : 401   (expect 401)
5. GET  /props             correct key : 404   (expect 404: not an approved route)
6. POST /v1/chat/completions correct key : 200   (expect 200; the first request loads the model, allow up to a few minutes)
   reply: OK
```
(The live `/v1/models` list comes from MAPLE and is the one in section 5.) Line 6 uses `fast`, the smallest model, with `max_tokens=20`. It does load a model on the GPU.

| Status | Meaning |
|---|---|
| 200 | Accepted and answered |
| 401 | Missing, wrong or malformed key (the proxy's answer; MAPLE never saw it) |
| 404 | Route not on the five-route allow list, or an unknown route |
| 411 | A POST without `Content-Length` |
| 413 | Body larger than 8 MiB |
| 502 | The proxy works, but MAPLE on 8000 did not answer (heartbeat should recover it) |
| 0 (PowerShell) / "cannot connect" | No HTTP answer at all: the forward, the proxy or the Kaggle session is down |

### Diagnosing a failure, in this order
| Symptom | Check | Fix |
|---|---|---|
| Status 0 / connection refused on 127.0.0.1:8088 | Is the forward window still open? `Get-NetTCPConnection -LocalPort 8088 -State Listen` | Restart the forward (8.1). |
| Forward starts but status stays 0, or `channel … open failed: connect failed` in the forward window | Is the proxy running? The port-listener check in 7.5 (`ss` shows `127.0.0.1:8088`; keyless request gives 401) | Start the proxy (7.3). |
| `ssh` itself cannot connect / `bad handshake` | Kaggle session expired or the hostname changed | Section 12.1 / 12.2. |
| 401 on every call including "correct key" | Proxy was restarted with a different key, or `$env:MAPLE_API_KEY` is empty/another key | Re-check `$env:MAPLE_API_KEY.Length` (43); restart the proxy with this window's key (7.3). |
| 502 | MAPLE down | Section 6: `heartbeat status`. If the heartbeat is stopped, start it (section 4); if it is recovering, wait. |
| 200 for `/health` but chat times out | A model is loading, or the generation is long | Wait; raise the timeout (section 10). |
| Everything works through 8000 but not 8088 | You are not using the proxy | That is the unauthenticated path; go back to 8088. |

---

## 10. Python OpenAI SDK

### 10.1 Install (once)
**[WINDOWS PowerShell]**
```powershell
python -m pip install openai
```
Tested with `openai 2.24.0` (validation record B's SDK test and the checks behind this guide). Other 1.x/2.x versions normally work, not verified.

### 10.2 The example
It reads the key **only** from the environment (`MAPLE_API_KEY`), sets `base_url` to `http://127.0.0.1:8088/v1`, lists the models first (this also proves the key), refuses an unknown model name before sending, sends one chat request, prints the serving model (`X-MAPLE-Model`) and the route, and never prints the key. Save as `maple_client.py`.

**[WINDOWS PowerShell]** **[MOCKED]**: tested with the **real OpenAI SDK** against a local stub (success, wrong key, no key, unknown model, `quant` alias, nothing listening, timeout, 502 all behave as written). The real SDK was also run through the real proxy process against a fake upstream (18/18, **[UNIT-TESTED]**). Against the real MAPLE it is **[UNVERIFIED]**.

```python
"""MAPLE client through the authenticated proxy (OpenAI SDK). The key is read from MAPLE_API_KEY and is never printed.

    pip install openai
    $env:MAPLE_API_KEY is already set in this PowerShell window (see the key step), then:  python maple_client.py "your question"
Optional environment: MAPLE_BASE_URL (default http://127.0.0.1:8088/v1), MAPLE_MODEL (default fast), MAPLE_TIMEOUT seconds (default 600).
"""
import os
import sys

import httpx
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, NotFoundError, OpenAI)

BASE_URL = os.environ.get("MAPLE_BASE_URL", "http://127.0.0.1:8088/v1")
MODEL = os.environ.get("MAPLE_MODEL", "fast")            # deepseek | gemma | qwen3 | quant (= qwen3) | fast | auto
TIMEOUT = float(os.environ.get("MAPLE_TIMEOUT", "600"))  # the first request for a model includes loading it

key = os.environ.get("MAPLE_API_KEY")
if not key:
    sys.exit("MAPLE_API_KEY is not set in this window. Create it first (see the guide), then run this script from the same window.")

client = OpenAI(base_url=BASE_URL, api_key=key, timeout=httpx.Timeout(TIMEOUT, connect=10.0), max_retries=0)
prompt = " ".join(sys.argv[1:]) or "Reply with exactly: OK"

try:
    ids = [m.id for m in client.models.list().data]                 # also proves that the key is accepted
    print("models offered:", ids, "(alias: quant -> qwen3)")
    if MODEL != "quant" and MODEL not in ids:
        sys.exit("model %r is not offered by this server; choose one of %s" % (MODEL, ids))
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=512, temperature=0.2)
    reply = raw.parse()
    choice = reply.choices[0]
    print("served by :", raw.headers.get("X-MAPLE-Model"), "| route:", raw.headers.get("X-MAPLE-Route"))
    print("finish    :", choice.finish_reason, "(\"length\" = max_tokens was used up before the answer finished)")
    print("reply     :", (choice.message.content or "").strip() or "(empty - raise max_tokens; reasoning models spend tokens on thinking)")
except AuthenticationError:
    sys.exit("HTTP 401: the API key is missing or wrong. The proxy was probably restarted with a different key; create/send a key again.")
except NotFoundError as e:
    sys.exit("HTTP 404: unknown model name or a path the proxy does not allow (%s)" % getattr(e, "message", e))
except APITimeoutError:
    sys.exit("Timed out after %.0f s. A model may still be loading or the generation is long; raise MAPLE_TIMEOUT." % TIMEOUT)
except APIConnectionError:
    sys.exit("Cannot connect to %s. Check, in this order: the SSH forwarding window is open, the proxy is running, the Kaggle session is alive." % BASE_URL)
except APIStatusError as e:
    sys.exit("HTTP %s from the server: %s%s" % (e.status_code, getattr(e, "message", e),
             "  (502 = the proxy works but MAPLE on port 8000 is not answering; the heartbeat normally brings it back)" if e.status_code == 502 else ""))
```

Run it (same window as the key). **[MOCKED]** as above:
```powershell
python maple_client.py "Hello"
python maple_client.py "Explain a mutex in two sentences."
$env:MAPLE_MODEL = "qwen3"; $env:MAPLE_TIMEOUT = "900"
python maple_client.py "Design a rate limiter."
```

Minimal form (the same call as the script, without the error handling; **[UNVERIFIED]** against the real MAPLE):
```python
import os
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key=os.environ["MAPLE_API_KEY"], timeout=600)
r = client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Hello"}], max_tokens=64)
print(r.choices[0].message.content)
```

### 10.3 Choosing the model name
Use the exact ids shown by `/v1/models`: `fast`, `gemma`, `deepseek`, `qwen3`, or `auto`. **`quant` is accepted as an alias for `qwen3`** (the server resolves it; it is not listed as its own model). With `auto`, MAPLE picks a specialist and tells you which in the `X-MAPLE-Model` / `X-MAPLE-Route` response headers (`with_raw_response` shows them, as in the example). An unknown name is an error, not a fallback. Because only one model is resident, alternating between models costs a load each time; keep a session on one model when you can.

### 10.4 Loading time and timeouts
- Observed model switches took about 2–7 s with warm disk cache (Stage 5: 2.5 s, 6.2 s, 4.4 s including a short answer).
- A cold start or a first load after a session restart can be slower; the heartbeat itself allows 600 s for MAPLE to start.
- **Recommended client timeout: 600 s** (`MAPLE_TIMEOUT`), and 900 s or more for long generations. The proxy itself waits up to 900 s for MAPLE.
- Set `max_retries=0` while debugging so errors are not hidden by silent retries (the example does).
- `qwen3` can use many tokens thinking: give it `max_tokens` 1024 or more, otherwise `content` can be empty with `finish_reason: length`.
- Do not send several requests that need *different* models at once; they will queue behind model switches.

### 10.5 Error handling
The example maps the SDK's exceptions: `AuthenticationError` = 401, `NotFoundError` = 404, `APITimeoutError`, `APIConnectionError` (forward/proxy/session down), `APIStatusError` (for example 502).

---

## 11. Daily shutdown

### 11.1 Recommended order
1. **Stop the Windows forward**: click its window, **Ctrl+C**. (Or use the guarded script in 8.5.)
2. **Stop the authentication proxy** (7.6). After this nothing can reach MAPLE from your PC, even if you re-open the forward.
3. **If you only want to disconnect, stop here.** The heartbeat and MAPLE keep running (and use GPU memory until the Kaggle session ends).
4. **Only for a complete MAPLE shutdown:** run `heartbeat stop --all` (below).
5. Verify.

### 11.2 `stop` versus `stop --all`
| Command | What it does |
|---|---|
| `maple_heartbeat.py stop` | Sends SIGTERM to the **supervisor only** and waits up to 30 s. **MAPLE keeps running, unsupervised**: if it fails later nothing will restart it. |
| `maple_heartbeat.py stop --all` | Stops the supervisor **first**, then sends SIGTERM to the MAPLE API and waits up to 180 s (`--stop-timeout`). MAPLE's own shutdown handler stops `llama-server` and frees the GPUs. Never uses SIGKILL. |

The order matters: if you stop only the MAPLE API while the heartbeat is running, the heartbeat notices that the process is gone at its next check and **starts MAPLE again** (within about a minute; verified in Stage 6a). To stop MAPLE for good, stop the heartbeat first, which is exactly what `stop --all` does.

### 11.3 Commands
**[WINDOWS PowerShell]** **[UNIT-TESTED]**, **[UNVERIFIED]** live:
```powershell
$S = "/kaggle/working/MAPLE_compatible"
$A = "--maple-home /kaggle/working/MAPLE-llama.cpp --state-dir $S/state"

# 2. proxy only
ssh kaggle2 "python3 -B $S/maple_tunnel.py stop $A"

# 4. complete MAPLE shutdown (can take up to ~3.5 minutes; do not interrupt it)
ssh kaggle2 "python3 -B $S/maple_heartbeat.py stop --all $A"
```
Expected (from the script's print formats; **not captured live**, pids illustrative):
```text
SIGTERM sent to auth-only manager pid 5012; it stops only its own auth proxy (MAPLE, llama-server, the heartbeat and SSH access are not touched)
tunnel manager stopped
port 8088 listening: False
...
SIGTERM sent to heartbeat supervisor pid 4044; waiting for it to exit ...
supervisor stopped
MAPLE API pid 4627 stopped (its handler also stops llama-server)
```
Exit codes: `stop --all` returns 0 on success, **1** if the supervisor or the API did not exit in time: it then tells you; it does **not** force anything. In that case wait, re-run the status check, and see section 12. Both stop commands are **[UNIT-TESTED]** and **[UNVERIFIED]** live (a full shutdown was not allowed in this task); the SIGTERM-based recovery path was **[LIVE-VERIFIED]** in Stage 6a. The expected output above is built from the scripts' print statements, not captured.

Equivalent notebook cells (**[KAGGLE NOTEBOOK]**) are **Cell 5** and **Cell 6** in Appendix A.

### 11.4 Verify the shutdown
**[LIVE-VERIFIED]** commands (read-only; run for real, in the running state):
```powershell
ssh kaggle2 "ps -eo pid,ppid,etime,args | grep -E 'maple|llama-server|cloudflared' | grep -v grep"
ssh kaggle2 "ss -ltn" | Select-String ':(8000|8081|8088)\s'
ssh kaggle2 "LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"
```
After a complete shutdown you should see **only** the cloudflared SSH process (pid 1971), no `maple_heartbeat`, no `maple serve`, no `llama-server`, nothing on 8000/8081/8088, and both GPUs at about `0 MiB` used.

### 11.5 Things never to do
- **Never `kill -9` / `pkill -9`** MAPLE, the heartbeat, `llama-server` or the proxy. SIGKILL skips the cleanup that frees GPU memory and can leave an orphan `llama-server` holding the GPUs. It is not a normal shutdown.
- **Never terminate pid 1971** (or any cloudflared) to stop a Windows forward. They are different machines and different processes.
- Never stop the API alone while the heartbeat is running (it restarts it).

---

## 12. Common problems and recovery

The diagnostic commands below reuse the ones tagged **[LIVE-VERIFIED]** in sections 6, 7.5, 8.5 and 11.4. The recovery steps are procedures: they were **not** executed as part of this guide. Section 14 lists the evidence.

For each: **cause → safe diagnostic → recovery → avoid.**

**12.1 Kaggle session expired.** *Cause:* time limit, idle shutdown or quota. *Diagnose:* `ssh kaggle2 "echo ok"` fails; the notebook shows the session as stopped. *Recover:* start a new session (GPU T4 x2, Internet on) → run your SSH-setup cell → update the hostname (12.2) → preflight (section 2.3) → start the heartbeat (4.5) → make a **new** key and start the proxy (7.3) → open the forward (8.1). *Avoid:* assuming the old pids, hostname, key or files still exist.

**12.2 SSH hostname changed.** *Cause:* every new Quick Tunnel has a new name. *Diagnose:* `ssh -v kaggle2` shows `websocket: bad handshake` or a failing ProxyCommand. *Recover:* run the hostname cell (3.3), put the new bare hostname in `HostName` of `Host kaggle2` (3.4), test (3.5). *Avoid:* including `https://`; editing keys/known_hosts.

**12.3 SSH connection refused / timeouts / `Permission denied`.** *Cause:* the Kaggle `sshd` or the cloudflared SSH tunnel is not running; wrong hostname; a changed host key; network down. *Diagnose:* `ssh -v kaggle2`; from the notebook: `ps -eo pid,args | grep -E "sshd|cloudflared"`. *Recover:* re-run your SSH-setup cell; if `cloudflared` is not on PATH run `$env:Path = "C:\cloudflared;" + $env:Path`. *Avoid:* killing or restarting a cloudflared you did not create; changing `~/.ssh/config` options other than `HostName`.

**12.4 Windows port 8088 already in use.** *Cause:* an old forward, or another program. With `ExitOnForwardFailure=yes` the new forward exits with `bind … Address already in use`. *Diagnose:* `Get-NetTCPConnection -LocalPort 8088 -State Listen` then `Get-CimInstance Win32_Process -Filter "ProcessId=<pid>" | Select CommandLine`. *Recover:* if it is your old `ssh -N … -L 8088…`, stop it with Ctrl+C or the guarded script (8.5). If it is something else, **leave it alone** and use another local port, for example `-L 18088:127.0.0.1:8088`, and `MAPLE_BASE_URL=http://127.0.0.1:18088/v1`. *Avoid:* `taskkill /IM ssh.exe`.

**12.5 MAPLE API unavailable.** *Cause:* heartbeat stopped, MAPLE crashed or starting, a model OOM. *Diagnose:* section 6 (`heartbeat status`, `heartbeat.log`, `state/maple_api.log`). *Recover:* if the heartbeat runs, wait: a vanished API process is restarted at the next check, a hung one after 3 failed checks (up to 5 restarts per hour; after that it reports `GAVE_UP` and stops trying: see the log, fix the cause, then restart the heartbeat). If the supervisor is not running, start it (4.5). *Avoid:* starting `python -m maple serve` yourself; `kill -9`.

**12.6 Heartbeat already running.** *Cause:* a supervisor holds `state/supervisor.lock`. *Diagnose:* the start cell says `ALREADY RUNNING`, or `run` exits with code 3. *Recover:* nothing to do; check `status`. *Avoid:* deleting the lock or starting a second one. A stale lock (dead pid) is replaced automatically.

**12.7 Authentication proxy already running.** *Cause:* a manager holds `state/tunnel.lock`. *Diagnose:* `start` prints `not started: a maple_tunnel manager is already running (pid N)` and returns 3. *Recover:* to change the key: `stop`, then `start`. *Avoid:* starting twice.

**12.8 HTTP 401.** *Cause:* missing/incorrect key, or the proxy was restarted with a different key. *Diagnose:* in PowerShell `$env:MAPLE_API_KEY.Length` (43 expected); run the verify script (9). *Recover:* restart the proxy with this window's key (7.3). *Avoid:* printing the key to compare it; pasting it into chat.

**12.9 HTTP 404.** *Cause:* a path outside the five-route allow list (for example `/props`, `/healthz`, a trailing slash, a double slash), or an unknown path. *Diagnose:* compare with the list in 7.1. *Recover:* use `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/route`. Unknown *model names* are reported by MAPLE, not as proxy 404. *Avoid:* "fixing" it by forwarding 8000.

**12.10 HTTP 502 or connection refused.** *Cause:* 502 = proxy up, MAPLE down or restarting. Connection refused on Windows = forward or proxy not running. *Diagnose:* section 9 table, then section 6. *Recover:* let the heartbeat recover MAPLE (a few minutes), or start the missing piece. *Avoid:* restarting the proxy for a MAPLE problem; it does not help.

**12.11 GPU out of memory.** *Cause:* another model/process holds VRAM, an orphan `llama-server`, or a context too large. *Diagnose:* the nvidia-smi command in 6.2; `ps … | grep llama-server`; `state/maple_api.log` (`load_failed` with an OOM). *Recover:* wait for MAPLE to unload; use a smaller context or a smaller model; if a clean `stop --all` leaves an orphan `llama-server`, see `docs/RECOVERY_GUIDE.md` and `docs/TROUBLESHOOTING.md` (do **not** use SIGKILL, and do not touch pid 1971). *Avoid:* running other GPU jobs in the same session; DeepSeek at 131 072 context (not validated).

**12.12 Model loading takes too long.** *Cause:* cold disk, a big switch, or a very long prompt. *Diagnose:* `heartbeat status` (`Model:` and `ONLINE (busy: model activity)`), `tail state/maple_api.log`. *Recover:* wait; raise the client timeout (10.4). By design the heartbeat does not count a slow `/health` right after model activity as a failure (`BUSY`, 180 s grace; not observed live) and allows 600 s for a start. *Avoid:* retrying in a loop (each retry can queue another load); restarting MAPLE.

**12.13 Proxy stopped but the heartbeat still running.** *Cause:* expected after step 2 of a daily shutdown, or the proxy crashed. *Diagnose:* the port-listener check in 7.5 (`ss` shows no 8088; the keyless request gives `HTTP 000`). Do not rely on the `EXPOSED` text of `status --auth-only`. *Recover:* start the proxy again (7.3); MAPLE is unaffected. *Avoid:* assuming MAPLE is down.

**12.14 Windows SSH forward disconnected.** *Cause:* network change, laptop sleep, the Kaggle session died, window closed. *Diagnose:* the forward window shows `client_loop: send disconnect` or has closed; `ssh kaggle2 "echo ok"`. *Recover:* if SSH works, re-open the forward (8.1); if not, 12.1/12.2. The proxy and MAPLE are unaffected. *Avoid:* running the forward in a window you will close.

**12.15 Missing model files.** *Cause:* a new machine without restored models. *Diagnose:* the preflight `PROBLEM model …` line names the exact path; `/health` returns 503 `degraded` with a `missing` list. *Recover:* follow `docs/RESTORATION.md` / `docs/KAGGLE_DEPLOYMENT.md`; models are external and verified by SHA-256; do not substitute another file. *Avoid:* downloading blindly, renaming files, replacing a model with a different one.

**12.16 Stale lock files.** *Cause:* a crashed supervisor or proxy leaves `supervisor.lock` / `tunnel.lock`. *Diagnose:* the lock's pid is not alive, or belongs to another program (`ps -p <pid> -o args=`). *Recover:* nothing: both scripts treat such a lock as stale, never signal the unrelated pid, and take over on the next start. *Avoid:* deleting lock files by hand while a supervisor might be alive.

---

## 13. Security requirements

1. MAPLE (8000), llama-server (8081) and the proxy (8088) must stay bound to **127.0.0.1**. Check with the `ss` command; any other address is a stop-and-investigate event.
2. **Never expose 8000 or 8081**, publicly or through a forward. Never use `--public`, a public Cloudflare tunnel for the API, or a forward that publishes the port.
3. **Never publish the API key**: not in Git, chat, screenshots, notebook outputs, issue reports, shell history or logs.
4. **Never print the key in a Kaggle notebook.** Saved notebook outputs are part of the notebook.
5. **Never commit `.env` files or any file containing credentials.** Both scripts and their tests were scanned for key material before commit. The key must never be written into any file.
6. Do not put the key in a command argument (`--key …` does not exist, on purpose); use stdin or the environment.
7. If a key may have been exposed (pasted, screenshotted, printed, shared a notebook with it): **rotate now**: stop the proxy, create a new key, start the proxy (7.7). Also close old PowerShell windows that hold the old key.
8. **SSH authentication and API-key authentication are different locks.** SSH proves who may get a shell on the Kaggle machine; the API key proves who may call the model API through the forwarded port. SSH access can bypass the key (it can reach port 8000 directly). Protect your SSH access like the root account it is.
9. **Whoever has both the endpoint URL and the API key can run inference on your GPUs** and burn your Kaggle quota. The endpoint exists only on your PC's loopback while the forward is open, but the key is a shared secret with no per-user identity, no rate limit and no expiry; rotate it as you would a password.
10. The proxy refuses everything outside five routes, so `/props`, slots and admin-style endpoints are never reachable through it.

---

## 14. Verification ledger: what is and is not proven

Four separate lists. An item appears in exactly one of the first four.

> **How the "notebook cells" were run.** Cells 1 and 4 (and the hostname check) were executed on the real Kaggle machine **through SSH** (`ssh kaggle2 'python3 -I -'`), with the same interpreter and files. They were **not** pasted into the Kaggle notebook UI. They use only absolute paths and `/proc`, so they should behave the same, but the notebook kernel itself was not used.

### 14.1 [LIVE-VERIFIED]: executed for real
| Item | Evidence |
|---|---|
| CLI arguments and exit codes of both scripts (`start` takes `--key-stdin`, no mode flag; `run` has no `--key-stdin`) | `--help` of every sub-command plus the source |
| Cell 1 Preflight (read-only) | ran on the live server; output in 2.3 |
| Cell 4 Status (read-only) | ran on the live server; output in 6.1 |
| Windows one-liners of 6.2: heartbeat status (and its exit code), `status --json`, log tail, `status.json`, supervisor lock, `ps`, `ss`, `nvidia-smi`, proxy status, model list | ran from PowerShell 5.1 against the live server; output in 6.3 |
| `Get-Content cell | ssh kaggle2 "python3 -I -"` pipe | ran with Cell 4 |
| Keyless port check and `status --auth-only` **with the proxy stopped** (shows HTTP 000, no listener, and the misleading lines) | ran live on 2026-11-20 |
| Hostname source (`/tmp/tunnel.log`, written by pid 1971) and the hostname-finding check | read-only, value masked |
| Windows SSH config shape, `cloudflared` on PATH | read on your PC, hostname masked |
| Windows key generator, and key delivery PowerShell → SSH → Kaggle stdin | a fresh key was accepted by the deployed `load_auth_only_key` and a short one rejected; nothing was started; nothing printed |
| Heartbeat start command, duplicate-supervisor exit 3, SIGTERM recovery in about 47 s, SSH forward of port 8000 | Stages 3, 4, 5 and 6a of the heartbeat validation |
| Cell 2's "already running" branch | ran against the real lock of pid 4044 |

### 14.2 [UNIT-TESTED]: real code under automated tests, not run on the live server
| Item | Evidence |
|---|---|
| Authentication proxy: 401 on missing/wrong/short/long key and Basic scheme, `hmac.compare_digest`, header removal, five-route allow list (404 otherwise), 127.0.0.1-only bind, 502 when MAPLE is down, startup self-test, no key in logs or state | 96 dedicated tests; mutation check 10/10 caught; real OpenAI SDK through a real proxy process to a fake upstream 18/18 |
| `maple_tunnel.py start` / `stop` / `status` logic and exit codes; stale-lock handling | the same tests |
| `maple_heartbeat.py stop`, `stop --all`, `--no-start` guards, duplicate lock, state machine (including `BUSY` and immediate process-exit recovery) | 87 heartbeat tests |
| Suites | Windows 245/245; Kaggle Linux 242/242 (3 process-launching tests excluded) |

### 14.3 [MOCKED]: the guide's own cells and scripts with a stand-in
| Item | Stand-in used |
|---|---|
| Cell 2 launch branch | process launch replaced; argument list, `cwd`, environment (no `API_KEY`), detach flags checked |
| Cell 3 (notebook-made key), Cell 5, Cell 6 | process launch replaced; argument lists and environment checked |
| Windows verify script and the short two-request form | local stub that imitates the proxy's answers |
| `maple_client.py` (real OpenAI SDK) | same local stub; success, 401, no key, unknown model, alias, timeout, 502, no listener |
| `win_stop_forward.ps1` selection logic | synthetic process list (the dry run on your PC found no ssh processes) |
| PowerShell syntax of every `.ps1`, compile check of every Python block | parser / compiler |

### 14.4 [UNVERIFIED]: never executed in any form
- The authentication proxy running against the **real MAPLE**, and any real request through it.
- `maple_tunnel.py start` launching the real proxy on Kaggle; whether the `ssh` call returns promptly after it.
- `maple_tunnel.py stop` and `maple_heartbeat.py stop` / `stop --all` on the live server.
- The Windows forward to **8088** (the 8000 forward is live-verified).
- A real first launch of the heartbeat from the notebook Cell 2.
- `win_stop_forward.ps1 -Stop`.
- The `BUSY` grace during a real model load (no health check overlapped a load in Stages 5/6a).
- The `status --auth-only` *running-state* output on the live server.
- Section 12 recovery steps (procedures only).
- DeepSeek at 131 072 context (never validated); gemma's empty replies on open-ended prompts (recorded failure in validation record B).

### 14.5 Known defects (not fixed; source unchanged)
- `status --auth-only` reports "EXPOSED" and "Key enforced: NO" when the proxy is stopped (seen live).
- The first heartbeat report after a start can say `UNKNOWN` (Stage 4).

The three validation records stay separate: **A** (old server, 56/56), **B** (kaggle2, 28/29 with the gemma failure kept, plus 10/10 SDK), **C** (supervisor: the stages and tests above). One does not validate another.

---

## Appendix A: Kaggle cells for stopping (run from a notebook)

**[KAGGLE NOTEBOOK]**, **Cell 5** (**[MOCKED]**, **[UNVERIFIED]** live): stop **only the proxy**:
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


# ---------------------------------------------------------------- stop ONLY the authentication proxy
r = subprocess.run(["python3", "-B", str(D / "maple_tunnel.py"), "stop"] + ARGS, capture_output=True, text=True, timeout=120)
print((r.stdout + r.stderr).strip())
```

**[KAGGLE NOTEBOOK]**, **Cell 6** (**[MOCKED]**, **[UNVERIFIED]** live): **full shutdown** (heartbeat, then MAPLE; run Cell 5 first):
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


# ---------------------------------------------------------------- FULL SHUTDOWN: heartbeat first, then MAPLE (and with it llama-server)
# Run the "stop proxy" cell first. This can take up to ~3 minutes: MAPLE finishes in-flight requests before it exits.
r = subprocess.run(["python3", "-B", str(D / "maple_heartbeat.py"), "stop", "--all"] + ARGS, capture_output=True, text=True, timeout=400)
print((r.stdout + r.stderr).strip())
print("exit code:", r.returncode, "(0 = everything stopped; 1 = something did not stop - read the message, do NOT use kill -9)")
```
**[MOCKED]**: tested with the launches replaced by stand-ins (the correct argument lists were produced, no key passed). **[UNVERIFIED]** live.

## Appendix B: file and port reference
| Item | Value |
|---|---|
| MAPLE project | `/kaggle/working/MAPLE-llama.cpp` |
| Supervisor scripts | `/kaggle/working/MAPLE_compatible/maple_heartbeat.py`, `maple_tunnel.py` |
| State directory | `/kaggle/working/MAPLE_compatible/state/` |
| Heartbeat files | `heartbeat.log`, `heartbeat_console.log`, `status.json`, `reports.jsonl`, `supervisor.lock`, `maple_api.pid`, `maple_api.log` |
| Proxy files | `tunnel_state.json`, `tunnel.lock`, `auth_proxy_console.log`, `tunnel_manager.log` |
| Models | `/kaggle/working/models/{deepseek,gemma,quant,fast}/` |
| Ports | MAPLE 8000, llama-server 8081 (internal), proxy 8088, sshd 2222 (all loopback) |
| GPU tools | `LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi` |
| Heartbeat defaults | check every 60 s; report every 5 min; 3 failures; start timeout 600 s; stop timeout 180 s; 5 restarts/hour with backoff 15–300 s |
