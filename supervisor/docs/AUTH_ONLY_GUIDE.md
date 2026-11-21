# --auth-only: a real API key in front of MAPLE (loopback only, no Cloudflare)

Status: implemented and unit-tested (mocks + loopback). Uploaded to the Kaggle host but **not yet activated and not yet tested live** against real MAPLE.

Paths below use the folder the scripts were deployed to on Kaggle, `/kaggle/working/MAPLE_compatible/`; adjust them if you copy `supervisor/` elsewhere.

## What it is
`maple_tunnel.py start [--key-stdin]` (detached; this is the normal way) or `maple_tunnel.py run --auth-only` (foreground, for debugging) starts a small reverse proxy. `start` has no `--auth-only` option because it always starts this proxy:

    Windows OpenAI SDK --(ssh -L 8088)--> 127.0.0.1:8088 [auth proxy] --> 127.0.0.1:8000 [MAPLE API] --> 127.0.0.1:8081 [llama-server, internal]

* Every request needs `Authorization: Bearer <API_KEY>`. Missing / wrong key / wrong scheme -> **HTTP 401**, checked with `hmac.compare_digest`, *before* any path handling.
* The `Authorization` header is **removed** before the request is forwarded; MAPLE never sees the key.
* Allowed: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/route`. Anything else (with the right key) -> 404. `/healthz`, `/props`, port 8081 are not reachable.
* Binds `127.0.0.1:8088` only. Starts no Cloudflare and no other process. `--auth-only` and `--public` cannot be combined.
* Self-test at start-up: the proxy is declared "enforced" only after it answered 401 to no key / wrong key / wrong scheme and accepted the right key. If not, it stops and exits non-zero (never claims protection it does not have).
* The key is read from `$API_KEY` (or the first stdin line with `--key-stdin`), must be 24-256 printable ASCII characters without spaces, is **never generated, printed, logged or written to disk**.

## What it is NOT
* MAPLE itself still has no authentication. Port 8000 on Kaggle stays unauthenticated; whoever has shell access on that machine can use it. Port 8088 is the protected door.
* While your old `ssh -L 8000:...` forward is up on the PC, any local program can still reach MAPLE **without** the key. For key-protected use, forward 8088 and stop the 8000 forward.
* One shared key, no rate limiting, no per-user keys, no TLS (the SSH tunnel provides the encryption). A stolen key is valid until you restart the proxy with a new one.
* Not tested against the real MAPLE/Kaggle yet; the local tests use a fake MAPLE-shaped upstream.

## Where the key comes from (important)
A variable set inside the notebook kernel (`os.environ["API_KEY"] = ...`) exists **only** in that kernel and its child processes. An SSH shell does not have it, and the already-running heartbeat/MAPLE never will. Choose one:

**Option 1 (recommended): key made on Windows, sent over SSH stdin. Nothing is ever printed.**

    $b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    $env:MAPLE_API_KEY = [Convert]::ToBase64String($b).TrimEnd('=').Replace('+','-').Replace('/','_')
    $env:MAPLE_API_KEY | ssh kaggle2 "python3 -B /kaggle/working/MAPLE_compatible/maple_tunnel.py start --key-stdin --maple-home /kaggle/working/MAPLE-llama.cpp --state-dir /kaggle/working/MAPLE_compatible/state"

(The key generator and the acceptance check were run locally. The ssh pipe itself has not been run against kaggle2.)

**Option 2: the key from your notebook cell.** Start the proxy FROM the notebook, so the child process inherits the variable:

    import os, subprocess
    r = subprocess.run(["python3","-B","/kaggle/working/MAPLE_compatible/maple_tunnel.py","start",
                        "--maple-home","/kaggle/working/MAPLE-llama.cpp","--state-dir","/kaggle/working/MAPLE_compatible/state"],
                       env=os.environ, capture_output=True, text=True); print(r.stdout, r.stderr)

Then you must get the same key to Windows yourself. Do **not** `print()` it in the notebook: notebook output is saved with the notebook. In PowerShell, type it into a hidden prompt:

    $s = Read-Host "API key" -AsSecureString
    $p = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
    $env:MAPLE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($p)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($p)

## Kaggle side
All commands are for the existing install; `--maple-home /kaggle/working/MAPLE-llama.cpp --state-dir /kaggle/working/MAPLE_compatible/state` is shared with the heartbeat.

* Start (detached, returns when ready): `python3 -B maple_tunnel.py start --key-stdin ...` (Option 1) or `... start ...` with `$API_KEY` set (Option 2).
* Foreground (debugging): `python3 -B maple_tunnel.py run --auth-only ...`
* Status: `python3 -B maple_tunnel.py status --auth-only ...` (add `--json` for JSON). Shows manager pid, listen address, upstream, whether an unauthenticated request is rejected. Never shows the key.
* Stop: `python3 -B maple_tunnel.py stop ...` -> SIGTERM to the proxy manager only. It stops its own proxy thread; it signals no other process.
* Console log: `/kaggle/working/MAPLE_compatible/state/` (tunnel_manager.log, tunnel_state.json). Contains no key.

Stopping the proxy does not stop MAPLE, llama-server, the heartbeat, or the cloudflared SSH tunnel (pid 1971). The unit tests cover this, including a protected `cloudflared ssh://` process and a leftover `--public` cloudflared.

## Windows side
1. Stop the old unauthenticated forward (identify it first: it is the `ssh.exe` that owns the listener on port 8000 and whose command line contains `-N` and `-L 8000:127.0.0.1:8000`; never stop `ssh -v kaggle2` or its cloudflared helper): only the `ssh -N ... -L 8000:127.0.0.1:8000 kaggle2` process.
2. Open the key-protected forward in its own window:

        ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 8088:127.0.0.1:8088 kaggle2

3. Test (PowerShell):

        $h = @{ Authorization = "Bearer $env:MAPLE_API_KEY" }
        Invoke-RestMethod http://127.0.0.1:8088/health -Headers $h          # 200
        Invoke-WebRequest http://127.0.0.1:8088/health -SkipHttpErrorCheck  # 401 without the key

4. OpenAI SDK:

        import os
        from openai import OpenAI
        client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key=os.environ["MAPLE_API_KEY"], timeout=300)
        print(client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Hello"}], max_tokens=50).choices[0].message.content)

   A wrong key raises `openai.AuthenticationError` (401).

## Rotating the key
`stop`, then `start` with a new key. Old keys stop working immediately.

## Heartbeat
The unmodified heartbeat reads `tunnel_state.json`; its 5-minute report then shows the loopback endpoint with "(Bearer key required)". This is covered by a unit test that feeds the real state file to the real heartbeat code.
