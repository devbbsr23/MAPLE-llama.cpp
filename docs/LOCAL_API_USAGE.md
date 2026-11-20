# Using the MAPLE API from your Windows PC

MAPLE serves an OpenAI-compatible subset of the HTTP API (details in [OPENAI_API.md](OPENAI_API.md)):
`GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming supported), `POST /v1/completions`, and `POST /v1/route`.
Nothing else is implemented, so clients that need embeddings, images, tools/function calling or other endpoints will not work.

**There is no authentication.** The server ignores the `Authorization` header: any key (or none) is accepted. It binds to
`127.0.0.1` by default. Never expose it on a public address or a public tunnel; anyone who can reach it can use your GPU session.
The `api_key="maple-local"` used below is only a placeholder that SDKs require.

## 1. Start the API on Kaggle

```bash
cd /kaggle/working/MAPLE-llama.cpp && source maple.env
scripts/start_maple.sh                     # 127.0.0.1:8000, from config/runtime.yaml
scripts/start_maple.sh --port 8010         # another port
scripts/start_maple.sh --host 127.0.0.1    # host/port can also be set in config/runtime.yaml
```
Inside the notebook you can call it directly (*tested*):
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

## 2. Reach it from your PC: SSH port forwarding (recommended)

The API is only listening on the notebook's own loopback, so forward a local port to it over an SSH connection to the session. How you get
SSH access to a Kaggle notebook is outside this project (MAPLE ships no SSH or tunnel tooling and stores no credentials). Once you can run
`ssh <your-host-alias>`, add a local forward:

```bash
ssh -N -L 8000:127.0.0.1:8000 <your-host-alias>        # leave this running; Ctrl+C to close
```
If the connection needs a helper such as `cloudflared`, make sure it is on your `PATH` for that command; keep keys and tokens in
`~/.ssh` and never put them in this repository. If local port 8000 is taken, use another local port (`-L 18000:127.0.0.1:8000`) and use it in the URLs below.

*Tested:* this exact forward (`-L <local>:127.0.0.1:<remote>`) from Windows to a live 2 x T4 session; then `curl` and the OpenAI Python
SDK 2.24.0 on the PC returned `/health`, `/v1/models`, chat completions from `fast`, the `quant` alias (served by `qwen3`), automatic routing
(`deepseek`), and HTTP 404 for an unknown model. Tunnels drop; if a request times out, check the `ssh` window and re-run it.

## 3. Check it from the PC

```bash
curl http://127.0.0.1:8000/health            # status "ok" when all four model files are found; 503 lists exactly which are missing
curl http://127.0.0.1:8000/v1/models         # deepseek, gemma, qwen3, fast, auto; aliases: {"quant": "qwen3"}
```

## 4. OpenAI Python SDK examples

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="maple-local", timeout=600)   # a model switch can take several seconds

def ask(model, prompt, max_tokens=512):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    print(model, "-> served by", r.model_extra["x_maple"]["model"])      # MAPLE adds x_maple (model, route, latency)
    print(r.choices[0].message.content)

ask("deepseek", "Review this Python function:\n\ndef add(a, b):\n    return a - b")    # systems / algorithms specialist
ask("gemma",    "Refactor this loop into a list comprehension: out = []\nfor x in xs:\n    out.append(x * 2)")
ask("qwen3",    "Compare a monolith and microservices for a small trading backtester.", max_tokens=1024)   # alias: "quant"
ask("fast",     "Write a Python regex for extracting email addresses.")
ask("auto",     "Explain lock-free data structures in C++20.")        # automatic routing (also what you get if you omit "model")
```
* **Explicit selection** (`deepseek`, `gemma`, `qwen3`, `quant`, `fast`) always wins and uses exactly that model.
* **Automatic routing** (`auto` or no `model`) uses a keyword scorer and falls back to `gemma` when nothing matches. It is a heuristic.
  Preview a decision without generating: `curl -s http://127.0.0.1:8000/v1/route -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Explain lock-free queues"}]}'`.
* Streaming: pass `stream=True`; chunks arrive as server-sent events. Only one model is loaded at a time, so the first request to a
  different model includes a load (1.5 to 6.5 s measured at 4,096 context).
* `qwen3` is a thinking model: with a small `max_tokens` the answer can be empty because the budget went to reasoning.
  Use 512 or more.

*Tested:* the `fast` and `auto` calls above, plus listing and streaming, through the tunnel from Windows; `deepseek` on the Kaggle host.
`gemma` and `qwen3` were verified through the test suite and the specialist smoke test rather than from the PC.

cURL equivalent:
```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer maple-local" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Write a Python regex for extracting email addresses."}], "max_tokens": 256}'
```

## 5. Error responses (OpenAI-style JSON `{"error": {...}}`)

| Status | Meaning |
|---|---|
| 400 | invalid JSON, or `messages` / `prompt` missing |
| 404 | unknown model name (never silently replaced), or unknown path |
| 503 | a model file is missing (the message names the exact path), or the model failed to load (log tail included) |
| 502 | the underlying llama-server returned something unusable |
| 500 | unexpected server error |

## 6. Editors and other applications

Any application that lets you set a custom OpenAI-compatible **base URL** (`http://127.0.0.1:8000/v1`), an arbitrary API key and a model name
can use MAPLE. Use the specialist IDs as model names. **Not tested** from VS Code or Continue; the setting names below follow those
tools' documentation as I know it and may differ in your version.

* **Continue (VS Code)**, `config.yaml`:
```yaml
models:
  - name: MAPLE deepseek
    provider: openai
    model: deepseek
    apiBase: http://127.0.0.1:8000/v1
    apiKey: maple-local
  - name: MAPLE auto
    provider: openai
    model: auto
    apiBase: http://127.0.0.1:8000/v1
    apiKey: maple-local
```
* **Open WebUI / other chat UIs:** add an "OpenAI API" connection with the same base URL and key; model lists come from `/v1/models`.
  Features that need other endpoints (embeddings, image generation, tool calling) are unavailable.
* Because the API is single-model-at-a-time, avoid configuring several tools to hit different specialists simultaneously: they would force
  repeated model switches.

More client notes: [LOCAL_CLIENTS.md](LOCAL_CLIENTS.md). Never commit SSH keys, Kaggle tokens or tunnel URLs.
