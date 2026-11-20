# OpenAI-compatible API

MAPLE serves a small, honest subset of the OpenAI HTTP API from a standard-library Python server (`maple.api`).
Everything below is implemented in `src/maple/api.py` and exercised over real HTTP by `tests/test_http_api.py`
(see "What is tested" at the end).

## Start the server

```bash
# after restoration (see RESTORATION.md); from the MAPLE project directory
source maple.env                       # sets MAPLE_HOME / MAPLE_LLAMA_BIN / LD_LIBRARY_PATH for this shell only
scripts/start_maple.sh                 # http://127.0.0.1:8000
scripts/start_maple.sh --port 8010     # another port
scripts/start_maple.sh --host 0.0.0.0  # all interfaces: read "Security" first
```

Defaults come from `config/runtime.yaml` (`host`, `api_port`); `--host/--port` override them. The default host is
`127.0.0.1`, i.e. reachable only from the same machine.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` (also `/healthz`; `?deep=1` adds GPU and RAM info) | 200 `{"status":"ok"}` when all four model files are valid; 503 `{"status":"degraded", "missing":[...]}` otherwise. Also reports the loaded model, in-flight requests and the number of model switches. |
| `GET /v1/models` | The four specialists plus `auto`, and the alias map (`{"quant": "qwen3"}`). |
| `POST /v1/chat/completions` | Chat completion. `stream: true` is supported (server-sent events, proxied from llama-server). |
| `POST /v1/completions` | Text completion (`prompt`). |
| `POST /v1/route` | MAPLE extension, not part of OpenAI: shows which specialist a request would use, without loading a model. |

Nothing else is implemented (no embeddings, no images, no audio, no function-calling logic of its own). Other paths return 404.
Each response carries `X-MAPLE-Model` and `X-MAPLE-Route` headers, and non-streaming JSON has an extra `x_maple` object
(`model`, `route`, `explicit`).

## Choosing a specialist

* **Explicit selection.** Set `"model"` to `deepseek`, `gemma`, `qwen3`, `quant` (alias of `qwen3`) or `fast`. An explicit model always wins.
* **Automatic routing.** Set `"model": "auto"` (or omit `model`). A transparent keyword/pattern scorer in `src/maple/router.py`
  picks the highest-scoring specialist; with no signal it uses `default_model` (`gemma`). Inspect a decision with
  `POST /v1/route` or `python -m maple route "your prompt"`. It is a simple heuristic, not a classifier: when it matters, choose explicitly.
* An unknown model name is rejected with 404 `model_not_found`; MAPLE does not silently fall back.

Only one model is resident at a time. A request for a different specialist waits for in-flight requests, unloads the current
`llama-server`, loads the next and continues (a few seconds for these model sizes; see the benchmark table in the README).

## Examples

Verified shapes: the responses follow the OpenAI `choices[0].message.content` structure.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer maple-local" \
  -d '{"model": "fast",
       "messages": [{"role": "user", "content": "Write a Python regex for extracting email addresses."}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="maple-local")

response = client.chat.completions.create(
    model="deepseek",
    messages=[{"role": "user", "content": "Explain lock-free data structures in C++20."}],
)
print(response.choices[0].message.content)
```

The `api_key` is required by the SDK but **ignored by MAPLE** (any value works). Streaming:

```python
for chunk in client.chat.completions.create(model="fast", stream=True,
        messages=[{"role": "user", "content": "Count from 1 to 5."}]):
    print(chunk.choices[0].delta.content or "", end="")
```

The Python example above was run against a live MAPLE API on the Kaggle host with the `openai` SDK (3.57.0): chat completion, `client.models.list()` and streaming all worked. Pure standard library (no packages): `scripts/smoke_api.py` calls the API this way for all four specialists.

### Qwen3 is a thinking model
`qwen3` / `quant` spends tokens on `reasoning_content` before answering. With a small `max_tokens` the visible `content` can be empty.
Leave `max_tokens` unset (the roster default is 4096) or use 1024+.

## Errors

| Status | `error.code` | Meaning |
|---|---|---|
| 400 | (none) | invalid JSON, empty body, missing `messages` (chat) or `prompt` (completions) |
| 404 | `model_not_found` | unknown model; the message lists the valid ones |
| 404 | `not_found` | unknown path |
| 503 | `model_missing` | the model file is not at its configured path; the message contains that exact path |
| 503 | `load_failed` | llama-server could not load the model; the message contains the end of its log |
| 502 | | upstream llama-server error passed through |
| 500 | `server_error` | unexpected failure |

## Security: there is no authentication

MAPLE does **not** check `Authorization`. The default bind (`127.0.0.1`) is safe on its own; anything that exposes the port
(`--host 0.0.0.0`, a tunnel, a port-forward on a shared machine) lets everyone who can reach it use your GPU. Put an
authenticating reverse proxy or an access-controlled tunnel in front before exposing it, and never publish a tunnel URL.
See `docs/LOCAL_CLIENTS.md` for reaching a Kaggle-hosted API from your own computer.

## What is tested

`tests/test_http_api.py` starts the real `python -m maple serve` process and talks to it over sockets.
* **Always run (no GPU, no models):** health 503 with the exact missing paths, model listing and alias, routing (explicit, `quant`, `auto`, default),
  invalid model 404, missing-model 503 naming the path, malformed requests, ignored authorization.
* **`MAPLE_LIVE=1` on the two-T4 Kaggle host with the four models:** real chat and completion calls, streaming, `quant` alias,
  automatic routing, model switching with exactly one `llama-server` resident, invalid model not disturbing the loaded one,
  and SIGTERM leaving no `llama-server` process. Results and the date are in the README.
