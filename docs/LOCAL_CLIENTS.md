# Using MAPLE from local applications

MAPLE speaks the OpenAI chat-completions protocol at `http://<host>:<port>/v1`. Any client that lets you set a custom
base URL and any API key (the key is ignored) can use it. The endpoint must be **reachable from the client's network**.

| Where MAPLE runs | Where the client runs | Base URL |
|---|---|---|
| Kaggle notebook | same notebook / same VM | `http://127.0.0.1:8000/v1` (default, works as is) |
| your own GPU machine | same machine | `http://127.0.0.1:8000/v1` |
| Kaggle notebook | **your computer** | needs a tunnel, see below |

## Kaggle versus your computer

A Kaggle VM is not reachable from the internet. Inside the notebook (a cell, `!curl`, a Python cell) the default
`127.0.0.1:8000` works with no extra setup. To use it from your own computer you need a tunnel into the VM:

1. **SSH port forward (recommended if you already have SSH into the session).** Standard OpenSSH feature:
   `ssh -N -L 8000:127.0.0.1:8000 <your-ssh-host-alias>`, then use `http://127.0.0.1:8000/v1` locally. The API stays bound to
   `127.0.0.1` on Kaggle and is reachable only through your SSH session. *(This is a generic OpenSSH feature that was not
   exercised as part of MAPLE's own tests; MAPLE's tested access paths were direct HTTP inside the session.)*
2. **A public tunnel service** (cloudflared, ngrok, ...). Possible, but the API has **no authentication**: anyone with the URL can
   spend your GPU time. Only use a tunnel that adds access control, never post the URL, and stop it when done.

Do not publish API keys or tunnel credentials in notebooks, commits or issues. Keep them in Kaggle Secrets or your own password manager.

## Python (OpenAI SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="maple-local")
print(client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "git commit message for a typo fix"}])
      .choices[0].message.content)
```
`client.models.list()` returns `deepseek`, `gemma`, `qwen3`, `fast` and `auto`.

## VS Code and other editors

Extensions that support a custom OpenAI-compatible provider (for example Continue, Cline or similar) need only: base URL
`http://127.0.0.1:8000/v1` (with the `/v1`), any API key, and a model name from the list above. Suggested split:
`fast` for completions or quick edits, `deepseek` for systems code, `gemma` for general coding, `qwen3` for design discussions.
Use `auto` if the extension lets you type a model name and you want MAPLE to choose. Two cautions: switching between models
costs a few seconds each time, so an extension that alternates models on every keystroke will feel slow; and extensions that send
very long contexts need a model validated for that length (README table).
*Not tested: these editor extensions were not run against MAPLE; this section follows from the implemented endpoints.*

## Open WebUI

In Open WebUI, Admin Settings, Connections, add an **OpenAI** connection with URL `http://127.0.0.1:8000/v1` (use
`http://host.docker.internal:8000/v1` when Open WebUI runs in Docker on the same machine) and any API key. Open WebUI
reads `GET /v1/models` and calls `/v1/chat/completions`, both of which MAPLE implements. *Not tested against Open WebUI.*
Set a generous max-tokens value for `qwen3` (thinking model).

## Anything else

curl, LangChain/LlamaIndex OpenAI providers, and scripts in other languages work the same way: base URL, any key, model name.
Features MAPLE does not implement (embeddings, images, audio, tool-calling orchestration) will fail with 404 or be ignored
depending on the client.
