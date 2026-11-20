# Architecture

**MAPLE is the orchestration and specialist-routing layer. llama.cpp is the native inference backend. The compiled CUDA
distribution is restored, never rebuilt.** MAPLE implements no inference engine.

```
client (OpenAI SDK / curl / editor)
   |  HTTP  /health /v1/models /v1/chat/completions /v1/completions /v1/route
   v
maple.api        ThreadingHTTPServer (stdlib), OpenAI-style errors, SSE passthrough
   v
maple.router     explicit model wins; "auto"/unset -> keyword scoring; no signal -> default_model
   v
maple.runtime    ModelManager: one llama-server resident; waits for in-flight requests; unload/load; health; log tail on failure
   v
maple.resource_manager   nvidia-smi probing; single GPU if it fits in 85% of free memory, else layer split; disk/RAM guards
   v
llama-server (restored build)   spawned per model with LD_LIBRARY_PATH set for that child only
```

## Modules (`src/maple/`)

| File | Role |
|---|---|
| `config.py` | `MODEL_ROSTER` (IDs, names, roles, paths, sampling defaults), `MODEL_ALIASES` (`quant` -> `qwen3`), runtime defaults, `MAPLE_MODEL_<ID>` path overrides, `build_env()` |
| `registry.py` | path/size/GGUF-magic checks against `manifests/models_manifest.json`; never modifies a model |
| `gguf_meta.py` | read-only GGUF metadata reader (architecture, context length) |
| `router.py` | transparent specialist scoring (`python -m maple route "..."`) |
| `runtime.py` | `ModelManager` process control, health, benchmarks hooks |
| `resource_manager.py` | GPU/RAM/disk probing and placement |
| `api.py` | HTTP server and OpenAI-style responses |
| `__main__.py` | `python -m maple serve / status / route` |

## Why llama.cpp and not Ollama

The upstream project routed through Ollama. This project uses `llama-server` directly because it gives explicit control over
GPU placement, context size, KV-cache type and process lifetime, produces a portable, checksummed artifact (a directory of binaries
that is restored by extraction) instead of a service installation, and exposes an OpenAI-compatible endpoint natively. Bonsai is not used anywhere.

## Design rules

* Models are external, read-only inputs. MAPLE never copies, moves, renames, downloads, quantizes or packages them.
* Never compile; never install or replace CUDA libraries; report incompatibility with the exact reason.
* One model at a time; two T4s are two independent 15 GiB devices, not one 32 GiB device.
* Validated context lengths only; no RoPE scaling is assumed.
* Environment changes are per child process; the system and shell are untouched.
