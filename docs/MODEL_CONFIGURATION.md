# Model configuration

MAPLE does not include, download, convert, quantize or redistribute any model. **You must supply your own GGUF files**
and you are responsible for complying with their licenses. The shipped configuration describes the four files the author
validated; it does nothing useful without them.

## The roster

| ID | Model | Specialist | Expected file (default path) |
|---|---|---|---|
| `deepseek` | DeepSeek-Coder-V2-Lite MAPLE | systems engineering, algorithms | `/kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf` |
| `gemma` | Gemma 4 26B-A4B MAPLE (IQ2_XXS) | general software, code review | `/kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf` |
| `qwen3` (alias `quant`) | Qwen3 8B Q4_K_M | architecture and reasoning | `/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf` |
| `fast` | Qwen2.5 Coder 1.5B Instruct Q4_K_M | fast coding tasks | `/kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` |

The authoritative definition is `MODEL_ROSTER` in `src/maple/config.py` (mirrored in `config/models.yaml`). The IDs are the API
model names. `quant` is accepted everywhere as an alias of `qwen3`. The DeepSeek and Gemma files are the "MAPLE" models
published by the upstream project author (see `NOTICE`); Qwen3 and Qwen2.5-Coder are the vendors' public releases.

## Providing the files

* **Kaggle:** upload the GGUF files once as a private Kaggle **Dataset**, attach it to each notebook, then run
  `python3 scripts/link_models.py --source /kaggle/input` (dry run) and add `--apply` to create symlinks at the paths above.
  The script only creates symlinks, never copies, moves or overwrites.
* **Other paths or your own models:** point MAPLE at them with environment variables, one per model, without editing code:
  `MAPLE_MODEL_DEEPSEEK`, `MAPLE_MODEL_GEMMA`, `MAPLE_MODEL_QWEN3`, `MAPLE_MODEL_FAST` (see `config/example.env`).
  The model *ID* stays the same; only the file location (and what the model really is) changes.

## Validation and checksums

* `python -m maple status` checks that every path exists, has the GGUF magic and (when a size is recorded) the recorded size, without loading anything.
* `manifests/models_manifest.json` records, for the author's four files: size, SHA256, GGUF architecture and the tested context lengths.
  `scripts/restore_kaggle.py` compares each file against it.
* **If you use different files**, those recorded sizes and SHA256 values will not match, by design. Run restoration with `--no-sha`
  and set `MAPLE_STRICT_MANIFEST=0` to skip the size comparison, or regenerate the manifest for your files with
  `python3 scripts/validate_models.py` (it reads GGUF metadata and loads each model once). Context limits recorded for the author's
  files do **not** transfer to yours.
* A real load is the final authority on compatibility: if the bundled llama.cpp build (commit `ef3848f`) does not support a model
  architecture, loading fails and MAPLE reports the llama-server log tail.

## Missing or invalid models

Nothing is substituted. `/health` returns 503 with the exact missing path; a request for that model returns 503 `model_missing`
with the path; restoration stops with exit code 3 and lists each missing or mismatching file.

## Context limits and loading behaviour

* Default context is conservative (`default_ctx` 4096 in `config/runtime.yaml`); raise `models.<id>.ctx` only up to a validated value
  (README benchmark table). Validated on two Tesla T4: gemma 131072, deepseek 65536, qwen3 32768, fast 32768. **128K is validated only for gemma.**
* Models load **sequentially**: one `llama-server` at a time. A model that fits one GPU is placed on one GPU (up to 85% of its free
  memory), otherwise layers are split across both GPUs. The two T4s are treated as two independent 15 GiB devices.
