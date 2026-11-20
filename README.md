# MAPLE-llama.cpp

**MAPLE is the orchestration and specialist-routing layer. llama.cpp is the native inference backend. The compiled CUDA
distribution is restored, never rebuilt.**

MAPLE puts one OpenAI-compatible API in front of four specialist GGUF models (DeepSeek-Coder, Gemma 4, Qwen3, Qwen2.5-Coder),
routes each request to the right specialist (or lets you pick), loads one model at a time on your GPU(s), and is designed so that a
temporary Kaggle session can be rebuilt from checksummed archives in minutes: no llama.cpp compilation and no model downloads.

> No inference engine is implemented here: `llama-server` from llama.cpp does the inference; Python only orchestrates.
> MAPLE-llama.cpp is an **independent project**, not an official fork, continuation or endorsed version of
> [AmSach/maple](https://github.com/AmSach/maple). See [Project Background and Acknowledgments](#project-background-and-acknowledgments)
> and [Licensing](#licensing-and-attribution).

**You must supply your own GGUF model files.** None are included or downloaded, and the configuration does nothing without them.

## Project Background and Acknowledgments

MAPLE-llama.cpp is an independent multi-model inference and orchestration project.

Its initial research and architectural inspiration were informed by the upstream MAPLE research repository:

https://github.com/AmSach/maple

This project is independently developed and maintained. It is not an official release, continuation, or endorsed fork of the upstream repository.

We acknowledge the original authors for their research, ideas, and applicable upstream contributions.

Any upstream source code retained or adapted is identified appropriately and remains subject to its original licensing and attribution requirements.

The MAPLE-llama.cpp runtime is built around native llama.cpp inference, Python-based orchestration, specialist model routing, and portable deployment infrastructure.

*Provenance review (2026-11-20):* all 29 files of the upstream repository at commit `5403b6f` were compared with this repository. No upstream source
line is retained or adapted; the only identical file is the Apache-2.0 `LICENSE` text, kept unchanged. Details in [NOTICE](NOTICE).

## Quick start (Kaggle, two Tesla T4)

Step-by-step for a free Kaggle account: [docs/KAGGLE_FREE_TIER_DEPLOYMENT.md](docs/KAGGLE_FREE_TIER_DEPLOYMENT.md). Using the API from
your PC: [docs/LOCAL_API_USAGE.md](docs/LOCAL_API_USAGE.md).

Prerequisites kept in persistent storage (Kaggle Datasets): the release archives + `checksums.sha256`, and your four GGUF files
([why](docs/KAGGLE_DEPLOYMENT.md)). Kaggle's `/kaggle/working` does not survive a session restart.

```bash
git clone <this-repo-url> MAPLE-llama.cpp && cd MAPLE-llama.cpp
python3 scripts/link_models.py --source /kaggle/input --apply                      # symlinks to the expected model paths
python3 scripts/restore_kaggle.py --artifacts /kaggle/input/<archive-dataset> --dest "$PWD"   # verify, restore llama.cpp, check CUDA, smoke test
source maple.env && scripts/start_maple.sh                                         # API on http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/health
```
```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer maple-local" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Write a Python regex for extracting email addresses."}]}'
```
Or run `notebooks/MAPLE_Kaggle_Restore.ipynb`. Details: [Kaggle deployment](docs/KAGGLE_DEPLOYMENT.md), [restoration](docs/RESTORATION.md).

## Specialists and routing

| ID | Model | Specialist |
|---|---|---|
| `deepseek` | DeepSeek-Coder-V2-Lite MAPLE | systems engineering, C++/Rust/CUDA, algorithms |
| `gemma` | Gemma 4 26B-A4B MAPLE (IQ2_XXS) | general software, Python, web, refactoring |
| `qwen3` (alias `quant`) | Qwen3 8B Q4_K_M | architecture, reasoning, quantitative work |
| `fast` | Qwen2.5 Coder 1.5B Q4_K_M | git, regex, shell, quick edits |

* **Explicit:** set `"model"` to one of the IDs above; it always wins.
* **Automatic:** `"model": "auto"` (or omit it) uses a transparent keyword scorer (`python -m maple route "..."` shows the decision); with
  no signal it uses `gemma`. It is a heuristic, not a classifier.
* Only one model is resident at a time; switching unloads and loads (a few seconds). Unknown model names are rejected, never silently replaced.

## Why llama.cpp and not Ollama

Direct `llama-server` control over GPU placement, context and KV-cache, plus a portable artifact (a restorable directory of binaries
instead of a service install). See [architecture](docs/ARCHITECTURE.md).

## Tested versus not tested

**Tested** (2026-11-20, Kaggle, 2 x Tesla T4, Ubuntu 24.04, Python 3.13, llama.cpp `ef3848f`):
* Full test suite: **56 tests, all passed**, with `MAPLE_LIVE=1` on the GPU host (real inference on all four models, switching,
  streaming, API over real HTTP, SIGTERM with no orphaned `llama-server`, GPU memory released). An earlier run of the first 41 tests also passed 41/41.
* Documented `curl` and OpenAI Python SDK examples run against a live API: on the Kaggle host (SDK 3.57.0) and **from a Windows PC through an SSH port-forward** (SDK 2.24.0: models, `fast`, `quant` alias, automatic routing, 404 for an unknown model).
* Restoration into a clean directory from the archives: restore steps and a 29-check probe passed (separate from the working install).
* Without GPU or models (any machine): the HTTP tests that need no model run and pass.
* A **second, fresh Kaggle session** (a different host, also 2 x Tesla T4; records kept separate from the first session): restore with the public
  `restore_kaggle.py` from archives uploaded from a PC, models downloaded from their sources, 28/29 local checks and 10/10 Windows-over-SSH-tunnel SDK checks.
  **One check failed and is recorded as a failure:** `gemma` returned an empty reply to one open-ended prompt because its reasoning used all `max_tokens`
  (it answered other prompts normally). Details: [`docs/validation/`](docs/validation/VALIDATION_B_kaggle2_fresh_environment.md);
  first-session history: [`docs/validation/`](docs/validation/VALIDATION_A_previous_kaggle_historical.md). Neither record validates the other.

**Not tested:** hardware other than two Tesla T4; Open WebUI and editor extensions (expected to work because the endpoints they use are implemented);
restoration from a Kaggle Dataset (**BLOCKED**: the public runtime Dataset does not exist yet, so Dataset-based restoration is untested); **128K context for any model except gemma**
(a DeepSeek 131072 attempt did not complete and is recorded as incomplete). Compatibility beyond Kaggle's CUDA 13 image is not claimed.

## Benchmark results

Measured on 2 x T4 (generated from `manifests/models_manifest.json`; f16 KV cache; throughput varies with prompt and load):

| Model | Max validated context | Placement at that context | Generation tok/s (short prompt) | Prompt-processing tok/s (long prompt) |
|---|---|---|---|---|
| `deepseek` | 65,536 | both T4s (layer split) | 93 | 415 |
| `gemma` | 131,072 | one T4 | 48 | 600 |
| `qwen3` | 32,768 (its trained limit is 40,960) | one T4 | 29 | 701 |
| `fast` | 32,768 (= trained limit) | one T4 | 118 | 2,702 |

At 4,096 context: load time 1.5 s (fast), 2.6 s (deepseek), 4.5 s (qwen3), 6.5 s (gemma). "Validated" means the server loaded at that context and
answered a prompt filling 60% of it. Details per run: `manifests/benchmark_results.json`.

## API

OpenAI-compatible subset: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (with streaming), `POST /v1/completions`, plus `POST /v1/route`.
**No authentication**: the default bind is `127.0.0.1`; read [the API guide](docs/OPENAI_API.md) before exposing it.
Use from Python, editors, Open WebUI and other clients: [local clients](docs/LOCAL_CLIENTS.md).

## Hardware and software

Tested on 2 x Tesla T4 (15 GiB each), CUDA 13 runtime, glibc 2.39. The compiled build needs CUDA 13 runtime libraries, cuBLAS and NCCL on the host
(not bundled) and targets compute capability 7.5. If they are missing, restoration reports exactly what and stops. See
[hardware requirements](docs/HARDWARE_REQUIREMENTS.md).

## Python dependencies

The runtime uses only the Python standard library; PyYAML is optional. [Installation](docs/INSTALLATION.md).

## The compiled llama.cpp archive (required) and its requirements

MAPLE does not contain llama.cpp. It restores a **pre-compiled CUDA build** from `MAPLE-llamacpp-cuda.tar.gz`, derived from the original
`downloaded-llama-cpp-full-cuda-t4.tar.gz` (235,142,051 bytes, SHA256 `541483fc880595fe20b02f04a0280eb0024d57c9bd4ac2082e7b04fd619eb5f0`;
llama.cpp commit `ef3848f`, CUDA 13.0, `sm_75`). To use it you need:

* a Linux host with **glibc >= 2.38** (Ubuntu 24.04 on Kaggle qualifies), x86_64;
* an **NVIDIA GPU of the Turing generation (compute capability 7.5)**, such as the Tesla T4; other GPUs are not claimed;
* an NVIDIA driver supporting **CUDA 13** (580.x on the tested image) and the host's **CUDA 13 runtime, cuBLAS and NCCL** libraries (not bundled);
* libstdc++, libssl 3 and libgomp (standard on Ubuntu 24.04).

If any requirement is missing, restoration reports exactly which and stops; it never compiles anything or substitutes another backend.

**Where to get the archives.** They are too large for Git (GitHub rejects files over 100 MB, and the original is 235 MB), so they are not in this
repository. They are intended to be published as assets of a GitHub release, with `checksums.sha256`; **at the time of writing no release has
been published yet**, so for now you need to supply the archives yourself. The original archive also contains 19 small
vocabulary-test `.gguf` files; `MAPLE-llamacpp-cuda.tar.gz` excludes them. No archive contains a model.
Four TAR.GZ archives exist: `MAPLE-llamacpp-cuda.tar.gz` (required), `MAPLE-runtime.tar.gz`, `MAPLE-python-deps.tar.gz`, `MAPLE-complete.tar.gz`.
Checksums and instructions: [`artifacts/README.md`](artifacts/README.md).

## Known limitations

* Validated on **2 x Tesla T4 only**; other GPUs, drivers or Kaggle images may not work.
* **128K context is validated for `gemma` only.** deepseek: 65,536; qwen3 and fast: 32,768. Two T4s are two separate 15 GiB devices, not one 32 GiB pool.
* One model resident at a time; switching reloads (seconds).
* **No API authentication**; do not expose the port publicly.
* Kaggle free-tier GPU availability, quotas and session lengths are controlled by Kaggle; sessions expire and `/kaggle/working` is wiped.
* Automatic routing is a keyword heuristic, not a classifier.
* Not tested: editors/Open WebUI, a brand-new Kaggle session started from nothing, and any non-Kaggle host.

## Recovery after a Kaggle session expires

Start a new session, attach the same two Datasets, and repeat the quick start. Nothing is rebuilt. From a Windows PC with SSH into the session,
`python scripts/push_to_kaggle.py --start` does upload, restore, start and smoke-test in one command. See [RECOVERY_GUIDE.md](RECOVERY_GUIDE.md).

## Troubleshooting

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): exit codes, missing libraries, missing models, thinking-model behaviour.

## Development

```bash
python -m unittest discover -s tests            # no GPU needed; GPU tests skip
```
[CONTRIBUTING](CONTRIBUTING.md), [SECURITY](SECURITY.md), [CHANGELOG](CHANGELOG.md), [release steps](docs/GITHUB_RELEASE.md).

## Licensing and attribution

MAPLE-llama.cpp is licensed under the **Apache License 2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)).
* **Upstream:** [AmSach/maple](https://github.com/AmSach/maple) (Aman Sachan, Apache-2.0), referenced at commit `5403b6f5ec3f904409d968ff6a3ea3a2b22767d1`. Its source is not copied; its Apache-2.0 text is
  reproduced as `LICENSE`, and `NOTICE` records the attribution and what differs (new codebase; Ollama and Bonsai are not carried over).
* **llama.cpp** (MIT) is redistributed in compiled form inside `MAPLE-llamacpp-cuda.tar.gz`, together with its own `LICENSE`.
  NVIDIA CUDA, cuBLAS and NCCL are required on the host and are **not** redistributed.
* **Models:** not included. You are responsible for complying with the license of every model you use (DeepSeek, Gemma, Qwen and the MAPLE
  fine-tunes each have their own terms). MAPLE claims no rights over any model weights.
