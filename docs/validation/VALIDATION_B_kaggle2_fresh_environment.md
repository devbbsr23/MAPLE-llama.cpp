# Validation record B: NEW Kaggle environment `kaggle2` (fresh test, 2026-11-20)

Host 86482ac06d47 (not the host of record A). Nothing was copied from the earlier server.

## Environment (inspected read-only before any change)
Ubuntu 24.04.4, x86_64, kernel 6.18, glibc 2.39, Python 3.13.16. 2x Tesla T4 (15360 MiB each, cc 7.5), driver 580.178.04, CUDA 13.0.
RAM 32 GiB (cgroup 32.2 GB), 4 CPUs, 20 GB disk on /kaggle/working (empty at start). Host libs present: libcudart.so.13, libcublas.so.13,
libcublasLt.so.13, libnccl.so.2, libcuda.so.1. The only /kaggle/input item was an unrelated SSH-key dataset, which was not read.

## How it was deployed
1. `git clone` of https://github.com/devbbsr23/MAPLE-llama.cpp (commit 7f570c0, 70 files).
2. Models: downloaded from their Hugging Face sources on this server only (user-authorised for this test; size + SHA256 matched the manifest first):
   deepseek and gemma from the AmSach/*-MAPLE-Sub5GB-GGUF repos, qwen3 from Qwen/Qwen3-8B-GGUF, fast from Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF.
3. The three archives were **uploaded from the PC over SSH** (SHA256 verified on the PC before and on kaggle2 after) into a flat folder.
   **Kaggle Dataset distribution: BLOCKED / NOT TESTED** (the public dataset does not exist yet; no Kaggle credentials). This record does not validate Dataset-based restoration.
4. The **public, unmodified** scripts/restore_kaggle.py (git blob identical to the repo) was run with `--artifacts <flat dir>`. A later, separate change (deeper archive search) post-dates this test and was not used.
   Dry run first: it writes only inside the project folder, kept 6 differing files unchanged, extracted 83 llama.cpp files, nothing outside the project, no protected path touched.
5. llama.cpp was restored from the archive; **nothing was compiled**. llama-server reports commit ef3848f.

## Restore script result
All steps ok: 3 archive checksums, no GGUF/unsafe paths, 83 llama.cpp files, offline PyYAML 6.0.3, llama-server --version, 2 GPUs / glibc 2.39,
4 model SHA256 matches, smoke test (fast loaded in 1.54 s, reply OK), API health.
After restore, `git status` shows 17 entries that are file-mode changes only (execute bits); no content changes.

## Local validation on kaggle2: 28/29 checks passed
| Result | Check | Detail |
|---|---|---|
| PASS | archive SHA256 (3 archives) on kaggle2 | MAPLE-llamacpp-cuda.tar.gz: OK; MAPLE-runtime.tar.gz: OK; MAPLE-python-deps.tar.gz: OK |
| PASS | restored executables are executable | 47 executables, non-executable: none |
| PASS | shared-library resolution (ldd, 0 unresolved) | unresolved: none |
| PASS | CUDA/cuBLAS/NCCL resolve from host, llama libs from restored dir | {"libcudart.so.13": "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13", "libcublas.so.13": "/usr/local |
| PASS | llama.cpp CUDA backend sees GPUs | [('CUDA0', 'Tesla T4', '14911'), ('CUDA1', 'Tesla T4', '14911')] |
| PASS | llama-server --version (no compile) | version: 1.0.0-dev (build 1, commit ef3848f) / built with GNU 13.3.0 for Linux x86_64 |
| PASS | GPUs idle before start | [(0, 0, 2), (1, 0, 2)] |
| PASS | API started via scripts/start_maple.sh on :8000 | pid 2371 |
| PASS | /health ok, all four model files ok, nothing loaded yet | {"status": "ok", "uptime_s": 0, "models": {"deepseek": "ok", "gemma": "ok", "qwen3": "ok", "fast": "ok"}, "mod |
| PASS | /v1/models lists 4 specialists + auto, alias quant->qwen3 | ['deepseek', 'gemma', 'qwen3', 'fast', 'auto'] {'quant': 'qwen3'} |
| PASS | /v1/route (cpp) | {"model": "deepseek", "reason": "highest specialist score (8) among ['deepseek']", "scores": {"deepseek": 8, " |
| PASS | /v1/route (general) | {"model": "gemma", "reason": "highest specialist score (4) among ['gemma']", "scores": {"deepseek": 0, "gemma" |
| PASS | /v1/route (arch) | {"model": "qwen3", "reason": "highest specialist score (13) among ['qwen3']", "scores": {"deepseek": 0, "gemma |
| PASS | /v1/route (quick) | {"model": "gemma", "reason": "highest specialist score (4) among ['gemma']", "scores": {"deepseek": 0, "gemma" |
| PASS | explicit deepseek -> deepseek | {"served_by": "deepseek", "http": 200, "seconds": 35.9, "completion_tokens": 1024, "gpu_peak_util": {"0": 97,  |
| **FAIL** | explicit gemma    -> gemma | {"served_by": "gemma", "http": 200, "seconds": 40.1, "completion_tokens": 1024, "gpu_peak_util": {"0": 95, "1" |
| PASS | explicit qwen3    -> qwen3 | {"served_by": "qwen3", "http": 200, "seconds": 18.3, "completion_tokens": 235, "gpu_peak_util": {"0": 98, "1": |
| PASS | explicit fast     -> fast | {"served_by": "fast", "http": 200, "seconds": 3.6, "completion_tokens": 32, "gpu_peak_util": {"0": 26, "1": 0} |
| PASS | explicit quant    -> qwen3 | {"served_by": "qwen3", "http": 200, "seconds": 11.6, "completion_tokens": 277, "gpu_peak_util": {"0": 98, "1": |
| PASS | automatic routing (model=auto) | deepseek / highest specialist score (11) among ['deepseek'] |
| PASS | automatic routing (no model field) | deepseek / highest specialist score (11) among ['deepseek'] |
| PASS | model switching (7 requests, exactly 1 llama-server each time) | switches 6->13 trace=[('fast', 'fast', 1), ('deepseek', 'deepseek', 1), ('fast', 'fast', 1), ('gemma', 'gemma' |
| PASS | OpenAI chat.completion response shape | keys=['choices', 'created', 'id', 'model', 'object', 'system_fingerprint', 'timings', 'usage', 'x_maple'] cont |
| PASS | streaming SSE (chunks + [DONE]) | 13 chunks, content-type=text/event-stream |
| PASS | /v1/completions | {"choices": [{"text": "\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b", "ind |
| PASS | unknown model -> 404 (not replaced) | {"error": {"message": "unknown model 'does-not-exist' (valid: deepseek, gemma, qwen3, fast, auto)", "type": "m |
| PASS | invalid JSON -> 400 | 400 |
| PASS | unknown path -> 404 | 404 |
| PASS | final health healthy; API still running | {"loaded": "fast", "inflight": 0, "switches": 14, "healthy": true} |

### Inference (measured, one prompt each; one llama-server resident at a time; every model fit on GPU 0, GPU 1 stayed idle)
| Model | Served by | Time | Output | GPU |
|---|---|---|---|---|
| deepseek | served_by=deepseek | 35.9s | 1024 tokens | GPU0 peak 5971 MiB / 97% util; GPU1 3 MiB |
| gemma | served_by=gemma | 40.1s | 1024 tokens | GPU0 peak 5189 MiB / 95% util; GPU1 3 MiB |
| qwen3 | served_by=qwen3 | 18.3s | 235 tokens | GPU0 peak 5269 MiB / 98% util; GPU1 3 MiB |
| fast | served_by=fast | 3.6s | 32 tokens | GPU0 peak 1243 MiB / 26% util; GPU1 3 MiB |
| quant | served_by=qwen3 | 11.6s | 277 tokens | GPU0 peak 5269 MiB / 98% util; GPU1 3 MiB |

Offload evidence: the live llama-server command line was `-ngl 99 -dev CUDA0`; GPU 0 held 5189 MiB with gemma loaded; GPU 1 stayed at 3 MiB.
The llama-server logs at the configured verbosity contain no "offloaded N/M layers" line, so layer counts are not recorded.

### The one failed check (kept as a failure)
`explicit gemma` on the prompt "Reply with one short sentence about what a mutex is." returned an empty reply: HTTP 200, finish_reason=length,
all tokens spent in reasoning_content (2917 chars at 1024 tokens; still no answer at 3000 tokens). Follow-up probes: "Reply with exactly: OK" -> `OK`;
"Write a Python function that returns the square of a number." -> full answer; "What is a mutex?" -> empty again at 1024 tokens.
Interpretation: gemma is a reasoning model and can exhaust max_tokens on open-ended questions. The deployment served gemma correctly; this was **not** compared
against the old server on the same prompt, so it is not shown to be new or old. Users should give gemma a larger max_tokens or a directive prompt.

## Windows -> kaggle2 through an SSH tunnel (OpenAI Python SDK 2.24.0): 10/10 passed
| Result | Check |
|---|---|
| PASS | PC curl-equivalent /health via tunnel |
| PASS | SDK models.list() |
| PASS | SDK chat deepseek |
| PASS | SDK chat gemma |
| PASS | SDK chat qwen3 |
| PASS | SDK chat fast |
| PASS | SDK chat quant |
| PASS | SDK automatic routing (model=auto) |
| PASS | SDK streaming |
| PASS | SDK unknown model raises NotFoundError |

## Shutdown and integrity (after all tests)
- SIGTERM to the API (pid 2371): exited in 0.5 s; llama-server stopped by the shutdown handler. 0 MAPLE or llama-server processes remain.
- Ports 8000 and 8081 no longer listening; requests fail; from the PC through the still-open tunnel the API returned no answer (HTTP 000). The tunnel process I opened was then stopped; PC port 8000 is free.
- GPU memory 0 MiB on both GPUs after shutdown.
- After the run: 3 archive checksums still match; all 4 model SHA256 still match; /kaggle/working/llama.cpp and /kaggle/working/maple do not exist (never created).
- Nothing was deleted. Logs, project, models, archives and the validation JSON remain on kaggle2 (/tmp/maple_api_8000.log, /kaggle/working/maple_kaggle2_validation/).

## Not verified here
Dataset-based restoration (BLOCKED); a fresh-session test with the later deeper-search restore script (not run on this server); long contexts and benchmarks on this host;
VS Code / Continue / Open WebUI; layer-level offload counts.
