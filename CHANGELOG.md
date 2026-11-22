# Changelog

## Unreleased: multi-resident models and qualification-aware routing (branch `feature/multi-resident-routing`)

* Four models resident at once, pinned to GPUs by `residency.layout`; memory-aware loading, busy models never evicted, controlled load failures with cooldown, per-model queues.
* Qualification-aware automatic routing (`routing` block, `src/maple/qualified_routing.py`): task classification, tiered qualification table, residency/queue/failure aware, deepseek experimental, `maple.require_certified`, certification headers.
* `--cache-ram 1024` per server (llama.cpp default 8192 MiB x 4 could exceed the container limit).
* New: `config/runtime.single_model.yaml` (original configuration kept for rollback), `docs/MULTI_MODEL.md`, `tests/test_multi_resident.py`.
* Without `residency` / `routing` blocks MAPLE behaves exactly as 0.1.0.

## 0.1.0 (unreleased)

First public version of MAPLE-llama.cpp.

* Independent codebase that uses [AmSach/maple](https://github.com/AmSach/maple) (commit `5403b6f`, Apache-2.0) as an upstream reference only.
* OpenAI-compatible API (`/health`, `/v1/models`, `/v1/chat/completions` with streaming, `/v1/completions`, `/v1/route`) with automatic or explicit specialist selection.
* Sequential model loading on `llama-server`, GPU placement for two Tesla T4, validated context lengths recorded per model.
* Restoration of a pre-compiled llama.cpp CUDA build (commit `ef3848f`) from checksummed TAR.GZ archives: checksum, disk, permission, CUDA/NCCL/glibc,
  `llama-server --version`, model checksum and smoke checks. No compilation anywhere in the deployment path.
* Kaggle build and restore notebooks, local recovery scripts, backup of the original compiled archive.
* Models are never included or downloaded; `MAPLE_MODEL_<ID>` overrides let you point at your own files.
* Tests: 56 passed on a two-T4 Kaggle host on 2026-11-20 (including HTTP-level tests against the real server); Bonsai and Ollama are not used.
* Validation records: `docs/validation/` keeps the first-session history (A) and a fresh second-session test (B) separate, with raw results in `manifests/validation/`.
  Record B includes one recorded failure (gemma empty reply on an open-ended prompt) and marks Kaggle Dataset restoration as BLOCKED (untested).
