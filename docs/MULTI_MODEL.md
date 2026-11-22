# Multi-resident models and qualification-aware routing

Status: implemented on branch `feature/multi-resident-routing`; verified on a 2 x Tesla T4 Kaggle session on 2026-11-22.
Evidence and measurements are kept outside this repository (qualification reports A-F); this page describes behaviour only.

## What it does
* **Four resident models** (`fast`, `deepseek`, `qwen3`, `gemma`) each run in their own `llama-server`, pinned to a GPU and port by the
  `residency.layout` block in `config/runtime.yaml` (default: gemma + deepseek on GPU 0, qwen3 + fast on GPU 1; ports 8101-8104).
* **No model switching** between requests: resident models are reused; an idle model may be evicted only if a load needs VRAM
  (free VRAM minus `vram_margin_mib`); a **busy model is never evicted**. A request for a model that cannot be loaded
  fails with a controlled error (503) and a cooldown (`fail_cooldown_s`), it never forces memory.
* **Parallelism**: one request per model (`-np 1`); requests to different models on different GPUs run concurrently. Requests to the
  same model queue (`queue_timeout_s`).
* **Per-model settings**: `reasoning off` for gemma/qwen3, `--cache-ram 1024` for every server (llama.cpp's default is 8192 MiB per
  server), deepseek defaults (temperature 0, repeat penalty 1.05, English system prompt).

## Routing
* **Explicit** (`"model": "qwen3"` or alias `quant`) is never changed. The response is annotated with the category, whether the model
  is certified for it, and whether verification is required.
* **Automatic** (`"model": "auto"`, enabled by the `routing` block): the request is classified into a task category, matched against
  the qualification table in `config/runtime.yaml`, and sent to the best *available* model of the best tier
  (`qualified` > `limited` > `best_effort`), weighing residency, queue length and recent load failures. `deepseek` is experimental and
  is never chosen automatically. Categories with no qualified model (test generation, triage) are served best-effort and marked
  `certified: false`, `verify_required: true`.
* Request control field (optional): `"maple": {"category": "...", "exclude": ["gemma"], "require_certified": true}`.
  `require_certified` returns **409 `no_qualified_model`** instead of serving an uncertified answer; no available model returns **503**.
* Response headers: `X-MAPLE-Model`, `X-MAPLE-Route`, `X-MAPLE-Category`, `X-MAPLE-Tier`, `X-MAPLE-Certified`, `X-MAPLE-Verify`;
  JSON: `x_maple.qualification`. `POST /v1/route` previews a decision without running inference.

## Turning it off / rollback
* Remove the `routing` block: automatic routing returns to the original keyword router.
* Remove the `residency` block: MAPLE returns to the original single-model behaviour.
* `config/runtime.single_model.yaml` is the original 0.1.0 configuration, kept verbatim. `git checkout main` restores the old code.

## Tests
`tests/test_multi_resident.py` (fake llama-servers, stub upstreams over loopback): residency, eviction, queues, failure handling,
routing decisions, HTTP errors and headers. The qualification table is provisional (n = 10 tasks per category) and must be re-measured
when models or settings change.
