# Troubleshooting

| Symptom | Cause and action |
|---|---|
| Restore exits **2** "no MAPLE archives found" / "cannot verify ... checksums.sha256 not found" | Pass `--artifacts <folder>`; keep `checksums.sha256` next to the archives. |
| Restore exits **2** checksum mismatch | The archive is corrupt or modified. Re-download from the release; do not extract. |
| Restore exits **3** model missing/invalid | The message lists each exact path. Attach your Dataset and run `scripts/link_models.py`; for your own files see `MODEL_CONFIGURATION.md` (`--no-sha`, `MAPLE_MODEL_<ID>`, `MAPLE_STRICT_MANIFEST=0`). MAPLE never downloads models. |
| Restore exits **4** unresolved `libcudart.so.13` / `libcublas.so.13` / `libnccl.so.2` | The host image lacks CUDA 13 runtime libraries. Use a compatible host. MAPLE does not install or substitute them and does not compile. |
| Restore exits **4** "glibc ... < 2.38" | The host OS is too old for the binaries (built on Ubuntu 24.04). |
| Restore exits **4** GPU not covered by sm_75 | Non-Turing GPU; the compiled architectures do not cover it. |
| `PermissionError ... llama-server` | Execute bits were stripped by a session snapshot. Run `restore_kaggle.py` again; it re-applies the archive modes. |
| "CUDA driver version is insufficient for CUDA runtime version" | Running llama.cpp outside MAPLE without the driver directory first on `LD_LIBRARY_PATH`. Use `source maple.env` or `scripts/start_maple.sh`. |
| `/health` returns 503 `degraded` | The `missing` list names the paths. |
| Request returns 503 `load_failed` | The message ends with the llama-server log; typical causes are out-of-memory (lower `models.<id>.ctx`) or an unsupported architecture. Full logs are in `logs/llama-<id>.log`. |
| A 32K+ DeepSeek context runs out of memory | Placement is automatic (single GPU if it fits in 85% of free memory, else layer split). Only validated contexts are claimed. |
| `qwen3` returns empty `content` | Thinking model; raise `max_tokens` (1024+). |
| Slow first answer after switching models | Each switch unloads and loads a model (about 2 to 7 s for these files); use one specialist for a stretch of work. |
| Nothing listening on port 8000 | Check `logs/api.log`; the port is configurable (`--port`, `api_port`). |
| Orphaned `llama-server` after a crash | `pkill -x llama-server` frees the GPU. A normal SIGTERM to the API stops it cleanly (tested). |
| `ssh ... websocket: bad handshake` (Windows helper scripts) | The tunnel is down or its hostname changed; fix your SSH config yourself, the scripts never edit it. |
| `cloudflared: not found` in Git Bash | Add the folder to `PATH` for that shell. |

If a documented command fails, please open an issue with the exact output and `manifests/restore_report.json`.
