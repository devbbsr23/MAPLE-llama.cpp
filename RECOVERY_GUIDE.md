# MAPLE recovery guide

**The permanent copy of MAPLE is this directory on your Windows PC**
(any folder, for example `C:\path	o\MAPLE-llama.cpp` on Windows). Kaggle is a temporary execution environment.
`/kaggle/working` and everything in it can disappear when a session ends, restarts, or is reset.

This guide gets a working inference server back quickly after a Kaggle crash or expiry,
**without recompiling llama.cpp and without downloading models.**

## 0. What is stored where

| Item | Location | Survives a Kaggle restart? |
|---|---|---|
| MAPLE source, scripts, notebooks, tests, docs, manifests | this directory (`src/`, `scripts/`, `notebooks/`, ...) | yes (Windows) |
| Compiled llama.cpp (CUDA, sm_75/T4) | `artifacts/MAPLE-llamacpp-cuda.tar.gz` | yes (Windows) |
| App bundle / Python deps bundle | `artifacts/MAPLE-runtime.tar.gz`, `artifacts/MAPLE-python-deps.tar.gz` | yes (Windows) |
| Checksums and manifests | `artifacts/checksums.sha256`, `artifacts/MANIFEST.json`, `LOCAL_BACKUP_MANIFEST.json`, `manifests/*.json` | yes (Windows) |
| **The four GGUF model files** | external; exact paths in `MODEL_ROSTER` | **No, unless you keep them in persistent storage (section 2)** |
| `/kaggle/working/*` (project, build, logs, models placed there) | Kaggle | **no** |

`MAPLE-complete.tar.gz` is a duplicate wrapper of the three archives above and is intentionally not stored locally
(`LOCAL_BACKUP_MANIFEST.json` records its checksum). The shared `downloaded-llama-cpp-full-cuda-t4.tar.gz`
(sha256 `541483fc880595fe20b02f04a0280eb0024d57c9bd4ac2082e7b04fd619eb5f0`) is your own reusable artifact and is not copied here;
`MAPLE-llamacpp-cuda.tar.gz` is its stripped copy of `build/bin` plus build metadata
(sha256 `0be6e25deedb37c2ab01fd53c761a7efcacddd3ff37d6713e4787bd9062a1b49`). Add `--include-original` to `pull_artifacts.py` if you want it stored too.

## 1. Quick recovery (about 5 minutes plus the model storage you already set up)

1. **Verify the local package** (Windows; takes seconds, needs only Python):
   `python scripts\verify_local_recovery.py --extract-test`
   It must print `LOCAL RECOVERY PACKAGE: COMPLETE`. If not, fix that first.
2. **Start a new Kaggle session**: Accelerator **GPU T4 x2**, Internet only if your tunnel needs it.
3. **Make the models available** at their exact paths (section 2). Do this before step 5.
4. **Establish SSH** to the session the way you normally do (cloudflare tunnel). A new quick tunnel usually has a **new hostname**;
   update `HostName` for `Host kaggle` in `~/.ssh/config` yourself. MAPLE tooling never edits your SSH config.
   In Git Bash, `cloudflared` must be on PATH (`export PATH="/c/cloudflared:$PATH"`); the MAPLE scripts add `C:\cloudflared` themselves.
   Test: `ssh kaggle hostname`
5. **Push, restore, start and test in one command** (Windows):
   `python scripts\push_to_kaggle.py --start`
   This verifies local checksums, checks remote disk/GPU/model paths (read-only), uploads the archives (SHA256-verified, resumable),
   runs `restore_kaggle.py`, starts the API on port 8000, and runs `smoke_api.py` against all four specialists.
6. **Use it**: from inside the session `http://127.0.0.1:8000/v1/chat/completions` (OpenAI compatible; models `deepseek`, `gemma`, `qwen3` (alias `quant`), `fast`, or `auto`).

If a step fails, the script stops and prints the reason; fix it and re-run, because every step is resumable.

### What the restore step does (and does not do)
Verifies archive SHA256 and integrity; refuses any archive containing GGUF or unsafe paths; extracts the application and the
llama.cpp build into `/kaggle/working/MAPLE-llama.cpp`; re-applies execute permissions recorded in the archive (Kaggle snapshots
strip them); checks glibc, shared libraries and CUDA (details below); verifies the four model files (size, and SHA256 when recorded);
runs a llama.cpp smoke test on the `fast` model; runs an API health check. It **never** compiles, downloads models, or installs/replaces CUDA libraries.
If the binaries are incompatible it stops and tells you why; nothing is rebuilt or substituted. The fix is a compatible host (`docs/HARDWARE_REQUIREMENTS.md`).

## 2. Persistent model storage (mandatory, not automatic)

The four model files total about 14.6 GiB and are **not** in any archive. Kaggle does not keep `/kaggle/working` across a
session restart, so you must provide them again each time. Choose one:

* **Kaggle Dataset (recommended).** Upload the four GGUF files once as a private Dataset (any folder layout, original filenames).
  Attach it to each new notebook, then create links at the exact roster paths:
  ```
  python3 /kaggle/working/maple_incoming/link_models.py --source /kaggle/input            # dry run
  python3 /kaggle/working/maple_incoming/link_models.py --source /kaggle/input --apply    # symlinks only
  ```
  `link_models.py` only creates symlinks for paths that do not exist; it never copies, moves, renames, downloads, or overwrites.
  (`push_to_kaggle.py` uploads this helper into `/kaggle/working/maple_incoming` even when models are missing, then stops and tells you.)
* **Kaggle "Save Version" output / another persistent volume you control.** Place the files at the exact paths below.
* Obtaining the files again from wherever you originally got them is **your** decision; MAPLE never downloads a replacement.

Exact paths MAPLE expects (`MODEL_ROSTER`; also in `config/models.yaml`):

```
/kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf   4830185248 bytes
/kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf            4745987168 bytes
/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf                          5027783488 bytes
/kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf        1117320768 bytes
```
Expected SHA256 values are in `manifests/models_manifest.json`; restore verifies them. If any model is missing, restore stops inference startup and lists the exact missing path.

## 3. CUDA / llama.cpp compatibility (checked on every restore)

* Compiled build: llama.cpp commit `ef3848f`, version 1.0.0-dev (build 1), GNU 13.3.0, Release, shared libs, **CUDA arch 75 (sm_75, Tesla T4)**.
* Needs on the host: glibc >= 2.38, libstdc++ (GLIBCXX 3.4.29+), **CUDA 13.x runtime** (`libcudart.so.13`), **cuBLAS 13** (`libcublas.so.13`, `libcublasLt.so.13`), **NCCL 2** (`libnccl.so.2`), libssl 3, libgomp, and an NVIDIA driver supporting CUDA 13 (tested: 580.178.04).
  These libraries are **not bundled** and are never installed or replaced by MAPLE.
* **LD_LIBRARY_PATH** is set per process only: real driver dir (`/usr/local/nvidia/lib64`) first, then the restored build dir. This matters:
  without it, a plain SSH shell can pick an older forward-compat `libcuda` and fail with
  "CUDA driver version is insufficient". The system and your shell are not modified.
* The binaries carry a hard-coded RUNPATH to `/kaggle/working/llama.cpp/build/bin`; MAPLE shadows it with `LD_LIBRARY_PATH` and never touches that directory.
* Two T4s are two independent 15 GiB devices (not one 32 GiB GPU). MAPLE places a model on one GPU when it fits (85% of free memory), otherwise layer-splits.
* Validated context lengths (from real runs, 2x T4): gemma 131072, deepseek 65536 (131072 with q8_0 KV: incomplete, unvalidated), qwen3 32768, fast 32768.

## 4. Manual recovery (no `push_to_kaggle.py`)

1. Upload `artifacts/MAPLE-*.tar.gz`, `artifacts/checksums.sha256` and `scripts/restore_kaggle.py` to one folder, or attach them as a Kaggle Dataset.
2. `python3 -I restore_kaggle.py --artifacts <folder> --dest /kaggle/working/MAPLE-llama.cpp`
3. `source /kaggle/working/MAPLE-llama.cpp/maple.env && /kaggle/working/MAPLE-llama.cpp/scripts/start_maple.sh`
4. `python3 -I /kaggle/working/MAPLE-llama.cpp/scripts/smoke_api.py --base http://127.0.0.1:8000`

Or run `notebooks/MAPLE_Kaggle_Restore.ipynb` in Kaggle with the archives attached as a Dataset.

## 5. Keeping the local copy current

* Source: `python scripts\sync_from_kaggle.py --dest <LOCAL_PROJECT_DIR>` (hash-verified, never deletes local files, reports conflicts instead of overwriting).
* Archives: `python scripts\pull_artifacts.py` (resumable; verifies against the server copy).
* After either: `python scripts\verify_local_recovery.py --extract-test`.
* If you changed source locally, repackage on Kaggle (`scripts/package_artifacts.py runtime` then `complete`) before pulling, or the runtime archive will lag behind the source. The verifier's extract test reports such a lag.

## 6. Troubleshooting

| Symptom | Cause / action |
|---|---|
| `websocket: bad handshake` on ssh | tunnel is down or the hostname changed; restart the tunnel, update `HostName` yourself |
| `exec: cloudflared: not found` | add `C:\cloudflared` to PATH for that shell |
| `PermissionError ... llama-server` | execute bits stripped; run `restore_kaggle.py` (re-applies them) |
| `CUDA driver version is insufficient` | `LD_LIBRARY_PATH` missing the real driver dir; use `maple.env` / `start_maple.sh` |
| restore stops: `model ... unavailable at exact path` | section 2 |
| restore stops: unresolved `libcudart/libcublas/libnccl` | the session image lacks CUDA 13 libs; report it; do not install over them |
| restore stops: checksum mismatch | archive corrupted or modified; re-pull from a good source, do not extract |
| `qwen3` returns an empty `content` for short requests | Qwen3 is a thinking model: it spends tokens on `reasoning_content` first. Give it a larger `max_tokens` (roster default 4096); `smoke_api.py` uses 1024 and requires a real answer |
| GPU not seen (`0 /dev/nvidia*`) | notebook has no GPU accelerator selected |

## Original compiled llama.cpp archive (separate local backup)

The shared, compiled CUDA archive is **not** part of this source folder. It is stored next to it:

```
<parent of the project folder>\backups\llamacpp\downloaded-llama-cpp-full-cuda-t4.tar.gz
<parent of the project folder>\backups\llamacpp\LLAMACPP_BACKUP_MANIFEST.json   (size, SHA256, Kaggle source path, verification time)
```

- SHA256 `541483fc880595fe20b02f04a0280eb0024d57c9bd4ac2082e7b04fd619eb5f0`, 235,142,051 bytes, hash-verified against the Kaggle original.
- Re-download or re-verify (resumable, read-only on Kaggle, never overwrites a differing local file):
  `python scripts/backup_llamacpp.py` / `python scripts/backup_llamacpp.py --verify-only`
- Restoring needs only `artifacts/MAPLE-llamacpp-cuda.tar.gz` (the MAPLE-packaged subset, 60 MB). The original archive is the
  upstream source of that package and the fallback if `MAPLE-llamacpp-cuda.tar.gz` is ever lost: `scripts/package_artifacts.py llamacpp --src-archive <path>`
  re-creates it without compiling. It contains 19 small vocabulary `.gguf` test files (not models); the MAPLE package excludes them.
- Models are never part of any backup. See "Persistent storage requirement" above.

## Clean-restoration test (2026-11-20)

`manifests/clean_restore_report.json` records a restore from these local archives into a separate empty directory on a Kaggle host
(working install untouched): executable bits, CUDA 13/cuBLAS/NCCL resolution, two T4s, four model SHA256s, offline wheel install,
llama.cpp smoke test, API health, explicit/alias/auto model selection with switching, library origin, SIGTERM shutdown with no
orphaned `llama-server` and 0 MiB GPU memory afterwards: 29/29 probe checks passed.
Limit: same Kaggle host and session as the working install; a brand-new session has not yet been exercised.
