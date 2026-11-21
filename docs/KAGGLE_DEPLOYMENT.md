# Kaggle deployment (without recompilation)

**Kaggle temporary storage disappears.** `/kaggle/working` (and `/tmp`) is wiped when a session ends or is reset, and nothing
you restored or placed there is guaranteed to come back. Keep these in persistent storage *you* control, as private Kaggle Datasets
or on your own computer: (1) the archive set from the GitHub release, (2) your four GGUF model files, (3) this repository.

## One-time preparation

1. Create a private Kaggle Dataset **`maple-artifacts`** with `MAPLE-llamacpp-cuda.tar.gz`, `MAPLE-runtime.tar.gz`,
   `MAPLE-python-deps.tar.gz` and `checksums.sha256`.
2. Create a private Kaggle Dataset **`maple-models`** with your four GGUF files (any folder layout, original file names).
   Model files are never part of the archives and are never downloaded by MAPLE.

## Every new session

> Shortcut: [KAGGLE_BOOTSTRAP.md](KAGGLE_BOOTSTRAP.md) replaces the manual steps below with one notebook cell (verified file install, restore if needed, heartbeat, API, `/health` and `/v1/models`). The manual steps remain valid.

1. New notebook, accelerator **GPU T4 x2**, attach both datasets (they appear under `/kaggle/input/...`).
2. **Clone MAPLE** (needs Internet enabled; or unpack `MAPLE-runtime.tar.gz` from the dataset):
   ```bash
   cd /kaggle/working && git clone <this-repo-url> MAPLE-llama.cpp && cd MAPLE-llama.cpp
   ```
3. **Link the models** to the paths MAPLE expects (dry run first; symlinks only, nothing is copied):
   ```bash
   python3 scripts/link_models.py --source /kaggle/input
   python3 scripts/link_models.py --source /kaggle/input --apply
   ```
4. **Restore** (verifies archive checksums, restores llama.cpp, checks CUDA 13 / cuBLAS / NCCL / glibc, verifies model checksums,
   installs the Python wheel, runs a smoke test and an API health check):
   ```bash
   python3 scripts/restore_kaggle.py --artifacts /kaggle/input/maple-artifacts --dest "$PWD"
   ```
   Read `docs/RESTORATION.md` for every check and exit code. If it stops, fix the reported cause and run it again.
5. **Start MAPLE** and check it:
   ```bash
   source maple.env && scripts/start_maple.sh          # foreground; use a second cell or nohup for background
   curl -s http://127.0.0.1:8000/health
   python3 scripts/smoke_api.py --base http://127.0.0.1:8000     # all four specialists + auto routing
   ```
   Background example for a notebook cell: `nohup bash -c 'source maple.env && exec scripts/start_maple.sh' > logs/api.log 2>&1 &`.
6. Call it: `docs/OPENAI_API.md`. From your own computer: `docs/LOCAL_CLIENTS.md`.

The same sequence is in `notebooks/MAPLE_Kaggle_Restore.ipynb` (restoration) and `notebooks/MAPLE_Kaggle_Build.ipynb`
(detects the environment, restores the compiled llama.cpp, validates models, benchmarks, packages and restore-tests; it
contains **no compilation step**).

## Recovering after a session expires or crashes

Start a new session and repeat "Every new session". Nothing needs to be rebuilt. If you work from a Windows PC with SSH into the
session, `scripts/push_to_kaggle.py --start` uploads the archives from your local copy, restores, starts the API and runs the smoke
test in one command (details in `RECOVERY_GUIDE.md`). Keep a local copy with `scripts/sync_from_kaggle.py` and `scripts/pull_artifacts.py`.

## Resource notes (two Tesla T4, about 15 GiB each)

Only one model is loaded at a time. The 5 GB-class models fit on one T4; deepseek at 32K+ context layer-splits across both.
Disk: the restored llama.cpp is about 560 MiB uncompressed, the archives about 60 MiB; `/kaggle/working` had roughly 4 GiB free in testing.
