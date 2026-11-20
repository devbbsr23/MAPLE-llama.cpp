# Installation

MAPLE has two parts that are installed differently:

| Part | How |
|---|---|
| Python application (`src/maple`) | `git clone`, nothing to compile. Standard library only; `pip install -r requirements.txt` adds optional PyYAML. `pip install .` also works (`pyproject.toml`). |
| llama.cpp CUDA backend | **restored from `MAPLE-llamacpp-cuda.tar.gz`**, never compiled: see `RESTORATION.md` |
| Models | your own GGUF files: see `MODEL_CONFIGURATION.md` |

## Python dependencies

```bash
python3 -m pip install -r requirements.txt            # optional: PyYAML
# fully offline, from MAPLE-python-deps.tar.gz (after unpacking it):
python3 -m pip install --no-index --find-links wheels -r requirements.lock
```
`scripts/restore_kaggle.py` does the offline install into `vendor/pydeps` for you when the dependency archive is present.
The wheel in the archive is for the Python and platform recorded in `wheels.json` (CPython 3.13, manylinux x86-64); on anything else install online.

## Quick local checks (any OS with Python 3.9+, no GPU)

```bash
python -m unittest discover -s tests          # registry, router, API (real server process, no models), archives
python -m maple route "write a lock-free queue in C++20"   # shows the chosen specialist
python -m maple status                         # checks the four model paths (exit code 2 if any is missing)
```
Run with `PYTHONPATH=src` if you did not `pip install .`. The GPU tests are skipped unless `MAPLE_LIVE=1` on a host that has the models and the restored build.

## Configuration

`config/runtime.yaml` (JSON-compatible YAML): `host`, `api_port`, `llama_port`, `default_ctx`, per-model `ctx`, `gpu_layers`,
`parallel_slots`, `flash_attn`, `default_model`, `driver_lib_dirs`. `config/models.yaml` documents the roster.
Environment variables: `MAPLE_HOME`, `MAPLE_LLAMA_BIN`, `MAPLE_MODEL_<ID>`, `MAPLE_STRICT_MANIFEST`, see `config/example.env`.
Copy it to `maple.env` (git-ignored) if you want a local file; `restore_kaggle.py` generates a `maple.env` for the Kaggle layout.

## Deploying on Kaggle

`KAGGLE_DEPLOYMENT.md`. Recovering after a session ends: `../RECOVERY_GUIDE.md`.
