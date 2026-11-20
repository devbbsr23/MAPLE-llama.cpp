# Artifacts

The archives are **not stored in Git** (they are large binaries). They are distributed as assets of a GitHub release; build or
verify them with the scripts in `scripts/`. This folder keeps only this file, `MANIFEST.json` and `checksums.sha256`.

| Archive | What is inside | Required? |
|---|---|---|
| `MAPLE-llamacpp-cuda.tar.gz` | compiled llama.cpp `build/bin` (llama-server, llama-cli, llama-quantize, ... and its shared libraries), `llama.cpp/LICENSE`, build metadata. About 60 MB. No GGUF files, no NVIDIA libraries. | **Yes** (the inference backend) |
| `MAPLE-runtime.tar.gz` | the MAPLE application: source, config, scripts, tests, docs, notebooks, manifests, licence files | Yes, unless you clone the repository |
| `MAPLE-python-deps.tar.gz` | offline wheel and lock file for the optional PyYAML | Optional |
| `MAPLE-complete.tar.gz` | the three above in one file | Convenience |

Reference checksum of the backend archive (must match what you download):
`MAPLE-llamacpp-cuda.tar.gz`  SHA256 `0be6e25deedb37c2ab01fd53c761a7efcacddd3ff37d6713e4787bd9062a1b49`.
All current values are in `checksums.sha256` (release assets carry the same file) and `MANIFEST.json`.

## Verify

```bash
sha256sum -c checksums.sha256                      # Linux / Git Bash, run in the folder holding the archives
python3 scripts/verify_artifacts.py <folder>       # also checks gzip integrity and that no *.gguf member exists
```
`restore_kaggle.py` performs the checksum verification itself before extracting anything.

## Restore

See `docs/RESTORATION.md`. In short: `python3 scripts/restore_kaggle.py --artifacts <folder> --dest <project dir>`.

## Why models are excluded

Model files are third-party works with their own licenses and are large (the validated four total about 14.6 GiB). MAPLE never packages,
copies, downloads or redistributes them, and no model archive is created. Provide your own files: `docs/MODEL_CONFIGURATION.md`.

## Build them yourself

On a host that has the compiled llama.cpp distribution you want to reuse (no compilation):
`python3 scripts/package_artifacts.py estimate`, then `llamacpp --src-archive <your compiled archive>`, `runtime`, `deps`, `complete`, `manifest`; then `scripts/verify_artifacts.py`.
