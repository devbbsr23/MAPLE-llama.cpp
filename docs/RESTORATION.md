# Restoring the compiled llama.cpp (no compilation, ever)

MAPLE's inference backend is a **pre-compiled llama.cpp CUDA distribution** that is restored from an archive.
Normal deployment never runs CMake, never needs a CUDA compiler (`nvcc`) and never rebuilds anything. If a machine cannot run the
binaries, restoration stops and says why; it does not compile and does not substitute another backend.

## The archives

| Archive | Contents | Needed for deployment |
|---|---|---|
| `MAPLE-llamacpp-cuda.tar.gz` | compiled llama.cpp `build/bin` (llama-server, llama-cli, llama-quantize, ... plus shared libraries) and build metadata; **no `*.gguf`**, no test binaries | **yes** |
| `MAPLE-runtime.tar.gz` | the MAPLE application: `src`, `config`, `scripts`, `tests`, `docs`, `notebooks`, `manifests`, licence files | yes (or clone this repository instead) |
| `MAPLE-python-deps.tar.gz` | offline wheel and lock file (PyYAML, optional) | optional; the runtime uses only the Python standard library |
| `MAPLE-complete.tar.gz` | the three above in one file | convenience wrapper |

Download them from the GitHub release page of your fork/copy of this repository (they are deliberately not stored in Git; see
`artifacts/README.md`) and keep them in persistent storage: a private Kaggle Dataset, or your own disk. Verify with `checksums.sha256`.

## Restore

```bash
git clone <this-repo-url> MAPLE-llama.cpp && cd MAPLE-llama.cpp          # or unpack MAPLE-runtime.tar.gz
# archives + checksums.sha256 in one folder, for example an attached Kaggle Dataset:
python3 scripts/restore_kaggle.py --artifacts /kaggle/input/<your-archive-dataset> --dest "$PWD"
```

`--artifacts` may be repeated; without it MAPLE searches `/kaggle/input`, `/kaggle/working/artifacts` and `/kaggle/working`.
In order, the script:

1. locates the archives and **verifies their SHA256** against `checksums.sha256` (refuses to continue without a recorded checksum);
2. refuses any archive containing a `*.gguf` or an unsafe path;
3. **checks free disk space** (needs about 1.1x the extracted size) before every extraction;
4. extracts the MAPLE application and llama.cpp to `<dest>/vendor/llamacpp/llama.cpp/build/bin`; files that already exist and differ are
   reported and kept (`--overwrite-source` to replace them);
5. **restores executable permissions** from the archive modes (some Kaggle snapshots strip them);
6. installs the offline Python wheel into `<dest>/vendor/pydeps` if the dependency archive is present;
7. writes `maple.env` (`MAPLE_HOME`, `MAPLE_LLAMA_BIN`, `LD_LIBRARY_PATH`) and sets `LD_LIBRARY_PATH` per process only;
8. checks **glibc, CUDA 13 runtime libraries, cuBLAS, NCCL, libssl/libgomp/libstdc++** by running `ldd` on `libggml-cuda` and `llama-server`;
9. runs **`llama-server --version`** and `--list-devices`, and reports GPUs that are not Turing;
10. verifies the four model paths, sizes and SHA256 (see `MODEL_CONFIGURATION.md`) without touching them;
11. runs a **real inference smoke test** on `fast` (`--smoke-model` to change) and an API `/health` check.

Options: `--no-sha`, `--skip-smoke`, `--skip-api`, `--smoke-model ID`, `--api-port N`.
Exit codes: 0 success, 2 archives missing or corrupt, 3 model file missing or invalid, 4 CUDA or binary incompatibility, 5 smoke or API test failed, 6 other (for example not enough disk).

## Dependencies of the compiled build (documented, never installed by MAPLE)

The build (llama.cpp commit `ef3848f`, GNU 13.3.0, Release, shared libraries, `CMAKE_CUDA_ARCHITECTURES=75`) needs on the host:

* NVIDIA driver supporting CUDA 13.0 (tested: 580.178.04)
* CUDA 13.x runtime `libcudart.so.13`, cuBLAS `libcublas.so.13` and `libcublasLt.so.13`, NCCL `libnccl.so.2`
* glibc 2.38 or newer, libstdc++ with GLIBCXX 3.4.29+, libssl.so.3, libgomp.so.1

These are **not bundled** and not redistributed. Tested only on Kaggle's Ubuntu 24.04 image with two Tesla T4. **Universal
compatibility is not claimed.** If a library is missing, the output names it (and the script exits with code 4); install it
yourself through your platform's normal channel or use a compatible host.

### LD_LIBRARY_PATH and RUNPATH

The binaries have a hard-coded `RUNPATH=/kaggle/working/llama.cpp/build/bin`. MAPLE leaves that path alone (a protected,
possibly shared directory) and shadows it per child process: `LD_LIBRARY_PATH` = real NVIDIA driver directory (`/usr/local/nvidia/lib64`),
then MAPLE's restored `build/bin`. This ordering matters: without it a plain shell may resolve an older forward-compat `libcuda`
and fail with "CUDA driver version is insufficient for CUDA runtime version". The restored location is always
`<dest>/vendor/llamacpp/llama.cpp/build/bin` (override with `MAPLE_LLAMA_BIN`).

## Fallback if you lose `MAPLE-llamacpp-cuda.tar.gz`

If you still hold the original compiled archive (`downloaded-llama-cpp-full-cuda-t4.tar.gz`, SHA256
`541483fc880595fe20b02f04a0280eb0024d57c9bd4ac2082e7b04fd619eb5f0`), recreate the package without compiling:
`python3 scripts/package_artifacts.py llamacpp --src-archive <path>`. That archive is **your own** artifact and is not published by this repository.
