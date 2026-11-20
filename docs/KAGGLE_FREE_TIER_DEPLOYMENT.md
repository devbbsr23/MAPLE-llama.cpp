# Deploying MAPLE on a free Kaggle account

MAPLE restores a **pre-compiled** llama.cpp CUDA build. **llama.cpp is restored, never compiled**: nothing in this guide runs CMake,
a CUDA compiler or a build step, and you do not need a compiler. You supply your own GGUF models.

Everything marked *tested* was run on a Kaggle notebook with 2 x Tesla T4 on 2026-11-20. Kaggle's free-tier rules change; the numbers
in section F are what I observed or what Kaggle documented at the time, not guarantees. Check Kaggle's current documentation.

## What survives a session and what does not

| Location | Survives a session restart? | Put here |
|---|---|---|
| `/kaggle/working`, `/tmp`, `/root` | **No** (wiped when the session ends or is reset; a notebook "Save Version" snapshot is not a safe backup of binaries) | everything MAPLE restores |
| Kaggle **Datasets** (`/kaggle/input/...`, read-only) | **Yes** | the four GGUF models, and the release archives + `checksums.sha256` |
| Your own PC | Yes | a local copy of the repo, the archives and `RECOVERY_GUIDE.md` |

Do not assume the models or the archives are still there after a restart. Keep both in Datasets.

## A. Kaggle setup

1. Create a Kaggle account and verify your phone number (Kaggle requires this for GPUs and internet access in notebooks).
2. Create a **new Notebook** (Code). In the notebook's settings, set **Accelerator** to a GPU, and turn **Internet** on (needed for `git clone`).
   * Choose **GPU T4 x2** (two Tesla T4). A single T4 also works for the models that fit on one GPU, but the validated numbers
     below were measured with two T4s.
   * **Do not choose GPU P100**: the compiled build targets compute capability 7.5 (Turing). On a P100 the restore stops with
     an explicit "may not be covered by the compiled CUDA arch" message; it never recompiles for you.
   * GPU availability is not guaranteed on the free tier; a GPU may be unavailable or your weekly GPU quota may be used up.
3. Check what you got (all read-only):
```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv   # expect two Tesla T4
free -g | head -2                                                            # system RAM
df -h /kaggle/working | tail -1                                              # disk
python3 --version; ldd --version | head -1                                   # Python 3.9+, glibc >= 2.38 needed
```
If `nvidia-smi` fails, use `LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /opt/bin/nvidia-smi` (what was needed on the tested image).
`python3 scripts/restore_kaggle.py` repeats these checks for you and stops with the exact reason if one fails.

## B. Model storage (private Dataset)

1. On kaggle.com choose **Datasets -> New Dataset**, set it **Private**, upload your four GGUF files. The expected file names are:
   * `DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf`
   * `gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf`
   * `Qwen3-8B-Q4_K_M.gguf`
   * `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
   (about 14.6 GiB in total; check your Dataset quota). Only upload models whose license permits it.
2. In the notebook choose **Add Input** and attach that Dataset. Find where it is mounted: `find /kaggle/input -maxdepth 3 -name '*.gguf'`.
3. MAPLE expects the models at fixed paths (`/kaggle/working/models/{deepseek,gemma,quant,fast}/<file>`, see `src/maple/config.py`).
   Create the links; nothing is copied:
```bash
python3 scripts/link_models.py --source /kaggle/input            # dry run: shows what would be linked
python3 scripts/link_models.py --source /kaggle/input --apply     # creates symlinks, never overwrites an existing path
```
4. Validate. Restoration (section D) checks existence, size, the GGUF header and, for the recorded models, SHA256. If your files are
   different builds, the recorded size/SHA256 will not match by design: see [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for `--no-sha`
   and for overriding paths.

## C. Restoring the compiled llama.cpp

The backend is `MAPLE-llamacpp-cuda.tar.gz` (about 60 MB: llama.cpp `build/bin`, its shared libraries, its `LICENSE` and build metadata;
no models and no NVIDIA libraries). Put it, with `checksums.sha256`, in a Kaggle Dataset (or fetch it from the repository's release
page once published). The restore script does the following, in order, and stops at the first failure:

1. locates the archives (`--artifacts <dir>`, default search includes `/kaggle/input`) and verifies **SHA256** against `checksums.sha256`;
2. checks free disk space before extracting;
3. extracts into `<MAPLE dir>/vendor/llamacpp/` (never into `/kaggle/working/llama.cpp`, which it does not touch);
4. re-applies the executable permission bits recorded in the archive (Kaggle snapshots can strip them);
5. checks glibc >= 2.38, the shared libraries (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libnccl.so.2`), the GPUs and T4 coverage;
6. runs a real smoke test (loads `fast`, asks for one word) and an API health check.

MAPLE sets `LD_LIBRARY_PATH` **per child process only** (real NVIDIA driver directory, then the restored `build/bin`), because the binaries
carry a fixed `RUNPATH` from where they were built. It writes `maple.env` with the same values; nothing global is changed.

The compiled build requires on the host: an NVIDIA driver that supports CUDA 13 (580.x on the tested image), the CUDA 13 runtime,
cuBLAS and NCCL libraries, glibc >= 2.38, libstdc++, libssl 3, libgomp. They are **not** bundled and are never installed or replaced
by MAPLE. Details and the incompatibility behaviour: [RESTORATION.md](RESTORATION.md), [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md).

## D. Restoring MAPLE and starting the API

```bash
cd /kaggle/working
git clone https://github.com/devbbsr23/MAPLE-llama.cpp.git && cd MAPLE-llama.cpp
python3 scripts/link_models.py --source /kaggle/input --apply
python3 scripts/restore_kaggle.py --artifacts /kaggle/input/<your-archive-dataset> --dest "$PWD"
source maple.env && scripts/start_maple.sh            # foreground; use a second cell or a background cell for the next commands
curl -s http://127.0.0.1:8000/health
python3 scripts/smoke_api.py --base http://127.0.0.1:8000     # one short answer from each specialist, then auto-routing
```
`restore_kaggle.py` installs the optional PyYAML from the offline bundle if `MAPLE-python-deps.tar.gz` is present; the runtime itself uses only
the Python standard library. Or open `notebooks/MAPLE_Kaggle_Restore.ipynb`, which runs the same steps.
Restore exit codes: 2 archives missing/corrupt, 3 a model file is missing or invalid (every path is listed), 4 binaries cannot run here,
5 smoke/API test failed, 6 not enough disk.

## E. Recovering after session loss

Disconnection, runtime restart, session expiry, a different GPU being assigned, or loss of `/kaggle/working` all end in the same place:
a fresh `/kaggle/working` with your Datasets still attached. Recovery is the same five commands:

```bash
cd /kaggle/working && git clone https://github.com/devbbsr23/MAPLE-llama.cpp.git && cd MAPLE-llama.cpp
python3 scripts/link_models.py --source /kaggle/input --apply
python3 scripts/restore_kaggle.py --artifacts /kaggle/input/<your-archive-dataset> --dest "$PWD"
source maple.env && scripts/start_maple.sh
```
* **GPU reassignment:** if you get a different GPU type, restore re-checks and tells you whether the build covers it (T4 yes; P100 no).
  With one GPU instead of two, the larger models may not fit at the context lengths measured on two T4s.
* **If you keep a copy on your PC** (see [RECOVERY_GUIDE.md](../RECOVERY_GUIDE.md)), `python scripts/push_to_kaggle.py --start` uploads the
  archives over SSH, restores, starts the API and smoke-tests all four specialists in one command. *Tested* from the PC against a clean
  directory on a live session; a completely new session started from nothing has not been tested.
* Restoring does not recompile anything and does not download any model.

## F. Free-tier limitations (read before relying on this)

* **No guarantee of availability.** GPU access, the weekly GPU quota, session length and idle timeouts are set by Kaggle and can change
  or be unavailable. Sessions end; this is why recovery is a first-class part of MAPLE. MAPLE is not an always-on service.
* **Disk and memory.** `/kaggle/working` is limited (this session had about 3.9 GiB free after the project, archives and test artifacts were in place);
  system RAM was about 31 GB. Models are read from the Dataset through symlinks, so they do not consume working disk.
* **Two T4s are two separate 15 GiB devices, not one 32 GiB pool.** MAPLE places a model on one GPU when it fits in about 85% of that GPU's free
  memory and otherwise splits layers across both. Context length is limited by KV-cache memory on the GPU(s) holding the model.
* **Context lengths actually validated on 2 x T4:** gemma 131,072; deepseek 65,536; qwen3 32,768; fast 32,768. **128K is not
  validated for deepseek, qwen3 or fast** (a deepseek 131,072 attempt did not complete). Default context is 4,096 until you raise it
  in `config/runtime.yaml` after your own testing.
* **One model resident at a time**; switching takes a few seconds (load times 1.5 to 6.5 s at 4,096 context were measured).
* **Hardware variation.** Only 2 x Tesla T4 was tested. Other GPUs or a different Kaggle image may not work, and the restore will say why.
* **Thinking models.** `qwen3` can spend its whole token budget reasoning; give it a larger `max_tokens` (512 or more) for short questions.

Next: [LOCAL_API_USAGE.md](LOCAL_API_USAGE.md) to call the running API from your PC.
