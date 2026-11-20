# Hardware and software requirements

## Tested configuration (the only one with evidence)

| Item | Value |
|---|---|
| Platform | Kaggle notebook, 2026-11-20 |
| GPUs | 2 x NVIDIA Tesla T4, 15360 MiB each (compute capability 7.5), driver 580.178.04 |
| RAM | about 31 GiB |
| OS / libc / Python | Ubuntu 24.04, glibc 2.39, Python 3.13 |
| llama.cpp | commit `ef3848f`, Release, CUDA 13.0, `CMAKE_CUDA_ARCHITECTURES=75`, shared libraries |
| Disk | `/kaggle/working` about 4 GiB free at the time of testing |

## Requirements of the restored build

* **GPU:** Turing (compute capability 7.5, e.g. T4, RTX 20xx) is what the binaries were compiled for. Restoration treats a GPU outside the recognised
  list as a compatibility problem and stops with exit code 4; other GPUs are not claimed to work.
* **Driver and libraries:** driver supporting CUDA 13; `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libnccl.so.2`;
  glibc >= 2.38; libstdc++ GLIBCXX >= 3.4.29; libssl.so.3; libgomp.so.1. None are bundled. No CUDA compiler is needed.
* **Python:** 3.9 or newer for the scripts (3.13 tested). The runtime uses only the standard library; PyYAML is optional.
* **Memory:** each model is loaded alone. Measured GPU memory after load at 4096 context: fast about 1.2 GiB, gemma about 5.0 GiB,
  qwen3 about 5.1 GiB, deepseek about 5.8 GiB, plus KV cache that grows with context (README benchmark table).
  A DeepSeek 131072-context attempt used about 29 GiB of host RAM and did not complete: do not rely on it.
* **Disk:** about 1.1x the archive contents free for extraction (560 MiB for llama.cpp) plus the model files (about 14.6 GiB, kept outside MAPLE).

## Not claimed

Other GPU generations, other CUDA versions, CPU-only use, Windows or macOS hosts, multi-GPU setups other than the two-T4 case,
and any context length above the validated values.
