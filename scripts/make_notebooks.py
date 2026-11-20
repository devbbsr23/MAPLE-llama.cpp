#!/usr/bin/env python3
"""Generate notebooks/MAPLE_Kaggle_Build.ipynb and MAPLE_Kaggle_Restore.ipynb (valid nbformat 4)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n").splitlines(True)}


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


COMMON = '''
import os, sys, json, subprocess, shutil
P = "/kaggle/working/MAPLE-llama.cpp"; ART = "/kaggle/working/artifacts"
SRC_ARCHIVE = "/kaggle/working/downloaded-llama-cpp-full-cuda-t4.tar.gz"   # reusable shared artifact: READ ONLY
os.environ["MAPLE_HOME"] = P
LD = "/usr/local/nvidia/lib64"   # real NVIDIA driver dir (session-only; set per command, never globally)
def sh(cmd, check=False, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, env=dict(os.environ, LD_LIBRARY_PATH=LD + ":" + os.environ.get("LD_LIBRARY_PATH", "")), **kw)
    print((r.stdout + r.stderr).rstrip()); 
    if check and r.returncode: raise SystemExit("FAILED rc=%d: %s" % (r.returncode, cmd))
    return r.returncode
'''

build = [
    md("# MAPLE build notebook\nIndependent, resumable stages. **No llama.cpp compilation** happens here: the existing compiled CUDA distribution is reused. "
       "Models are external, read-only inputs (never downloaded, copied or packaged). Run stages in order; each is safe to re-run."),
    code(COMMON + '\nprint("common helpers loaded; project dir:", P)'),
    md("## Stage 1 - Environment detection"),
    code('''
sh("nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version --format=csv")
sh("free -h | head -2; df -h /kaggle/working | tail -1")
sh("python3 --version; ldd --version | head -1; (/usr/local/cuda/bin/nvcc --version | tail -1) 2>/dev/null || echo 'no CUDA compiler present - not needed: llama.cpp is restored, never compiled'")
sh("ls -l /kaggle/working/models/*/*.gguf")
fresh = not os.path.isdir(P + "/src/maple")
print("session:", "FRESH (project not present)" if fresh else "RESTORED/EXISTING (project present)")
print("artifacts present:", sorted(os.listdir(ART)) if os.path.isdir(ART) else "none")
'''),
    md("## Stage 2 - Repository setup and commit record\nThe project lives in `/kaggle/working/MAPLE-llama.cpp`. Set `REPO_URL` to your GitHub repository to clone it when the folder is missing (leave empty to restore from artifacts instead)."),
    code('''
REPO_URL = ""   # e.g. "https://github.com/<you>/MAPLE-llama.cpp.git"
if not os.path.isdir(P + "/src/maple"):
    if REPO_URL: sh(["git", "clone", REPO_URL, P], check=True)
    else: print("Project missing: use the Restore notebook (artifacts) instead.")
else:
    if os.path.isdir(P + "/.git"): sh(["git", "-C", P, "log", "-1", "--format=project commit %H"])
    else: print("project has no .git (clean history is created at publish time, see docs/GITHUB.md)")
print("upstream MAPLE commit recorded: 5403b6f5ec3f904409d968ff6a3ea3a2b22767d1")
sh("%s/vendor/llamacpp/llama.cpp/build/bin/llama-server --version 2>&1 | head -2" % P)   # llama.cpp commit
'''),
    md("## Stage 3 - CUDA distribution (reused, NOT recompiled)\nValidates and packages the existing compiled build: checks the source archive hash, packages `MAPLE-llamacpp-cuda.tar.gz` (excluding every `*.gguf`), restores it to the portable `vendor/` dir with its modes, and checks libraries, sm_75 code and the CUDA backend."),
    code('''
ALLOW_REBUILD = False   # compilation is intentionally not implemented in this notebook
os.makedirs(ART, exist_ok=True)
sh("sha256sum " + SRC_ARCHIVE)
if not os.path.exists(ART + "/MAPLE-llamacpp-cuda.tar.gz"):
    sh(["python3", "-I", P + "/scripts/package_artifacts.py", "estimate"])
    sh(["python3", "-I", P + "/scripts/package_artifacts.py", "llamacpp"], check=True)
else: print("MAPLE-llamacpp-cuda.tar.gz already exists - not overwritten")
B = P + "/vendor/llamacpp/llama.cpp/build/bin"
sh("cd %s && python3 -I scripts/restore_kaggle.py --skip-smoke --skip-api --no-sha" % P)
sh("ldd %s/llama-server | grep -E 'not found|cuda|cublas|nccl'" % B)
sh("%s/llama-server --list-devices" % B)
sh("cuobjdump --list-elf %s/libggml-cuda.so.1 | sed 's/.*arch = //' | sort | uniq -c" % B)
for t in ("llama-server", "llama-cli", "llama-quantize", "llama-gguf", "llama-gguf-split"): print(t, os.access(B + "/" + t, os.X_OK))
'''),
    md("## Stage 4 - Model preparation\nExisting files only: exact paths, sizes, SHA256, GGUF metadata. Missing model => stop and report the exact path (nothing is downloaded)."),
    code('sh(["python3", "-I", P + "/scripts/validate_models.py"])'),
    md("## Stage 5 - Runtime testing\nLoad time, GPU/CPU memory, KV cache, speeds and a context ladder through the real runtime. A context is only validated if a prompt filling the stated fraction succeeded. Edit `CTX` per model; long contexts take minutes."),
    code('''
MODELS = "fast,qwen3,deepseek,gemma"; CTX = "4096,16384"; KV = ""   # KV e.g. "--kv-type q8_0"
sh("cd %s && python3 -I scripts/benchmark.py --keep --models %s --ctx %s %s" % (P, MODELS, CTX, KV))
'''),
    md("## Stage 6 - Unified runtime (API)\nStarts the OpenAI-compatible API (explicit selection, automatic routing, sequential model switching) and exercises it."),
    code('''
import time, urllib.request
env = dict(os.environ, LD_LIBRARY_PATH=LD + ":" + P + "/vendor/llamacpp/llama.cpp/build/bin", PYTHONPATH=P + "/src")
api = subprocess.Popen([sys.executable, "-s", "-m", "maple", "serve", "--port", "8000"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
def call(path, body=None):
    r = urllib.request.Request("http://127.0.0.1:8000" + path, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as x: return json.loads(x.read()), dict(x.headers)
for _ in range(60):
    try: print(call("/health")[0]); break
    except Exception: time.sleep(0.5)
for m in ("fast", "quant", "deepseek", "gemma", "auto"):
    o, h = call("/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "Write a regex for an IPv4 address."}], "max_tokens": 40})
    print(m, "->", h["X-MAPLE-Model"], "|", (o["choices"][0]["message"].get("content") or "")[:60].replace("\\n", " "))
api.terminate()
'''),
    md("## Stage 7 - Final validation, packaging, restore test"),
    code('''
sh("cd %s && MAPLE_LIVE=1 python3 -u -m unittest discover -s tests -v 2>&1 | tail -40" % P)
for step in ("runtime", "deps", "complete"):
    sh(["python3", "-I", P + "/scripts/package_artifacts.py", step], check=True)
sh(["python3", "-I", P + "/scripts/make_build_manifest.py"])
sh(["python3", "-I", P + "/scripts/verify_artifacts.py", ART, "--mark"])
print(open(ART + "/packaging-report.txt").read()[:1500])
'''),
    code('''
# Restore test in a CLEAN directory (own copy of the application + llama.cpp; models stay at their original paths)
CLEAN = "/kaggle/working/_restore_test"
shutil.rmtree(CLEAN, ignore_errors=True)
sh("python3 -I %s/scripts/restore_kaggle.py --artifacts %s --dest %s/MAPLE-llama.cpp --api-port 8010" % (P, ART, CLEAN))
shutil.rmtree(CLEAN, ignore_errors=True)   # safe: only the scratch copy created above
'''),
    md("## Local backup\nOn your Windows PC run `python scripts/sync_from_kaggle.py --dest <LOCAL_PROJECT_DIR>` to pull all source files (never archives/binaries/models)."),
]

restore = [
    md("# MAPLE restore notebook\nFresh Kaggle session: attach the dataset that holds the `MAPLE-*.tar.gz` artifacts + `checksums.sha256` (and make sure the four GGUF models exist at their original paths). Restores without recompiling llama.cpp and without downloading models."),
    code(COMMON),
    md("## 1. Locate artifacts"),
    code('sh("ls -la /kaggle/input/*/ /kaggle/working/artifacts 2>/dev/null | head -40")'),
    md("## 2. Restore (checksums, extraction, CUDA compatibility, model paths, smoke test, API health)\nThe script is the single source of truth. It stops with an exact diagnostic if anything is wrong (exit 2 artifacts, 3 models, 4 CUDA/binaries, 5 tests)."),
    code('''
# If the project is not extracted yet, bootstrap just the restore scripts from MAPLE-complete.tar.gz or MAPLE-runtime.tar.gz
import glob, tarfile
if not os.path.isfile(P + "/scripts/restore_kaggle.py"):
    for cand in glob.glob("/kaggle/input/*/MAPLE-runtime.tar.gz") + glob.glob(ART + "/MAPLE-runtime.tar.gz"):
        with tarfile.open(cand) as t: t.extractall("/kaggle/working"); print("bootstrapped from", cand); break
sh("python3 -I %s/scripts/restore_kaggle.py" % P)
'''),
    md("## 3. Start the API and use it"),
    code('''
import time, urllib.request
env = dict(os.environ, LD_LIBRARY_PATH=LD + ":" + P + "/vendor/llamacpp/llama.cpp/build/bin", PYTHONPATH=P + "/src", MAPLE_HOME=P)
api = subprocess.Popen([sys.executable, "-s", "-m", "maple", "serve", "--port", "8000"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
def call(path, body=None):
    r = urllib.request.Request("http://127.0.0.1:8000" + path, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as x: return json.loads(x.read()), dict(x.headers)
for _ in range(60):
    try: print(call("/health")[0]); break
    except Exception: time.sleep(0.5)
'''),
    code('''
for m in ("deepseek", "gemma", "quant", "fast", "auto"):
    o, h = call("/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "Write a git commit message for fixing a null check."}], "max_tokens": 48})
    print(m, "->", h["X-MAPLE-Model"], "|", (o["choices"][0]["message"].get("content") or "")[:80].replace("\\n", " "))
'''),
    code('api.terminate()   # stops the API (models unload)'),
]

out = ROOT / "notebooks"
out.mkdir(exist_ok=True)
(out / "MAPLE_Kaggle_Build.ipynb").write_text(json.dumps(nb(build), indent=1))
(out / "MAPLE_Kaggle_Restore.ipynb").write_text(json.dumps(nb(restore), indent=1))
print("notebooks written")
