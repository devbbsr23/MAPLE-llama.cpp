"""Authoritative configuration: MODEL_ROSTER, aliases and runtime settings.

Model paths are EXTERNAL references. MAPLE never copies, moves, renames,
downloads, quantizes or packages model files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Authoritative model registry (exact IDs and paths supplied by the owner).
# ---------------------------------------------------------------------------
MODEL_ROSTER = {
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek-Coder-V2-Lite MAPLE",
        "role": "Systems Engineering and Algorithm Specialist",
        "specialties": ["C++20", "Rust", "Concurrency", "Lock-Free Structures", "Algorithms"],
        "backend": "llama",
        "model_path": "/kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "gemma": {
        "id": "gemma",
        "name": "Gemma 4 26B MAPLE",
        "role": "General Software and Code Review Specialist",
        "specialties": ["Python", "Web Development", "Frontend", "Refactoring", "Documentation"],
        "backend": "llama",
        "model_path": "/kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf",
        "temperature": 0.5,
        "max_tokens": 4096,
    },
    "qwen3": {
        "id": "qwen3",
        "name": "Qwen3 8B Q4_K_M",
        "role": "Strategic Reasoning and Architecture Specialist",
        "specialties": [
            "Architectural Decisions", "Strategic Planning", "Complex Reasoning",
            "System Design", "Trading Analysis", "Decision Making",
        ],
        "backend": "llama",
        "model_path": "/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf",
        "temperature": 0.6,
        "max_tokens": 4096,
    },
    "fast": {
        "id": "fast",
        "name": "Qwen2.5 Coder 1.5B Q4_K_M",
        "role": "Ultra-Fast Triage and Terminal Copilot",
        "specialties": ["Git Commits", "Regex", "Shell", "Quick Syntax Checks", "Fast Edits"],
        "backend": "llama",
        "model_path": "/kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
}

# The brief lists the quantitative specialist as "quant"; MODEL_ROSTER keys it
# "qwen3". The roster key is canonical; "quant" is accepted as an alias.
MODEL_ALIASES = {"quant": "qwen3"}

MODEL_IDS = tuple(MODEL_ROSTER)


def _apply_env_overrides(roster=MODEL_ROSTER):
    """MAPLE_MODEL_<ID>=/path/to/file.gguf (e.g. MAPLE_MODEL_DEEPSEEK) replaces that model's path. The IDs, specialties and
    routing are unchanged; no file is read, copied or modified here."""
    for mid, m in roster.items():
        p = os.environ.get("MAPLE_MODEL_" + mid.upper())
        if p:
            m["model_path"] = p


_apply_env_overrides()

DEFAULT_RUNTIME = {
    "host": "127.0.0.1",
    "api_port": 8000,
    "llama_port": 8081,
    "default_ctx": 4096,
    "startup_timeout_s": 240,
    "gpu_layers": 99,
    "parallel_slots": 1,
    "flash_attn": "auto",
    "unload_timeout_s": 20,
    "driver_lib_dirs": ["/usr/local/nvidia/lib64"],
    "nvidia_smi_candidates": ["/opt/bin/nvidia-smi", "/usr/bin/nvidia-smi", "nvidia-smi"],
    "default_model": "gemma",
    "models": {},
}


def maple_home() -> Path:
    env = os.environ.get("MAPLE_HOME")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def llama_bin_dir() -> Path:
    env = os.environ.get("MAPLE_LLAMA_BIN")
    if env:
        return Path(env).resolve()
    return maple_home() / "vendor" / "llamacpp" / "llama.cpp" / "build" / "bin"


def _load_structured(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # optional
        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text)  # config files are JSON-compatible YAML


def load_runtime_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_RUNTIME))
    p = maple_home() / "config" / "runtime.yaml"
    if p.exists():
        user = _load_structured(p)
        for k, v in user.items():
            if k == "models":
                cfg["models"].update(v)
            else:
                cfg[k] = v
    return cfg


def resolve_model_id(name: str) -> str:
    n = (name or "").strip().lower()
    n = MODEL_ALIASES.get(n, n)
    if n not in MODEL_ROSTER:
        raise KeyError(name)
    return n


def build_env(cfg: dict | None = None, base: dict | None = None) -> dict:
    """Environment for llama.cpp child processes.

    LD_LIBRARY_PATH order: real NVIDIA driver dir, then the restored MAPLE
    llama.cpp bin dir (shadows the hard-coded RUNPATH baked into the binaries),
    then whatever was already set. System/shell env is never modified."""
    cfg = cfg or load_runtime_config()
    env = dict(os.environ if base is None else base)
    parts = [d for d in cfg.get("driver_lib_dirs", []) if Path(d).is_dir()]
    parts.append(str(llama_bin_dir()))
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env
