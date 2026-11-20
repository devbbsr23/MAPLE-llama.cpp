#!/usr/bin/env python3
"""Make EXISTING model files appear at the exact MODEL_ROSTER paths by creating symlinks.

For fresh Kaggle sessions where the GGUF files arrive through an attached Dataset under /kaggle/input.
Never downloads, copies, moves, renames or modifies a model file; never overwrites an existing path.
Dry run by default.

    python3 link_models.py --source /kaggle/input            # show what would be linked
    python3 link_models.py --source /kaggle/input --apply     # create the symlinks
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Authoritative paths (same as MODEL_ROSTER in src/maple/config.py; this script is standalone on purpose).
ROSTER = {
    "deepseek": "/kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf",
    "gemma": "/kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf",
    "qwen3": "/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf",
    "fast": "/kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
}


def find(root: Path, filename: str, max_depth=6):
    base = len(root.parts)
    for d, dirs, files in os.walk(root):
        if len(Path(d).parts) - base >= max_depth:
            dirs[:] = []
        if filename in files:
            return Path(d) / filename
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="directory to search (e.g. /kaggle/input); repeatable")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    missing = 0
    for mid, target in ROSTER.items():
        t = Path(target)
        if t.exists():
            print("[ok]      %-8s already present: %s" % (mid, t))
            continue
        src = next((s for s in (find(Path(r), t.name) for r in a.source) if s), None)
        if not src:
            print("[MISSING] %-8s %s  (no file named %s under %s)" % (mid, t, t.name, a.source))
            missing += 1
            continue
        print("[%s] %-8s %s -> %s" % ("link" if a.apply else "would link", mid, t, src))
        if a.apply:
            t.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, t)
    if missing:
        print("\n%d model file(s) not found. MAPLE will not start inference without them. Attach the Dataset that holds them "
              "(see RECOVERY_GUIDE.md, 'Persistent model storage'). Nothing was downloaded." % missing)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
