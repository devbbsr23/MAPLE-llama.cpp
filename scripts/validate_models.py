#!/usr/bin/env python3
"""Read-only validation of the four external GGUF models -> manifests/models_manifest.json.

Checks: path exists, readable, size, SHA256, GGUF header + metadata, and whether
the pinned llama.cpp build knows the architecture string. Models are only read.
Never downloads or replaces a model: if one is missing the script stops and
reports its exact path."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maple import gguf_meta  # noqa: E402
from maple.config import MODEL_ROSTER, llama_bin_dir, maple_home  # noqa: E402

GIB = 1024 ** 3


def arch_in_binary(arch: str, lib_dir: Path) -> bool | None:
    libs = sorted(lib_dir.glob("libllama.so.*.*.*")) or sorted(lib_dir.glob("libllama.so*"))
    if not libs or not arch:
        return None
    data = libs[0].read_bytes()
    return (b"\x00" + arch.encode() + b"\x00") in data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(maple_home() / "manifests" / "models_manifest.json"))
    ap.add_argument("--no-sha", action="store_true", help="skip SHA256 (reuse recorded value if size+mtime match)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    prev = json.loads(out.read_text()).get("models", {}) if out.exists() else {}
    models, missing = {}, []
    for mid, m in MODEL_ROSTER.items():
        p = m["model_path"]
        if not os.path.isfile(p):
            missing.append((mid, p))
            continue
        st = os.stat(p)
        rec = {"id": mid, "name": m["name"], "original_path": p, "size_bytes": st.st_size,
               "size_gib": round(st.st_size / GIB, 3), "mtime_ns": st.st_mtime_ns, "readable": os.access(p, os.R_OK)}
        old = prev.get(mid, {})
        if old.get("sha256") and old.get("size_bytes") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
            rec["sha256"] = old["sha256"]
            rec["sha256_source"] = "reused (size+mtime unchanged)"
        elif a.no_sha:
            rec["sha256"] = None
        else:
            t0 = time.time()
            rec["sha256"] = gguf_meta.sha256_file(p)
            rec["sha256_source"] = "computed in %.0fs" % (time.time() - t0)
        try:
            meta = gguf_meta.summarize(gguf_meta.read_metadata(p))
            rec["gguf"] = meta
            known = arch_in_binary(meta["architecture"], llama_bin_dir())
            rec["llamacpp_knows_architecture"] = known
            # String presence is only a hint (observed false negative: 'qwen2' loads fine). A real
            # load by scripts/benchmark.py is the authoritative compatibility test.
            rec["validation_status"] = "ok" if rec["readable"] else "unreadable"
            rec["architecture_string_in_libllama"] = known
            if known is False and not old.get("tested", {}).get("max_validated_ctx"):
                rec["validation_status"] = "unverified-architecture-string-absent (run benchmark.py)"
        except Exception as e:  # noqa
            rec["gguf_error"] = "%s: %s" % (type(e).__name__, e)
            rec["validation_status"] = "invalid-gguf"
        rec["tested"] = old.get("tested", {})  # runtime results are merged later by benchmark.py
        models[mid] = rec
        print("[%s] %s  %.3f GiB  arch=%s moe=%s -> %s" % (mid, os.path.basename(p), rec["size_gib"],
              rec.get("gguf", {}).get("architecture"), rec.get("gguf", {}).get("is_moe"), rec["validation_status"]))
    if missing:
        for mid, p in missing:
            print("MISSING model '%s': %s" % (mid, p), file=sys.stderr)
        print("STOP: not downloading replacements.", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "note": "metadata only; no model binaries", "models": models}, indent=2))
    print("wrote", out)
    return 0 if all(r["validation_status"] == "ok" for r in models.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
