"""Minimal read-only GGUF header/metadata reader (stdlib only).

Reads only the header and key/value section; tensor data is never touched.
Large arrays (e.g. tokenizer vocab) are summarised by length."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_MAX_STR = 200


class GGUFError(Exception):
    pass


def _read_str(f, keep=True):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 28:
        raise GGUFError("implausible string length %d" % n)
    data = f.read(n)
    if not keep:
        return None
    s = data.decode("utf-8", "replace")
    return s if len(s) <= _MAX_STR else s[:_MAX_STR] + "...(%d chars)" % len(s)


def _read_value(f, t, keep=True):
    if t in _SCALARS:
        fmt, sz = _SCALARS[t]
        return struct.unpack(fmt, f.read(sz))[0]
    if t == 8:
        return _read_str(f, keep)
    if t == 9:
        (et, n) = struct.unpack("<IQ", f.read(12))
        if et in _SCALARS:
            sz = _SCALARS[et][1]
            f.seek(sz * n, 1)
        else:
            for _ in range(n):
                _read_value(f, et, keep=False)
        return {"__array__": n, "elem_type": et}
    raise GGUFError("unknown value type %d" % t)


def read_metadata(path) -> dict:
    """Return {'version','tensor_count','kv_count','kv':{...}}."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError("bad magic (not a GGUF file)")
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise GGUFError("unsupported GGUF version %d" % version)
        kv = {}
        for _ in range(kv_count):
            key = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            kv[key] = _read_value(f, t)
    return {"version": version, "tensor_count": tensor_count, "kv_count": kv_count, "kv": kv}


def summarize(meta: dict) -> dict:
    """Pick the architecture-relevant fields without assuming MoE."""
    kv = meta["kv"]
    arch = kv.get("general.architecture")

    def g(suffix):
        return kv.get("%s.%s" % (arch, suffix))

    experts = g("expert_count")
    used = g("expert_used_count")
    toks = kv.get("tokenizer.ggml.tokens")
    return {
        "gguf_version": meta["version"],
        "tensor_count": meta["tensor_count"],
        "architecture": arch,
        "name": kv.get("general.name"),
        "file_type": kv.get("general.file_type"),
        "quantization_version": kv.get("general.quantization_version"),
        "context_length_trained": g("context_length"),
        "block_count": g("block_count"),
        "embedding_length": g("embedding_length"),
        "feed_forward_length": g("feed_forward_length"),
        "head_count": g("attention.head_count"),
        "head_count_kv": g("attention.head_count_kv"),
        "expert_count": experts,
        "expert_used_count": used,
        "is_moe": bool(experts and experts > 1),
        "rope_freq_base": g("rope.freq_base"),
        "tokenizer_model": kv.get("tokenizer.ggml.model"),
        "vocab_size": toks["__array__"] if isinstance(toks, dict) else None,
        "has_chat_template": "tokenizer.chat_template" in kv,
    }


def sha256_file(path, chunk=8 << 20, progress=None) -> str:
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress:
                progress(done)
    return h.hexdigest()
