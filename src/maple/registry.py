"""Unified model registry built from MODEL_ROSTER (+ recorded manifest)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import gguf_meta
from .config import MODEL_ALIASES, MODEL_ROSTER, maple_home, resolve_model_id


class ModelMissing(Exception):
    def __init__(self, model_id, path, reason="file not found"):
        super().__init__("model '%s' unavailable at exact path %s (%s)" % (model_id, path, reason))
        self.model_id, self.path, self.reason = model_id, path, reason


class Registry:
    def __init__(self, roster=None, manifest_path=None):
        self.roster = roster or MODEL_ROSTER
        self.manifest_path = Path(manifest_path) if manifest_path else maple_home() / "manifests" / "models_manifest.json"
        self._manifest = None

    # -- lookup --------------------------------------------------------
    def ids(self):
        return list(self.roster)

    def get(self, name: str) -> dict:
        return self.roster[resolve_model_id(name)]

    def aliases(self):
        return dict(MODEL_ALIASES)

    # -- manifest ------------------------------------------------------
    def manifest(self) -> dict:
        if self._manifest is None:
            if self.manifest_path.exists():
                self._manifest = json.loads(self.manifest_path.read_text())
            else:
                self._manifest = {"models": {}}
        return self._manifest

    def recorded(self, model_id: str) -> dict:
        return self.manifest().get("models", {}).get(model_id, {})

    # -- validation ----------------------------------------------------
    def check_path(self, model_id: str) -> dict:
        """Cheap existence/readability/magic check; never modifies the file."""
        m = self.get(model_id)
        p = m["model_path"]
        res = {"id": m["id"], "path": p, "exists": os.path.isfile(p), "readable": False, "size": None,
               "gguf_magic": False, "size_matches_manifest": None, "status": "missing"}
        if not res["exists"]:
            return res
        res["readable"] = os.access(p, os.R_OK)
        res["size"] = os.path.getsize(p)
        rec = self.recorded(m["id"])
        # MAPLE_STRICT_MANIFEST=0 skips the size-vs-manifest comparison (for your own, differently quantized models).
        if rec.get("size_bytes") is not None and os.environ.get("MAPLE_STRICT_MANIFEST", "1") != "0":
            res["size_matches_manifest"] = rec["size_bytes"] == res["size"]
        if res["readable"]:
            try:
                with open(p, "rb") as f:
                    res["gguf_magic"] = f.read(4) == b"GGUF"
            except OSError:
                res["readable"] = False
        ok = res["exists"] and res["readable"] and res["gguf_magic"] and res["size_matches_manifest"] is not False
        res["status"] = "ok" if ok else "invalid"
        return res

    def require(self, model_id: str) -> dict:
        m = self.get(model_id)
        r = self.check_path(m["id"])
        if not r["exists"]:
            raise ModelMissing(m["id"], m["model_path"])
        if r["status"] != "ok":
            raise ModelMissing(m["id"], m["model_path"], "present but invalid: %s" % json.dumps(r))
        return m

    def check_all(self) -> list[dict]:
        return [self.check_path(i) for i in self.ids()]

    def metadata(self, model_id: str) -> dict:
        m = self.require(model_id)
        return gguf_meta.summarize(gguf_meta.read_metadata(m["model_path"]))
