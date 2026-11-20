import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maple import resource_manager as rm  # noqa: E402
from maple.config import MODEL_ALIASES, MODEL_ROSTER, resolve_model_id  # noqa: E402
from maple.registry import ModelMissing, Registry  # noqa: E402
from maple.router import Router  # noqa: E402

EXPECTED_PATHS = {
    "deepseek": "/kaggle/working/models/deepseek/DeepSeek-Coder-V2-Lite-MAPLE-Sub5GB.gguf",
    "gemma": "/kaggle/working/models/gemma/gemma-4-26B-A4B-MAPLE-IQ2_XXS.gguf",
    "qwen3": "/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf",
    "fast": "/kaggle/working/models/fast/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
}


class TestRegistry(unittest.TestCase):
    def test_exact_model_paths(self):
        self.assertEqual({k: v["model_path"] for k, v in MODEL_ROSTER.items()}, EXPECTED_PATHS)

    def test_four_ids_and_alias(self):
        self.assertEqual(set(MODEL_ROSTER), {"deepseek", "gemma", "qwen3", "fast"})
        self.assertEqual(resolve_model_id("quant"), "qwen3")
        self.assertEqual(resolve_model_id("QWEN3"), "qwen3")
        with self.assertRaises(KeyError):
            resolve_model_id("bonsai")

    def test_models_yaml_mirrors_roster(self):
        cfg = json.loads((ROOT / "config" / "models.yaml").read_text())
        self.assertEqual(cfg["models"], MODEL_ROSTER)
        self.assertEqual(cfg["aliases"], MODEL_ALIASES)

    def test_no_bonsai_or_ollama_in_runtime(self):
        for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "config").glob("*")):
            t = p.read_text(errors="ignore").lower()
            self.assertNotIn("bonsai", t, str(p))
            self.assertNotIn("import ollama", t, str(p))
            self.assertNotIn("11434", t, str(p))  # Ollama default port

    def test_missing_model_reports_exact_path(self):
        reg = Registry(roster={"fast": dict(MODEL_ROSTER["fast"], model_path="/nonexistent/x.gguf")})
        with self.assertRaises(ModelMissing) as cm:
            reg.require("fast")
        self.assertIn("/nonexistent/x.gguf", str(cm.exception))

    def test_gguf_magic_check(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"NOTG" + bytes(32))
        try:
            res = Registry(roster={"fast": dict(MODEL_ROSTER["fast"], model_path=f.name)}).check_path("fast")
        finally:
            os.unlink(f.name)
        self.assertEqual(res["status"], "invalid")
        self.assertFalse(res["gguf_magic"])


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.r = Router()

    def sel(self, text, req=None):
        return self.r.select(text, req).model

    def test_specialists(self):
        self.assertEqual(self.sel("Implement a lock-free queue in C++20 with atomics"), "deepseek")
        self.assertEqual(self.sel("Write a Rust function with unsafe pointer arithmetic"), "deepseek")
        self.assertEqual(self.sel("Refactor this Python Flask app and update the docstring"), "gemma")
        self.assertEqual(self.sel("Fix the CSS and React frontend layout"), "gemma")
        self.assertEqual(self.sel("Design a portfolio risk model with drawdown limits and sharpe backtest"), "qwen3")
        self.assertEqual(self.sel("What are the architecture trade-offs for this system design?"), "qwen3")
        self.assertEqual(self.sel("write a git commit message for this diff"), "fast")
        self.assertEqual(self.sel("give me a regex for dates"), "fast")

    def test_default_when_no_signal(self):
        d = self.r.select("hello there")
        self.assertEqual(d.model, "gemma")
        self.assertIn("default", d.reason)

    def test_explicit_wins_and_alias(self):
        self.assertEqual(self.sel("Implement a lock-free queue in C++", "fast"), "fast")
        self.assertEqual(self.sel("anything", "quant"), "qwen3")
        d = self.r.select("x", "deepseek")
        self.assertTrue(d.explicit)

    def test_auto_and_unknown(self):
        self.assertEqual(self.sel("regex please", "auto"), "fast")
        with self.assertRaises(KeyError):
            self.sel("x", "gpt-4")


class TestPlacement(unittest.TestCase):
    T4 = lambda self, i, free=14800: {"index": i, "name": "Tesla T4", "mem_total_mib": 15360, "mem_used_mib": 15360 - free, "mem_free_mib": free}

    def test_cpu_when_no_gpu(self):
        p = rm.plan_placement(4 << 30, 4096, [])
        self.assertEqual(p["devices"], [])

    def test_small_model_single_gpu(self):
        p = rm.plan_placement(1 << 30, 4096, [self.T4(0), self.T4(1)])
        self.assertEqual(len(p["devices"]), 1)
        self.assertEqual(p["split_mode"], "none")

    def test_large_ctx_splits_across_two_independent_gpus(self):
        p = rm.plan_placement(4830185248, 65536, [self.T4(0), self.T4(1)], kv_bytes_per_token=270 * 1024)
        self.assertEqual(p["devices"], [0, 1])
        self.assertEqual(p["split_mode"], "layer")

    def test_not_treated_as_one_32gb_gpu(self):
        # 20 GiB estimate: no single 14.8 GiB T4 fits, and 85% of the combined free memory does -> split, never "single"
        p = rm.plan_placement(5 << 30, 60000, [self.T4(0), self.T4(1)], kv_bytes_per_token=270 * 1024)
        self.assertEqual(p["split_mode"], "layer")
        p2 = rm.plan_placement(5 << 30, 500000, [self.T4(0), self.T4(1)], kv_bytes_per_token=270 * 1024)
        self.assertIn("exceeds", p2["reason"])

    def test_disk_guard(self):
        with self.assertRaises(OSError):
            rm.require_disk("/", 10 ** 18, "test")


if __name__ == "__main__":
    unittest.main()
