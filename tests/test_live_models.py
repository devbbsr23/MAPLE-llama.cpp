"""LIVE tests: real llama.cpp CUDA build + the four real GGUF models.

Run on the Kaggle GPU host only:  MAPLE_LIVE=1 python -m unittest discover -s tests -p 'test_live*' -v
Read-only on model files (they are only opened by llama-server)."""
import json
import os
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIVE = os.environ.get("MAPLE_LIVE") == "1"

if LIVE:
    from maple import resource_manager as rm
    from maple.config import MODEL_ROSTER, build_env, load_runtime_config
    from maple.registry import Registry
    from maple.runtime import ModelManager

PORT = 8123


def post(base, path, body, timeout=300):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read()), dict(resp.headers)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


@unittest.skipUnless(LIVE, "set MAPLE_LIVE=1 on the GPU host")
class LiveModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_runtime_config()
        cls.reg = Registry()
        cls.gpus = rm.gpu_info(cls.cfg)

    def test_00_two_t4_detected(self):
        self.assertEqual(len(self.gpus), 2, self.gpus)
        self.assertTrue(all("T4" in g["name"] for g in self.gpus))

    def test_01_all_model_paths_and_checksums(self):
        for r in self.reg.check_all():
            self.assertEqual(r["status"], "ok", r)

    def test_02_each_model_loads_and_infers_and_uses_gpu(self):
        mgr = ModelManager(self.reg, self.cfg)
        for mid in ("fast", "qwen3", "deepseek", "gemma"):
            with self.subTest(model=mid):
                with mgr.use(mid, 4096) as base:
                    before = max(g["mem_used_mib"] for g in rm.gpu_info(self.cfg))
                    out, _ = post(base, "/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 48, "temperature": 0})
                    msg = out["choices"][0]["message"]
                    self.assertTrue((msg.get("content") or msg.get("reasoning_content") or "").strip())
                    self.assertGreater(out["timings"]["predicted_per_second"], 1)
                    self.assertGreater(before, 500, "model weights not resident on a GPU")
                    self.assertEqual(mgr.current, mid)
        mgr.unload()

    def test_03_model_switching_sequential_unload(self):
        mgr = ModelManager(self.reg, self.cfg)
        order = ["fast", "deepseek", "fast", "qwen3", "gemma"]
        for mid in order:
            with mgr.use(mid, 2048) as base:
                self.assertEqual(mgr.current, mid)
                self.assertEqual(get(base, "/health")[0], 200)
        self.assertEqual(mgr.status()["switches"], 5)   # 5 loads: fast, deepseek, fast (reload after switch), qwen3, gemma
        mgr.unload()
        time.sleep(2)
        mem = [g["mem_used_mib"] for g in rm.gpu_info(self.cfg)]
        self.assertTrue(all(m < 600 for m in mem), "GPU memory not released after unload: %s" % mem)


@unittest.skipUnless(LIVE, "set MAPLE_LIVE=1 on the GPU host")
class LiveAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = dict(build_env(load_runtime_config()), MAPLE_HOME=str(ROOT), PYTHONPATH=str(ROOT / "src"))
        cls.proc = subprocess.Popen([sys.executable, "-s", "-m", "maple", "serve", "--port", str(PORT)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        cls.base = "http://127.0.0.1:%d" % PORT
        for _ in range(60):
            time.sleep(0.5)
            try:
                get(cls.base, "/health")
                break
            except Exception:
                pass

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(30)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def test_health_deep(self):
        s, h = get(self.base, "/health?deep=1")
        self.assertEqual((s, h["status"]), (200, "ok"))
        self.assertEqual(len(h["gpus"]), 2)

    def test_openai_chat_all_models_and_switching(self):
        seen = []
        for m in ("fast", "quant", "deepseek", "gemma", "fast"):
            out, hdr = post(self.base, "/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "Reply with one word: ready"}], "max_tokens": 32, "temperature": 0}, timeout=600)
            msg = out["choices"][0]["message"]
            self.assertTrue((msg.get("content") or msg.get("reasoning_content") or "").strip(), m)
            seen.append(hdr["X-MAPLE-Model"])
        self.assertEqual(seen, ["fast", "qwen3", "deepseek", "gemma", "fast"])
        s, h = get(self.base, "/health")
        self.assertEqual(h["runtime"]["loaded"], "fast")

    def test_auto_routing_end_to_end(self):
        out, hdr = post(self.base, "/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": "Write a regex that matches an IPv4 address"}], "max_tokens": 48, "temperature": 0}, timeout=600)
        self.assertEqual(hdr["X-MAPLE-Model"], "fast")

    def test_gpu_utilization_during_generation(self):
        import threading
        peak = {"u": 0}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                for g in rm.gpu_info():
                    peak["u"] = max(peak["u"], g["util_pct"] or 0)
                time.sleep(0.2)
        t = threading.Thread(target=sample)
        t.start()
        post(self.base, "/v1/chat/completions", {"model": "qwen3", "messages": [{"role": "user", "content": "Explain B-trees in 200 words."}], "max_tokens": 200, "temperature": 0}, timeout=600)
        stop.set()
        t.join()
        self.assertGreater(peak["u"], 30, "GPU utilization stayed low: %s" % peak)


@unittest.skipUnless(LIVE, "set MAPLE_LIVE=1 on the GPU host")
class LiveShutdown(unittest.TestCase):
    """Regression: SIGTERM to the API must unload the model (no orphaned llama-server on the GPU)."""

    def test_sigterm_leaves_no_llama_server(self):
        import signal
        env = dict(build_env(load_runtime_config()), MAPLE_HOME=str(ROOT), PYTHONPATH=str(ROOT / "src"))
        port = PORT + 1
        p = subprocess.Popen([sys.executable, "-s", "-m", "maple", "serve", "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        base = "http://127.0.0.1:%d" % port
        for _ in range(60):
            time.sleep(0.5)
            try:
                get(base, "/health")
                break
            except Exception:
                pass
        post(base, "/v1/chat/completions", {"model": "fast", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}, timeout=300)
        p.send_signal(signal.SIGTERM)
        p.wait(60)
        time.sleep(3)
        ps = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
        self.assertNotIn("llama-server", ps, "orphaned llama-server after API SIGTERM: " + ps[-400:])
        mem = [g["mem_used_mib"] for g in rm.gpu_info()]
        self.assertTrue(all(m < 600 for m in mem), mem)


if __name__ == "__main__":
    unittest.main()
