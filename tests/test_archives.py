"""Archive integrity + safety tests. Skip when artifacts are not present."""
import json
import os
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("MAPLE_ARTIFACTS", "/kaggle/working/artifacts"))
NAMES = ["MAPLE-llamacpp-cuda.tar.gz", "MAPLE-runtime.tar.gz", "MAPLE-python-deps.tar.gz", "MAPLE-complete.tar.gz"]


@unittest.skipUnless(ART.is_dir() and (ART / "checksums.sha256").exists(), "artifacts not built")
class ArchiveTests(unittest.TestCase):
    def test_no_model_weights_anywhere(self):
        for n in NAMES:
            p = ART / n
            if not p.exists():
                continue
            with tarfile.open(p, "r:gz") as t:
                names = t.getnames()
                self.assertFalse([x for x in names if x.endswith(".gguf")], n)
                self.assertFalse([x for x in names if "/models/" in "/" + x], n)
                self.assertFalse([x for x in names if "/.git/" in "/" + x or x.startswith(".git/")], n)
                self.assertFalse([x for x in names if "__pycache__" in x or x.endswith(".pyc")], n)

    def test_no_models_tar(self):
        self.assertFalse((ART / "MAPLE-models.tar.gz").exists())

    def test_all_artifacts_are_targz(self):
        for n in NAMES:
            if (ART / n).exists():
                with open(ART / n, "rb") as f:
                    self.assertEqual(f.read(2), b"\x1f\x8b", n)

    def test_verify_script_passes(self):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "verify_artifacts.py"), str(ART)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_cuda_archive_known_checksum(self):
        import hashlib
        p = ART / "MAPLE-llamacpp-cuda.tar.gz"
        if not p.exists():
            self.skipTest("cuda archive absent")
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), "0be6e25deedb37c2ab01fd53c761a7efcacddd3ff37d6713e4787bd9062a1b49")

    def test_runtime_archive_has_roster_and_no_secrets(self):
        p = ART / "MAPLE-runtime.tar.gz"
        if not p.exists():
            self.skipTest("runtime archive absent")
        with tarfile.open(p, "r:gz") as t:
            cfg = t.extractfile("MAPLE-llama.cpp/src/maple/config.py").read().decode()
            self.assertIn("/kaggle/working/models/quant/Qwen3-8B-Q4_K_M.gguf", cfg)
            for m in t.getmembers():
                self.assertFalse(m.name.endswith((".env", ".pem", "id_ed25519", "id_rsa")), m.name)


if __name__ == "__main__":
    unittest.main()
