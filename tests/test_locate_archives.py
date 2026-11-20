"""Archive discovery in scripts/restore_kaggle.py (no GPU, no Kaggle needed)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "restore_kaggle.py"
spec = importlib.util.spec_from_file_location("restore_kaggle_under_test", SCRIPT)
rk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rk)

CUDA = "MAPLE-llamacpp-cuda.tar.gz"


def touch(p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


class LocateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        rk.explicit_roots.clear()

    def tearDown(self):
        rk.explicit_roots.clear()
        self.tmp.cleanup()

    def test_flat_and_one_two_levels_still_found(self):
        for rel in (CUDA, "a/" + CUDA, "a/b/" + CUDA):
            d = Path(tempfile.mkdtemp(dir=self.root))
            touch(d / rel)
            self.assertEqual(rk.locate([str(d)]).get("llamacpp"), d / rel, rel)

    def test_deep_dataset_mount_found_for_explicit_artifacts(self):
        deep = self.root / "datasets" / "owner" / "maple-llamacpp-runtime"
        touch(deep / CUDA)
        rk.explicit_roots.add(str(self.root))
        self.assertEqual(rk.locate([str(self.root)]).get("llamacpp"), deep / CUDA)

    def test_deep_dataset_mount_found_under_kaggle_input_style_root(self):
        deep = self.root / "datasets" / "owner" / "maple-llamacpp-runtime"
        touch(deep / CUDA)
        found = rk.locate([str(self.root)], deep_roots=(str(self.root),))
        self.assertEqual(found.get("llamacpp"), deep / CUDA)

    def test_other_roots_keep_the_two_level_limit(self):
        deep = self.root / "a" / "b" / "c" / "d"
        touch(deep / CUDA)
        self.assertNotIn("llamacpp", rk.locate([str(self.root)]))   # 4 levels, not an input/explicit root

    def test_shallowest_wins_and_duplicates_do_not_change_choice(self):
        near = touch(self.root / "v2" / CUDA)
        touch(self.root / "old" / "nested" / CUDA)
        found = rk.locate([str(self.root)], deep_roots=(str(self.root),))
        self.assertEqual(found["llamacpp"], near)

    def test_missing_roots_and_empty_ok(self):
        self.assertEqual(rk.locate([str(self.root / "nope"), str(self.root)]), {})

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_symlinked_directories_are_not_followed(self):
        outside = Path(tempfile.mkdtemp(dir=self.root))
        touch(outside / CUDA)
        scan = Path(tempfile.mkdtemp(dir=self.root))
        try:
            os.symlink(outside, scan / "link", target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation not permitted")
        self.assertNotIn("llamacpp", rk.locate([str(scan)], deep_roots=(str(scan),)))

    def test_all_four_names_found_together(self):
        d = self.root / "datasets" / "o" / "s"
        for n in rk.NAMES.values():
            touch(d / n)
        rk.explicit_roots.add(str(self.root))
        self.assertEqual(set(rk.locate([str(self.root)])), set(rk.NAMES))


if __name__ == "__main__":
    unittest.main()
