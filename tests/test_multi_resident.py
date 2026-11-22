"""Multi-resident / qualification-aware routing tests. Each check script runs in a clean interpreter against fake
llama-servers and stub upstreams: no GPU, no model files and no network beyond loopback are needed."""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = str(ROOT / "config" / "runtime.yaml")
CASES = [("check_multimodel.py", []), ("check_snapshot.py", []), ("check_routing.py", [CFG]), ("check_http.py", [CFG])]


@pytest.mark.parametrize("script,extra", CASES, ids=[c[0] for c in CASES])
def test_multi_resident(script, extra):
    p = subprocess.run([sys.executable, "-I", str(ROOT / "tests" / "multi_model" / script), str(ROOT / "src"), *extra],
                       capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-1500:]
    assert "FAIL" not in p.stdout
