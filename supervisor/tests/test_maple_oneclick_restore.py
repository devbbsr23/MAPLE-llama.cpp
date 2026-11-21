"""Tests for maple_oneclick_restore.py and notebooks/maple_bootstrap_cell.py.

Nothing here starts MAPLE, llama-server, a heartbeat, SSH or Cloudflare, and nothing uses the network. Dataset layouts are built in
temporary folders. Tests that execute the bootstrap cell need Linux (it uses os.uname) and are skipped elsewhere.
"""
import ast
import contextlib
import hashlib
import http.server
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUP = HERE.parent
REPO = SUP.parent
SCRIPT = SUP / "maple_oneclick_restore.py"
CELL = REPO / "notebooks" / "maple_bootstrap_cell.py"
MANIFEST = REPO / "notebooks" / "BOOTSTRAP_MANIFEST.sha256"


def load_script():
    spec = importlib.util.spec_from_file_location("oc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cell_manifest():
    """Evaluate the FILES literal of the bootstrap cell without running the cell."""
    tree = ast.parse(CELL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("FILES not found")


OLD_MSG = "pipe the key into `maple_tunnel.py start --key-stdin` (or set $API_KEY and run `maple_tunnel.py start`)"
STALE_MSG = "`maple_tunnel.py start --auth-only` (key from $API_KEY)"
OLD_TUNNEL_PINS = {"42f17d0985eed27a95bce29806641b4f55a42ae25ecad91c302cd2be1841fe80",     # copy at the pinned GitHub commit
                   "b255bce5d806124bf646487a51b325cf53b7f98bf72d41fd19282755cbfe3a75"}     # copy already deployed on Kaggle


def load_tunnel():
    spec = importlib.util.spec_from_file_location("maple_tunnel_under_test", SUP / "maple_tunnel.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class AuthCommandDocs(unittest.TestCase):
    """The documented authentication commands are the ones the real CLI accepts (no behaviour is exercised: argparse only)."""

    def parse(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                return load_tunnel().build_parser().parse_args(argv)
            except SystemExit:
                return None

    def test_real_cli_accepts_the_documented_commands(self):
        self.assertIsNotNone(self.parse(["start", "--key-stdin"]))
        self.assertIsNotNone(self.parse(["start"]))
        self.assertIsNotNone(self.parse(["run", "--auth-only"]))
        self.assertIsNotNone(self.parse(["status", "--auth-only"]))
        self.assertIsNotNone(self.parse(["status", "--json"]))
        self.assertIsNotNone(self.parse(["stop"]))

    def test_real_cli_rejects_the_stale_forms(self):
        self.assertIsNone(self.parse(["start", "--auth-only"]))
        self.assertIsNone(self.parse(["stop", "--auth-only"]))

    def test_no_stale_command_anywhere_in_docs_or_code(self):
        bad = re.compile(r"(?:start|stop)(?:\|\w+)*\s+--auth-only|\|(?:start|stop)\b[^`\n]{0,24}--auth-only")
        hits = []
        for p in list(REPO.rglob("*.md")) + list(REPO.rglob("*.py")) + list(REPO.rglob("*.txt")):
            if "tests" in p.parts or "__pycache__" in p.parts or ".git" in p.parts:
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if bad.search(line):
                    hits.append("%s:%d: %s" % (p.relative_to(REPO), n, line.strip()[:90]))
        self.assertEqual(hits, [])

    def test_ssh_instructions_name_the_real_command(self):
        src = (SUP / "maple_tunnel.py").read_text(encoding="utf-8")
        self.assertIn(OLD_MSG, src)
        self.assertNotIn(STALE_MSG, src)

    def test_the_fix_is_text_only(self):
        """Rebuilding the older copy by reverting that one sentence must give exactly the pinned older hash: nothing else changed."""
        new = (SUP / "maple_tunnel.py").read_bytes()
        old = new.replace(OLD_MSG.encode(), STALE_MSG.encode())
        self.assertIn(hashlib.sha256(old).hexdigest(), OLD_TUNNEL_PINS)
        self.assertEqual(hashlib.sha256(old).hexdigest(), "42f17d0985eed27a95bce29806641b4f55a42ae25ecad91c302cd2be1841fe80")

    def test_cell_still_accepts_the_older_tunnel_copies(self):
        by = {f[0]: f for f in cell_manifest()}["maple_tunnel.py"]
        self.assertTrue(OLD_TUNNEL_PINS <= set(by[2]))
        self.assertIn(62699, by[3])
        self.assertIn(sha(SUP / "maple_tunnel.py"), by[2])


class Static(unittest.TestCase):
    def test_syntax(self):
        for p in (SCRIPT, CELL):
            ast.parse(p.read_text(encoding="utf-8"))

    def test_no_secrets_or_local_paths(self):
        pat = re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}|BEGIN [A-Z ]*PRIVATE KEY|ghp_|github_pat_|[A-Za-z]:\\\\|Users[\\/]|Ai_projects|trycloudflare")
        for p in (SCRIPT, CELL, MANIFEST):
            self.assertIsNone(pat.search(p.read_text(encoding="utf-8")), p.name)

    def test_api_host_is_loopback_and_fixed(self):
        oc = load_script()
        self.assertEqual(oc.API_HOST, "127.0.0.1")
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("0.0.0.0", src.replace("0.0.0.0:8888", ""))   # only the Jupyter port is ever mentioned as non-loopback

    def test_never_starts_auth_proxy_or_ssh(self):
        src = SCRIPT.read_text(encoding="utf-8")
        for bad in ("maple_tunnel.py\", \"start", "cloudflared tunnel", "pkill", "kill -9", "SIGKILL", "--key-stdin"):
            self.assertNotIn(bad, src, bad)

    def test_no_compile_no_model_download(self):
        src = SCRIPT.read_text(encoding="utf-8")
        for bad in ("cmake", "make -j", "huggingface", "urlretrieve", "wget ", "curl "):
            self.assertNotIn(bad, src, bad)
        self.assertNotIn("--overwrite-source", src.split("def s_restore")[1].split("def s_models")[0].replace("never uses --overwrite-source", ""))


class Manifest(unittest.TestCase):
    def test_cell_pins_match_repository_files(self):
        by_name = {f[0]: f for f in cell_manifest()}
        for name, path in (("maple_oneclick_restore.py", SCRIPT), ("maple_heartbeat.py", SUP / "maple_heartbeat.py"), ("maple_tunnel.py", SUP / "maple_tunnel.py")):
            _n, _where, hashes, size, _gh, _role = by_name[name]
            self.assertIn(sha(path), hashes, name)
            self.assertIn(path.stat().st_size, size if isinstance(size, tuple) else (size,), name)

    def test_manifest_file_matches_cell_pins(self):
        pins = {}
        for f in cell_manifest():
            pins[f[0]] = f[2]
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(" *")
            self.assertIn(digest, pins[name], name)
        listed = {l.split(" *")[1] for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")}
        self.assertEqual(listed, set(pins))

    def test_supervisor_sha256sums_cover_oneclick(self):
        sums = (SUP / "SHA256SUMS.txt").read_text(encoding="utf-8")
        self.assertIn("%s *maple_oneclick_restore.py" % sha(SCRIPT), sums)

    def test_github_pin_is_a_full_commit_hash_not_a_branch(self):
        m = re.search(r'GITHUB_COMMIT\s*=\s*"([0-9a-f]+)"', CELL.read_text(encoding="utf-8"))
        self.assertIsNotNone(m)
        self.assertEqual(len(m.group(1)), 40)

    def test_github_paths_exist_in_repo(self):
        for name, _w, _h, _s, gh, _r in cell_manifest():
            if gh:
                self.assertTrue((REPO / gh).is_file(), gh)
        self.assertIn("supervisor/maple_oneclick_restore.py", [f[4] for f in cell_manifest()])

    def test_no_second_commit_needed_for_the_pin(self):
        """The pinned commit is an existing commit that contains the two supervisor scripts; the one-click script is never requested from it."""
        src = CELL.read_text(encoding="utf-8")
        self.assertIn('IN_PINNED_COMMIT = ("maple_heartbeat.py", "maple_tunnel.py")', src)
        self.assertNotIn("0000000", src)

    def test_models_never_in_manifest(self):
        for f in cell_manifest():
            self.assertFalse(f[0].lower().endswith((".gguf", ".bin", ".safetensors")))


@unittest.skipUnless(sys.platform.startswith("linux"), "the script splits Kaggle paths on '/'; it runs on Linux only")
class DatasetGuard(unittest.TestCase):
    """safe_walk must never enter a Dataset that is not named maple* (for example an SSH-key Dataset)."""

    def setUp(self):
        self.oc = load_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "input"
        for d in ("datasets/someone/maple-bootstrap/sub", "datasets/someone/ssh-keys/private", "maple-direct", "other-data/deep"):
            (self.root / d).mkdir(parents=True)
        (self.root / "datasets/someone/maple-bootstrap/sub/a.gguf").write_bytes(b"GGUF")
        (self.root / "datasets/someone/ssh-keys/private/id_x").write_bytes(b"x")
        (self.root / "maple-direct/b.gguf").write_bytes(b"GGUF")
        (self.root / "other-data/deep/c.gguf").write_bytes(b"GGUF")
        self.oc.INPUT_ROOT = str(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_visits_only_maple_datasets(self):
        seen = [dp for dp, _dn, _fn in self.oc.safe_walk(str(self.root), 4)]
        joined = "\n".join(seen)
        self.assertIn("maple-bootstrap", joined)
        self.assertIn("maple-direct", joined)
        self.assertNotIn("ssh-keys", joined)
        self.assertNotIn("other-data", joined)

    def test_gguf_sources_are_maple_only(self):
        found = self.oc.find_gguf_sources(None)
        self.assertEqual(sorted(Path(p).name for p in found), ["maple-direct", "sub"])

    def test_explicit_folder_is_always_searched(self):
        found = self.oc.find_gguf_sources([str(self.root / "other-data")])
        self.assertTrue(any("deep" in p for p in found))


class ArtifactSearch(unittest.TestCase):
    def test_needs_checksums_file(self):
        oc = load_script()
        oc.SEARCH_ROOTS = []                     # hermetic: never look at a real /kaggle/working
        with tempfile.TemporaryDirectory() as t:
            for n in oc.ARCHIVES.values():
                (Path(t) / n).write_bytes(b"x")
            self.assertIsNone(oc.find_artifacts([t])[0])      # archives but no checksums.sha256 -> not accepted
            (Path(t) / "checksums.sha256").write_text("x")
            self.assertEqual(oc.find_artifacts([t])[0], t)


@unittest.skipUnless(sys.platform.startswith("linux"), "bootstrap cell uses os.uname (Linux/Kaggle)")
class CellSandbox(unittest.TestCase):
    """Run the cell against a temporary Dataset folder; never touches /kaggle."""

    def run_cell(self, src_dir, inst, gh_base="http://127.0.0.1:9/%s/", patch=None):
        c = CELL.read_text(encoding="utf-8")
        if patch:
            c = patch(c)
        c = c.replace('"/kaggle/input/%s" % DATASET_NAME', '"%s"' % src_dir)
        for k, v in (("INSTALL_DIR", inst + "/inst"), ("ARTIFACT_DIR", inst + "/art"), ("PROJECT_DIR", inst + "/proj")):
            c = re.sub(r'(%s\s*=\s*)"[^"]+"' % k, lambda m: m.group(1) + '"%s"' % v, c, count=1)
        c = c.replace("RUN_AFTER_INSTALL = True", "RUN_AFTER_INSTALL = False")
        c, n = re.subn(r'GH_BASE = "[^"]+"', lambda m: 'GH_BASE = "%s"' % gh_base, c, count=1)    # never the real network in tests
        self.assertEqual(n, 1)
        f = Path(inst) / "cell.py"
        f.write_text(c)
        return subprocess.run([sys.executable, "-B", str(f)], capture_output=True, text=True, timeout=120)

    def make_src(self, d, tamper=False):
        d.mkdir(parents=True)
        for name, _w, hashes, size, _gh, role in cell_manifest():
            data = {"maple_oneclick_restore.py": SCRIPT, "maple_heartbeat.py": SUP / "maple_heartbeat.py", "maple_tunnel.py": SUP / "maple_tunnel.py"}.get(name)
            if data:
                b = data.read_bytes()
                if tamper and name == "maple_oneclick_restore.py":
                    b = b[:100] + bytes([b[100] ^ 1]) + b[101:]      # flip one bit in the middle: the SHA256 must no longer match
                (d / name).write_bytes(b)

    def test_rejects_tampered_and_never_installs_it(self):
        with tempfile.TemporaryDirectory() as t:
            self.make_src(Path(t) / "src", tamper=True)
            self.assertNotEqual(sha(Path(t) / "src/maple_oneclick_restore.py"), sha(SCRIPT))
            r = self.run_cell(Path(t) / "src", t)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertFalse((Path(t) / "inst/maple_oneclick_restore.py").exists())
            self.assertIn("REJECTED", r.stdout)

    def test_installs_verified_files_and_keeps_user_edits(self):
        with tempfile.TemporaryDirectory() as t:
            self.make_src(Path(t) / "src")
            (Path(t) / "inst").mkdir()
            mine = b"# my own edit\n" + (SUP / "maple_heartbeat.py").read_bytes()
            (Path(t) / "inst/maple_heartbeat.py").write_bytes(mine)
            # the project is "missing" in this sandbox, so artifacts are required; provide only scripts -> expect exit 2 AFTER scripts are handled
            r = self.run_cell(Path(t) / "src", t)
            self.assertEqual((Path(t) / "inst/maple_heartbeat.py").read_bytes(), mine)
            self.assertIn("KEPT", r.stdout)
            self.assertEqual(sha(Path(t) / "inst/maple_oneclick_restore.py"), sha(SCRIPT))
            self.assertFalse(list(Path(t).rglob(".*.part")))


    # ------------------------------------------------------------------ self-contained: empty /kaggle/working + Dataset only
    def test_empty_session_is_fully_restored_from_the_dataset(self):
        """Nothing in the install dirs: every pinned file (incl. the one-click script and the archives) comes from the Dataset only."""
        dummies = {"MAPLE-llamacpp-cuda.tar.gz": b"A" * 1000, "MAPLE-runtime.tar.gz": b"B" * 500,
                   "MAPLE-python-deps.tar.gz": b"C" * 300, "checksums.sha256": b"D" * 100}

        def patch(c):                                   # pin the dummy archives instead of the 60 MB real ones
            for name, data in dummies.items():
                pat = r"\('%s', 'artifact', \('[0-9a-f]{64}',\), \d+, None, 'artifact'\)" % re.escape(name)
                c, n = re.subn(pat, "('%s', 'artifact', ('%s',), %d, None, 'artifact')" % (name, hashlib.sha256(data).hexdigest(), len(data)), c)
                self.assertEqual(n, 1, name)
            return c
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "src"
            self.make_src(src)
            for name, data in dummies.items():
                (src / name).write_bytes(data)
            r = self.run_cell(src, t, patch=patch)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(r.stdout.count("[OK     ]"), 7, r.stdout)
            for name in ("maple_oneclick_restore.py", "maple_heartbeat.py", "maple_tunnel.py"):
                self.assertEqual(sha(Path(t) / "inst" / name), sha(SUP / name), name)
            for name, data in dummies.items():
                self.assertEqual((Path(t) / "art" / name).read_bytes(), data, name)
            stamp = {p: p.stat().st_mtime_ns for p in Path(t).glob("inst/*")} | {p: p.stat().st_mtime_ns for p in Path(t).glob("art/*")}
            r2 = self.run_cell(src, t, patch=patch)             # idempotent: nothing rewritten, nothing duplicated
            self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
            self.assertEqual(r2.stdout.count("present, SHA256 verified"), 7)
            self.assertEqual(stamp, {p: p.stat().st_mtime_ns for p in stamp})
            self.assertFalse(list(Path(t).rglob(".*.part")))

    # ------------------------------------------------------------------ GitHub fallback over a LOCAL loopback server
    def serve(self, files, requests):
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                data = files.get(self.path)
                self.send_response(200 if data is not None else 404)
                self.end_headers()
                if data is not None:
                    self.wfile.write(data)

            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)       # runs first (cleanups are LIFO), then the socket is closed
        return srv.server_address[1]

    def test_github_fallback_order_and_hash_gate(self):
        pin = re.search(r'GITHUB_COMMIT\s*=\s*"([0-9a-f]{40})"', CELL.read_text(encoding="utf-8")).group(1)
        good = {"maple_oneclick_restore.py": SCRIPT.read_bytes(), "maple_heartbeat.py": (SUP / "maple_heartbeat.py").read_bytes(), "maple_tunnel.py": (SUP / "maple_tunnel.py").read_bytes()}
        files = {"/%s/supervisor/maple_heartbeat.py" % pin: good["maple_heartbeat.py"], "/%s/supervisor/maple_tunnel.py" % pin: good["maple_tunnel.py"],
                 "/main/supervisor/maple_oneclick_restore.py": good["maple_oneclick_restore.py"]}
        requests = []
        port = self.serve(files, requests)
        with tempfile.TemporaryDirectory() as t:
            r = self.run_cell(Path(t) / "no-dataset", t, gh_base="http://127.0.0.1:%d/%%s/" % port)
            for name, data in good.items():
                self.assertEqual((Path(t) / "inst" / name).read_bytes(), data, name)     # one-click script came from main, the others from the pinned commit
            self.assertEqual(r.returncode, 2)                                          # the archives are not on GitHub: named exactly, nothing else broken
            self.assertIn("MAPLE-llamacpp-cuda.tar.gz", r.stdout)
        self.assertNotIn("/%s/supervisor/maple_oneclick_restore.py" % pin, requests)    # never asks the pinned commit for a file it does not contain
        self.assertIn("/main/supervisor/maple_oneclick_restore.py", requests)

    def test_older_tunnel_from_the_pinned_commit_is_accepted(self):
        new = (SUP / "maple_tunnel.py").read_bytes()
        old = new.replace(OLD_MSG.encode(), STALE_MSG.encode())
        pin = re.search(r'GITHUB_COMMIT\s*=\s*"([0-9a-f]{40})"', CELL.read_text(encoding="utf-8")).group(1)
        files = {"/%s/supervisor/maple_tunnel.py" % pin: old, "/%s/supervisor/maple_heartbeat.py" % pin: (SUP / "maple_heartbeat.py").read_bytes(),
                 "/main/supervisor/maple_oneclick_restore.py": SCRIPT.read_bytes()}
        port = self.serve(files, [])
        with tempfile.TemporaryDirectory() as t:
            self.run_cell(Path(t) / "no-dataset", t, gh_base="http://127.0.0.1:%d/%%s/" % port)
            self.assertEqual((Path(t) / "inst/maple_tunnel.py").read_bytes(), old)       # the older, pinned-commit copy passes the gate unchanged

    def test_github_content_that_does_not_match_the_pin_is_rejected(self):
        bad = SCRIPT.read_bytes()
        bad = bad[:100] + bytes([bad[100] ^ 1]) + bad[101:]
        requests = []
        port = self.serve({"/main/supervisor/maple_oneclick_restore.py": bad}, requests)
        with tempfile.TemporaryDirectory() as t:
            r = self.run_cell(Path(t) / "no-dataset", t, gh_base="http://127.0.0.1:%d/%%s/" % port)
            self.assertFalse((Path(t) / "inst/maple_oneclick_restore.py").exists())
            self.assertIn("REJECTED", r.stdout)
            self.assertIn("maple_oneclick_restore.py", r.stdout.split("FAILED:")[1])


if __name__ == "__main__":
    unittest.main()
