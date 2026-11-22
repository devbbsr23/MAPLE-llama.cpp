# =====================================================================================================
# MAPLE ONE-CELL BOOTSTRAP + RESTORE  -  run this ONE cell at the start of every new Kaggle session.
# Enable the GPU first (Settings > Accelerator > GPU T4 x2). Optional: add your Dataset (Input panel) - see DATASET_NAME.
# What it does: finds the MAPLE scripts/archives (already in /kaggle/working, else your Dataset, else GitHub), checks SHA256
# against the manifest pinned BELOW before anything is written or run, never overwrites an existing file, never downloads
# models, then runs maple_oneclick_restore.py (restore if needed -> models -> heartbeat -> MAPLE API -> /health, /v1/models).
# It never starts the authentication proxy and never touches SSH/Cloudflare. No credentials are used or printed.
# Documentation: docs/KAGGLE_BOOTSTRAP.md   |   Pinned hashes below: notebooks/BOOTSTRAP_MANIFEST.sha256
# =====================================================================================================
import glob, hashlib, os, subprocess, sys, time, urllib.request
from pathlib import Path

# ----------------------------------- CONFIGURATION (normally no edit needed) -----------------------------------
DATASET_NAME      = "maple-bootstrap-multi-20261122"   # slug of YOUR Kaggle Dataset holding the files below (used only if a file is missing).
                                                  # The original single-model Dataset is "maple-bootstrap" (rollback; set this name back to use it, with the old pins).
GITHUB_COMMIT     = "e2be404016c3a0c66f79af1984d826c67f9d80d2"   # commit that already contains maple_heartbeat.py / maple_tunnel.py (tried first for those two)
INSTALL_DIR       = "/kaggle/working/MAPLE_compatible"
ARTIFACT_DIR      = "/kaggle/working/maple_artifacts_flat"
PROJECT_DIR       = "/kaggle/working/MAPLE-llama.cpp"
RUN_ARGS          = []                     # extra options for maple_oneclick_restore.py, e.g. ["--check-only"] or ["--sha"] (slow)
RUN_AFTER_INSTALL = True

# ----------------------------------- TRUSTED MANIFEST (pinned in this cell) -----------------------------------
# (file, where, accepted SHA256 values, size in bytes (or a tuple of accepted sizes), path in the GitHub repo or None, role)
# maple_tunnel.py accepts 3 versions on purpose: the current repository copy, the copy at the pinned GitHub commit, and the older copy already deployed on Kaggle.
FILES = [
 ('maple_oneclick_restore.py', 'install', ('25afb7ca823d659474b7aa56a8e786800f06eb0582535d55e30710041438b0c3',), 34883, 'supervisor/maple_oneclick_restore.py', 'required'),
 ('maple_heartbeat.py', 'install', ('8af6007bba5d6970567e4b9e33f0b8018cf616c2188e2e5b8073c6fcf532061d',), 53241, 'supervisor/maple_heartbeat.py', 'required'),
 ('maple_tunnel.py', 'install', ('1a14583571b3d67d137cdc2e18d90c82c1debf0efeca08e962fa8af707cfd99f', '42f17d0985eed27a95bce29806641b4f55a42ae25ecad91c302cd2be1841fe80', 'b255bce5d806124bf646487a51b325cf53b7f98bf72d41fd19282755cbfe3a75'), (62747, 62699), 'supervisor/maple_tunnel.py', 'optional'),
 ('MAPLE-llamacpp-cuda.tar.gz', 'artifact', ('0be6e25deedb37c2ab01fd53c761a7efcacddd3ff37d6713e4787bd9062a1b49',), 60260524, None, 'artifact'),
 ('MAPLE-runtime.tar.gz', 'artifact', ('81c54fa92e39dbc3b8df3b3d6840bfb7f95bf0c4f55f2d1e23f8fd1988c9774e',), 178703, None, 'artifact'),
 ('MAPLE-python-deps.tar.gz', 'artifact', ('1c1d0a3f99c95d3a27b0e5b7aef9d0fa9531bf679351803ee14e4d87f70e3b1b',), 796586, None, 'artifact'),
 ('checksums.sha256', 'artifact', ('85922584568fb08d5ef4e563dccc101fcb44a637e50705ae25308729bfc859f8',), 271, None, 'artifact'),
]
# --------------------------------------------------------------------------------------------------------------
GH_BASE = "https://raw.githubusercontent.com/devbbsr23/MAPLE-llama.cpp/%s/"   # %s = commit or branch. Safe with a branch: nothing is written or run unless its size and SHA256 match the pins above.
IN_PINNED_COMMIT = ("maple_heartbeat.py", "maple_tunnel.py")                     # files that exist at GITHUB_COMMIT; everything else is read from main
BASE = {"install": INSTALL_DIR, "artifact": ARTIFACT_DIR}


def sha_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            d.update(b)
    return d.hexdigest()


def dataset_roots():
    out = []
    for pat in ("/kaggle/input/%s" % DATASET_NAME, "/kaggle/input/datasets/*/%s" % DATASET_NAME, "/kaggle/input/*/%s" % DATASET_NAME):
        out += [r for r in glob.glob(pat) if os.path.isdir(r)]
    return list(dict.fromkeys(out))


def fetch(name, ok_hashes, size, ghpath, notes):
    """Return (bytes, origin) from the first source whose content matches the pinned SHA256 + size; record rejections."""
    sizes = size if isinstance(size, tuple) else (size,)
    cands = [("Dataset " + r, (lambda r=r: Path(r, name).read_bytes()), Path(r, name).is_file()) for r in dataset_roots()]
    if ghpath:
        for ref in ([GITHUB_COMMIT] if name in IN_PINNED_COMMIT else []) + ["main"]:
            cands.append(("GitHub @" + ref[:7], (lambda ref=ref: urllib.request.urlopen(urllib.request.Request(GH_BASE % ref + ghpath, headers={"User-Agent": "maple-bootstrap"}), timeout=60).read(max(sizes) + 1)), True))
    for origin, getter, exists in cands:
        if not exists:
            continue
        try:
            data = getter()
        except Exception as e:  # noqa
            notes.append("%s: %s from %s" % (name, type(e).__name__, origin))
            continue
        if len(data) in sizes and hashlib.sha256(data).hexdigest() in ok_hashes:
            return data, origin
        notes.append("%s REJECTED from %s: SHA256/size mismatch (not written, not executed)" % (name, origin))
    return None, None


def ensure(name, where, ok_hashes, size, ghpath, notes):
    dest = Path(BASE[where], name)
    if dest.is_file():
        if sha_file(dest) in ok_hashes:
            return "OK", "present, SHA256 verified"
        return "KEPT", "present but differs from the pinned version: YOUR file was left untouched (not overwritten)"
    data, origin = fetch(name, ok_hashes, size, ghpath, notes)
    if data is None:
        return "MISSING", "no verified copy: not in /kaggle/working, and no source matched the pinned SHA256 (Dataset '%s'%s); see notes below" % (DATASET_NAME, ", GitHub" if ghpath else "; this file is not on GitHub")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(".%s.part" % name)
    tmp.write_bytes(data)
    try:
        os.link(tmp, dest)                  # fails if the file appeared meanwhile: never overwrites
    except FileExistsError:
        return "KEPT", "appeared while downloading; left untouched"
    finally:
        tmp.unlink()
    os.chmod(dest, 0o644)
    return "OK", "installed from %s after SHA256 check (%d bytes)" % (origin, len(data))


print("=" * 78 + "\nMAPLE BOOTSTRAP\n" + "=" * 78)
_smi = subprocess.run("LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH $(command -v nvidia-smi || echo /opt/bin/nvidia-smi) -L", shell=True, capture_output=True, text=True).stdout
print("host %s | python %s | GPUs visible: %d" % (os.uname().nodename, sys.version.split()[0], _smi.count("GPU ")))
roots = dataset_roots()
print("Dataset '%s' mounted: %s" % (DATASET_NAME, roots[0] if roots else "no (fine if everything is already in /kaggle/working)"))
need_restore = not (Path(PROJECT_DIR, "scripts/start_maple.sh").is_file() and os.access(Path(PROJECT_DIR, "vendor/llamacpp/llama.cpp/build/bin/llama-server"), os.X_OK))
print("MAPLE restore needed (project or executable llama-server missing): %s" % need_restore)

notes, failed, rows = [], [], []
for name, where, ok_hashes, size, ghpath, role in FILES:
    if role == "artifact" and not need_restore and not Path(BASE[where], name).exists():
        rows.append(("SKIP", name, "not needed (project already restored)"))
        continue
    st, detail = ensure(name, where, ok_hashes, size, ghpath, notes)
    rows.append((st, name, detail))
    if st == "MISSING" and role in ("required", "artifact"):
        failed.append(name)
    if st == "MISSING" and role == "optional":
        rows[-1] = ("WARN", name, detail + " (optional now; needed later for the authentication proxy)")
for st, name, detail in rows:
    print("  [%-7s] %-28s %s" % (st, name, detail))
for n in notes:
    print("  note:", n)
if failed:
    print("\nFAILED: required files are unavailable: %s" % ", ".join(failed))
    print("Fix: attach your Kaggle Dataset '%s' to this notebook (Input panel > Add Input > Your Datasets) and run this cell again;" % DATASET_NAME)
    print("     the Dataset must contain: " + ", ".join(f[0] for f in FILES) + ". Nothing was changed or started.")
    sys.exit(2)

script = Path(INSTALL_DIR, "maple_oneclick_restore.py")
if not RUN_AFTER_INSTALL:
    print("\nInstall/verify finished; RUN_AFTER_INSTALL is False, so the restore was not run.")
else:
    print("\n" + "=" * 78 + "\nRUNNING %s %s\n" % (script.name, " ".join(RUN_ARGS)) + "=" * 78, flush=True)
    p = subprocess.Popen(["python3", "-B", str(script)] + RUN_ARGS, cwd=INSTALL_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    print("\nexit code: %d -> %s" % (rc, "READY FOR AUTHENTICATION SETUP" if rc == 0 else "FAILED: read the FAILED stage above, fix it, run this cell again (safe to repeat)"))
    if rc != 0:
        sys.exit(rc)
