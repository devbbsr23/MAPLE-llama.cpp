# Kaggle one-cell bootstrap and restore

Goal: in a new Kaggle session, run **one notebook cell** and end with MAPLE restored, the heartbeat running, the MAPLE API healthy on `127.0.0.1:8000`, and `/health` and `/v1/models` checked. The authentication proxy is **not** started by this procedure; that is the next step (see section 8).

Evidence labels follow the operating guide ([MAPLE_COMPLETE_OPERATING_GUIDE.md](MAPLE_COMPLETE_OPERATING_GUIDE.md), section 0): **[LIVE-VERIFIED]** ran for real on a Kaggle session, **[UNIT-TESTED]** covered by the automated tests only, **[UNVERIFIED]** never executed. Section 11 lists each one.

## 1. The three pieces

| Piece | File | Role |
|---|---|---|
| Bootstrap cell | `notebooks/maple_bootstrap_cell.py` | The only thing you paste. Finds the files below, checks each against a pinned SHA256, installs only what is missing, then runs the restore script. |
| Restore script | `supervisor/maple_oneclick_restore.py` | 13 stages: session, GPU, disk, artifacts, restore (only if needed), models, dependencies, `maple.env`, llama-server, heartbeat, API readiness, `/health` + `/v1/models`, processes and ports. |
| Supervisor | `supervisor/maple_heartbeat.py` | Started by the restore script. It starts and watches the MAPLE API. |

`supervisor/maple_tunnel.py` (the authentication proxy) is only copied into place; nothing starts it.

## 2. Persistence: what was observed versus what is guaranteed

| | Statement | Basis |
|---|---|---|
| **Observed** (one session, 2026-11-21) | Everything under `/kaggle/working` (project, scripts, archives, the four models) was already present when the notebook started, with its old file times. | Looked at, once. **[LIVE-VERIFIED]** for that session only. |
| **Observed** (an earlier session) | A snapshot had stripped the execute permission bits from the `llama-server` build; the restore script repaired them. | **[LIVE-VERIFIED]** |
| **Not guaranteed** | That `/kaggle/working` is kept in any other session. It depends on how Kaggle saved that notebook and on Kaggle's own behaviour, which this project does not control. The repository README states the opposite default (it does not survive a restart). | Treat as a convenience, never as a dependency. |
| **Not persistent** | `/tmp`, running processes, lock files, and the Cloudflare tunnel (every new session gets a new host name). | Kaggle sessions are ephemeral. |
| **What this procedure depends on** | The private Dataset `maple-bootstrap-multi-20261122` (stored by Kaggle until you delete it) and the public repository. | Attaching the Dataset in a *fresh* session has **not** been tested here **[UNVERIFIED]**; the same logic was tested with a local folder standing in for it **[UNIT-TESTED]**. |

So the design assumes `/kaggle/working` may be completely empty: the one-click script, both supervisor scripts and the restore archives are all recoverable from the Dataset alone. Only the four models are outside it (section 7).

## 3. One-time setup: the private `maple-bootstrap-multi-20261122` Dataset

Do this once, on the Kaggle website (it needs your own login; nothing here uses or stores credentials).

1. Collect these files in one folder. `notebooks/BOOTSTRAP_MANIFEST.sha256` lists them with their SHA256:
   - `maple_oneclick_restore.py`, `maple_heartbeat.py`, `maple_tunnel.py` (from `supervisor/`)
   - `MAPLE-llamacpp-cuda.tar.gz`, `MAPLE-runtime.tar.gz`, `MAPLE-python-deps.tar.gz`, `checksums.sha256` (release assets, see [RESTORATION.md](RESTORATION.md))
2. Verify them before upload: `sha256sum -c BOOTSTRAP_MANIFEST.sha256` (Linux/Git Bash), or compare with `Get-FileHash -Algorithm SHA256` in PowerShell.
3. Kaggle > Datasets > New Dataset. Select the files, name it **`maple-bootstrap-multi-20261122`** (the URL slug must be exactly that), set it **Private**, create it.
4. **Never put model files, API keys, SSH keys, tokens or Kaggle credentials in this Dataset.** Models stay where they are (section 7).

## 4. Every new session

1. Settings > Accelerator > **GPU T4 x2**.
2. Input panel > Add Input > Your Datasets > `maple-bootstrap-multi-20261122`. (Not required while `/kaggle/working` still has the files, but required for a truly empty session.)
3. Paste the contents of `notebooks/maple_bootstrap_cell.py` into one cell and run it.

Options (edit the `RUN_ARGS` line in the cell): `["--check-only"]` changes nothing and starts nothing; `["--sha"]` also recomputes the model hashes (reads about 16 GB, several minutes); `["--force-restore"]` re-runs `restore_kaggle.py`.

## 5. What the cell does, in order

1. Prints the host, Python version and the number of GPUs visible.
2. For each pinned file: if it already exists and matches, it is left alone; if it exists but **differs**, it is **kept untouched** and reported (your edits are never overwritten); if it is missing, it is taken from the attached Dataset, otherwise (the three scripts only; the archives are not in Git) from GitHub: first the pinned commit, for the two supervisor scripts it contains, then branch `main`. A branch is safe here because a candidate is written **only after** its size and SHA256 match the pinned values, whatever source or ref delivered it. A rejected candidate is never written or run.
3. Installs with a hard link, so a file that appears meanwhile is not overwritten.
4. If the project or an executable `llama-server` is missing, the archives are also required; otherwise they are not touched.
5. Runs `maple_oneclick_restore.py`, which stops at the first failed stage with an exit code (2 artifacts missing, 3 environment or GPU or disk, 4 restore, 5 start) and says exactly what is missing.

Nothing is compiled, no model is downloaded, no authentication proxy, SSH or Cloudflare process is started or stopped.

## 6. How the restore script works

- **Restore stage:** runs only when the project or `llama-server` is missing or not executable. It calls `scripts/restore_kaggle.py --no-sha --skip-smoke --skip-api` on archives whose `checksums.sha256` verify, never with `--overwrite-source`, so modified project files are kept and reported. It re-applies the archive's execute permissions (62 modes in the live run).
- **Artifact search:** `/kaggle/working/maple_artifacts_flat`, `/kaggle/working/artifacts`, `/kaggle/working`, and inside `/kaggle/input` **only Dataset folders whose name starts with `maple`**. Other Datasets (for example one holding SSH keys) are never opened. A differently named Dataset can be passed explicitly with `--artifacts DIR` or `--models-source DIR`.
- **Models stage:** checks that each of the four files exists, has a valid GGUF header and the manifest size. If one is missing and a `maple*` Dataset holds `.gguf` files, `scripts/link_models.py --apply` creates symlinks only. Models are never copied, downloaded or replaced.
- **Heartbeat stage:** starts `maple_heartbeat.py run --maple-home ... --state-dir ...` detached **only if no supervisor holds the lock**. The heartbeat itself starts the MAPLE API with `scripts/start_maple.sh --host 127.0.0.1 --port 8000` and keeps it alive.
- **Readiness:** waits up to 600 s for `/health` HTTP 200, then requests `/health` and `/v1/models` and requires the roster `deepseek, gemma, qwen3, fast, auto`. No model is loaded at this point; models load on the first request.
- **Safety:** the API host is fixed to `127.0.0.1` in the script; running it twice finds the existing heartbeat and API and starts nothing.

## 7. If something is missing

| Report | Meaning and fix |
|---|---|
| `MISSING ... no verified copy` | The file is not in `/kaggle/working`, not in an attached `maple*` Dataset, and (for scripts) GitHub did not deliver a matching copy. Attach `maple-bootstrap-multi-20261122` and run the cell again. |
| `REJECTED ... SHA256/size mismatch` | A source file differs from the pinned version. It was not written. Re-upload the staged file; never edit the pinned hash to make it pass unless you changed the file on purpose. |
| `KEPT ... YOUR file was left untouched` | You modified an installed file. Fine for running; update the pin in the cell if you want it to count as verified. |
| Stage 4 `FAILED` | A required directory or script is missing; the message names it. |
| Stage 6 `MISSING` model | The exact paths are listed. Attach a Dataset that holds your models (name starting `maple`) or place the files yourself. They are not downloadable by this procedure, by design. |
| Restore stage exit 4 | CUDA 13 libraries, NCCL, glibc or GPU architecture incompatible; see [RESTORATION.md](RESTORATION.md). Nothing is built or substituted. |
| Only 1 GPU reported | The cell says so. Two-GPU operation is then not available. |

If neither the Dataset nor GitHub is available, rebuild the Dataset from the repository (`supervisor/`) and the release archives, as in section 3.

## 8. Next step: the authentication proxy

After `READY FOR AUTHENTICATION SETUP`, follow the operating guide: generate the key on Windows and start the proxy with `maple_tunnel.py start --key-stdin` (section 7.3), verify it with the port-listener check (section 7.5), and forward 8088 from Windows (section 8). The key is never part of the bootstrap.

## 9. Changed hostnames and new sessions

A new Kaggle session gets a **new Cloudflare host name**. The bootstrap does not create or change the SSH access; run your existing SSH setup cell, take the host name it reports, set it as `HostName` in the `kaggle2` block of `%USERPROFILE%\.ssh\config`, and test with `ssh -v kaggle2`. The first connection to a new host fails host-key checking until you compare the key: on Kaggle run `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`, compare the fingerprint with the one `ssh` shows, and only then accept it. Details: guide section 3. The container name printed by the bootstrap is **not** the Cloudflare host name.

## 10. Updating the pinned hashes

If you change a script, recompute its SHA256, then update three places together: the `FILES` list in the cell, `notebooks/BOOTSTRAP_MANIFEST.sha256` and `supervisor/SHA256SUMS.txt`. The tests in `supervisor/tests/test_maple_oneclick_restore.py` fail if they disagree.

`maple_tunnel.py` is pinned with three accepted versions on purpose: the current repository copy, the older copy served by the pinned commit, and the older copy already deployed on Kaggle (they differ only by the version string and one sentence of help text). Any other content is rejected.

**Why no second commit is ever needed for the pin.** `GITHUB_COMMIT` is an existing commit (`e2be404...`) that already contains `maple_heartbeat.py` and `maple_tunnel.py`; it never has to name the commit that adds the bootstrap. The one-click script is read from `main`, and the SHA256 pin, not the ref, is what makes that safe: a `main` that serves a different file is rejected and the cell falls back to the Dataset. If you later change a script, update the three places above; the pinned commit can stay as it is.

## 11. What is and is not verified

**[LIVE-VERIFIED]** on a real two-T4 Kaggle session (2026-11-21):
- the restore script in `--check-only` and real mode against the running install (13 stages, exit 0, no process changed, no duplicates);
- the bootstrap cell on the production paths (all pinned files verified, restore skipped as not needed, exit 0);
- the GitHub fallback fetching `maple_heartbeat.py` and `maple_tunnel.py` at the pinned commit, byte-identical (anonymous, live network);
- the final cell, installed into a temporary folder and run in `--check-only` mode against the real project: all pinned files verified, 11 stages SUCCESS, exit 0, no running process and no script changed;
- the model hashes of the four models against `manifests/models_manifest.json`.

**[UNIT-TESTED]** (temporary folders; no live service): hash pins match the repository files, tampered files are rejected and never written, user-modified files are kept, installs are idempotent with no leftover temp files, an **empty install directory is fully restored from a Dataset-style folder alone** (scripts and archives; dummy archives stand in for the 60 MB real ones), the GitHub fallback order (pinned commit for the two scripts, `main` for the one-click script, never a file from a commit that lacks it) and its hash gate (checked against a local loopback server, no real network), the Dataset guard never enters non-`maple*` Datasets, archives without `checksums.sha256` are not accepted, no secrets or local paths in the files. 21 tests (Linux: all; Windows: 13 run, 8 skipped as Linux-only).

**[UNVERIFIED]**:
- Installing from a **real attached Dataset**: only a local folder standing in for it was tested.
- The restore stage and the heartbeat start inside a **truly empty** session: each part was proven separately, never as one run from nothing.
- GitHub delivery of `maple_oneclick_restore.py` from `main`: it works only once this file has been pushed; before that GitHub answers 404 and the cell correctly falls back to the Dataset (seen live, exit 2 when no Dataset is attached). The logic is unit-tested against a local server only.
- `--sha` and `--force-restore` end to end.

## 12. Known limitations

- One shared Dataset slug; the cell looks for `maple-bootstrap-multi-20261122` only.
- The cell trusts the hashes pinned inside itself. Verify the cell itself (its SHA256 is not self-checked).
- Time: about 1 s when everything is already running (measured). A cold start of the API is estimated at 1-3 minutes (not measured end to end).
- The older `notebooks/MAPLE_Kaggle_Restore.ipynb` flow does not start the heartbeat and remains available.


## Multi-model Dataset (2026-11-22)
The multi-model release uses the Dataset **`maple-bootstrap-multi-20261122`** (runtime archive built from the branch `feature/multi-resident-routing`). The original single-model Dataset `maple-bootstrap` is kept untouched as the rollback: to roll back, attach only that Dataset and set `DATASET_NAME = "maple-bootstrap"` and the old pins in the cell (commit `41d9b07`). Attach only ONE of the two Datasets in a session. The four GGUF model files are NOT in either Dataset; they need their own Dataset whose slug starts with `maple` (for example `maple-models`) or another way to place them at the exact roster paths (see RECOVERY_GUIDE.md section 2).

## Fresh-session procedure with the bootstrap archive (2026-11-22)
1. You start SSH, then Cloudflare (your own cells). 2. From Windows: `scripts\push_bootstrap.ps1` in the maple_client environment copies
`maple-bootstrap-multi-20261122.tar.gz` to `/kaggle/working/` and verifies its SHA-256 on both sides (skip if it is already there).
3. Run `notebooks/maple_bootstrap_cell.py` as ONE cell. It verifies the archive against `ARCHIVE_SHA256`, extracts it to `/kaggle/working/.maple_bootstrap_stage`,
installs the pinned scripts, restores MAPLE (four resident models, GPU 0 gemma+deepseek, GPU 1 qwen3+fast, qualification-aware routing), starts the
heartbeat/API, asks for the proxy key (hidden; or `MAPLE_API_KEY`) and runs `maple_verify_multi.py`. Safe to run again. The model GGUF files are NOT in the archive:
they must already be at the roster paths under `/kaggle/working/models` (or be attached as a Dataset and linked); the cell never downloads them.
