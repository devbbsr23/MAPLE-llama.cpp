# Contributing

Thanks for helping. Ground rules, which reflect how this project works:

* **No compilation of llama.cpp in the normal path, and no model files in Git.** Changes must not add a build step, a downloader or a model.
* Keep the runtime on the Python standard library (PyYAML stays optional).
* **Do not document what you have not run.** Docs distinguish tested behaviour from expectations; keep that distinction.
* Add tests for behaviour changes. Run them before opening a pull request:
  ```bash
  python -m unittest discover -s tests          # no GPU needed; GPU tests are skipped
  MAPLE_LIVE=1 python -m unittest discover -s tests   # only on a host with the restored build and the four models
  python scripts/verify_local_recovery.py       # if you maintain a local recovery folder
  ```
* Never commit secrets, local paths, tunnel URLs, `maple.env`, archives (`*.tar.gz`), binaries (`*.so`) or `*.gguf`.
* Licensing: by contributing you agree your contribution is under Apache-2.0. Keep `LICENSE` and `NOTICE`; do not copy code from
  incompatible licenses, and note third-party additions in `NOTICE`.
* Commit messages: short imperative subject, details in the body. Update `CHANGELOG.md` for user-visible changes.
