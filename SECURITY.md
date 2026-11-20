# Security

## Supported versions
Only the latest release (`main`) receives fixes.

## Reporting a vulnerability
Please report privately through GitHub's "Report a vulnerability" (Security advisories) on this repository, not in a public issue.
Include the version, how to reproduce and the impact.

## Things to know
* **The API has no authentication.** It ignores the `Authorization` header and binds to `127.0.0.1` by default. Do not expose it to
  the internet (`--host 0.0.0.0`, public tunnels) without an authenticating proxy.
* Never commit tokens, tunnel URLs, SSH keys, `.env` / `maple.env` files or model files. `.gitignore` excludes the common cases; check `git status` before every commit.
* The restore script refuses archives containing `*.gguf` files or path traversal and verifies SHA256 before extracting; the archives
  contain executable binaries, so only use archives whose checksums you have verified against a source you trust.
* Models are third-party data and are loaded by llama.cpp; use files from sources you trust.
