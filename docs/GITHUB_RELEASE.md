# Publishing this repository (maintainers)

Nothing here is automatic. Publishing and releasing are separate, deliberate steps. Use **your own** repository; never push to the
upstream project `AmSach/maple` (it is only a reference).

## 1. Check before the first push

```bash
git status --short                      # only source, docs, manifests, scripts, config, notebooks, tests
git ls-files | grep -Ei '\.(gguf|so|tar\.gz|exe|dll|env)$' || echo "no models/binaries/archives tracked"
git ls-files | xargs grep -InE 'ghp_|hf_[A-Za-z0-9]{10}|BEGIN (RSA|OPENSSH)|trycloudflare' || echo "no secrets found"
python -m unittest discover -s tests
python scripts/verify_local_recovery.py      # if you keep a local recovery folder with the archives
```

## 2. Create the repository and push `main`

With GitHub CLI (`gh auth login` first):
```bash
gh repo create MAPLE-llama.cpp --public --source . --remote origin --description "Portable multi-model llama.cpp inference with an OpenAI-compatible API"
git push -u origin main
```
or on github.com create an empty repository (no README/licence), then:
```bash
git remote add origin https://github.com/<you>/MAPLE-llama.cpp.git
git push -u origin main
```
Use `--private` first if you want to look before making it public (`gh repo edit --visibility public --accept-visibility-change-consequences` later).

## 3. Versioned release with the four archives

Verify the archives and checksums first (`artifacts/README.md`), then:
```bash
git tag -a v0.1.0 -m "MAPLE-llama.cpp 0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "MAPLE-llama.cpp 0.1.0" --notes-file CHANGELOG.md \
  artifacts/MAPLE-llamacpp-cuda.tar.gz artifacts/MAPLE-runtime.tar.gz artifacts/MAPLE-python-deps.tar.gz \
  artifacts/MAPLE-complete.tar.gz artifacts/checksums.sha256
```
`MAPLE-complete.tar.gz` is not kept in the local recovery folder by default; fetch it with `python scripts/pull_artifacts.py --include-complete`
or regenerate it on the host that has the compiled distribution. Each asset must stay under GitHub's per-file limit (2 GiB); these are far below it.
The release redistributes compiled llama.cpp (MIT, licence included in the archive). Do **not** attach models or the original shared archive.

## Provenance recorded in the repository

* Upstream reference: `AmSach/maple` commit `5403b6f5ec3f904409d968ff6a3ea3a2b22767d1` (Apache-2.0), `NOTICE`.
* llama.cpp commit `ef3848f`, CUDA 13.0, sm_75: `manifests/build_manifest.json`.
* Artifact sizes and SHA256: `artifacts/MANIFEST.json`, `manifests/artifacts_manifest.json`.
* Model metadata (sizes, SHA256, GGUF architecture, tested contexts; no model data): `manifests/models_manifest.json`.
