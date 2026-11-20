#!/usr/bin/env bash
# Restore MAPLE from TAR.GZ artifacts in a fresh Kaggle session. Does not compile,
# download models, or install CUDA. Usage: scripts/restore_kaggle.sh [restore_kaggle.py options]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "python3 not found" >&2; exit 6; }
exec "$PY" -I "$HERE/restore_kaggle.py" "$@"
