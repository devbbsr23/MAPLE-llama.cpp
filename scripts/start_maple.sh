#!/usr/bin/env bash
# Start the MAPLE OpenAI-compatible API (models load on demand, one at a time).
# Uses the restored llama.cpp; LD_LIBRARY_PATH is set for this process only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAPLE_HOME="${MAPLE_HOME:-$HERE}"
export PYTHONPATH="$MAPLE_HOME/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$MAPLE_HOME/maple.env" ] && source "$MAPLE_HOME/maple.env"
exec python3 -s -m maple serve "$@"
