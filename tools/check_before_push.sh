#!/usr/bin/env bash
# Run before pushing: fast tests (same as CI) + tests that run the Cellpose model on real images.
#   tools/check_before_push.sh          # everything (~1-2 min on a GPU, much longer on CPU)
#   tools/check_before_push.sh --fast   # only the fast tests that CI runs
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

echo "== Fast tests (what GitHub CI runs)"
"$PY" -m pytest -q

if [ "${1:-}" = "--fast" ]; then
    exit 0
fi

echo
echo "== Model tests: Cellpose on real images, compared with manual measurements and a reviewed run"
"$PY" -m pytest -m slow -q -p no:warnings
echo
echo "All checks passed."
