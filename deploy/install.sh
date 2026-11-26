#!/usr/bin/env bash
# Install oneocr-linux into a fresh venv and run a smoke test.
# Usage: bash deploy/install.sh [--skip-smoke]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

echo "== python: $($PY -V)"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip >/dev/null
python -m pip install -e .

echo "== onnxruntime: $(python -c 'import onnxruntime as o; print(o.__version__)')"
echo "== numpy:       $(python -c 'import numpy as n; print(n.__version__)')"

if [ ! -f models/onnx/detector.onnx ]; then
  echo "WARNING: models/onnx/detector.onnx missing; copy the extracted graphs first." >&2
  exit 2
fi

if [ "${1:-}" != "--skip-smoke" ]; then
  echo "== smoke test"
  time oneocr-linux samples/sample.png
fi