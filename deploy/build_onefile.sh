#!/usr/bin/env bash
# Build a self-contained one-file binary. Usage: build_onefile.sh [project_root]
set -euo pipefail
ROOT="${1:-/opt/oneocr_linux}"
cd "$ROOT"
. .venv/bin/activate

STAMP=$(date +%H%M%S)
OUT="dist_pkg_$STAMP"
WORK="build_pkg_$STAMP"

python -m pip install -q pyinstaller

python -m PyInstaller --noconfirm --onefile --name oneocr-linux \
  --add-data "$PWD/models/onnx:models/onnx" \
  --collect-all onnxruntime \
  --collect-data oneocr_linux \
  --distpath "$OUT" --workpath "$WORK" --specpath "$WORK" \
  packaging/entry.py

ls -lh "$OUT/oneocr-linux"
echo "BIN=$OUT/oneocr-linux"