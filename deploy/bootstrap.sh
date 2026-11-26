#!/usr/bin/env bash
set -euo pipefail
cd /opt/oneocr_linux
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q onnxruntime numpy pillow
python - <<'PY'
import onnxruntime as o, numpy as n, PIL, sys
print("python", sys.version.split()[0])
print("ort", o.__version__)
print("numpy", n.__version__)
print("pillow", PIL.__version__)
PY