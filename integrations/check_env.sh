#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/ocr_bench
tar -xzf bundle.tar.gz
ls -1
V=/home/ubuntu/TGBotNet/ocr/venv/bin/python
"$V" - <<'PY'
import numpy, cv2, onnxruntime, PIL
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
print("ort", onnxruntime.__version__)
print("PIL", PIL.__version__)
try:
    import rapidocr_onnxruntime as r
    print("rapidocr", getattr(r, "__version__", "?"))
except Exception as e:
    print("rapidocr missing:", e)
PY