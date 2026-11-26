#!/usr/bin/env bash
# Switch the TGBot OCR service from RapidOCR to oneocr_linux (reversible).
set -euo pipefail
OCR=/home/ubuntu/TGBotNet/ocr
BENCH=/home/ubuntu/ocr_bench
STAMP=$(date +%Y%m%d%H%M%S)

cp -a "$OCR/ocr_server.py" "$OCR/ocr_server.py.bak-$STAMP"
cp -a "$BENCH/oneocr_linux" "$OCR/"
mkdir -p "$OCR/models"
cp -a "$BENCH/models/onnx" "$OCR/models/"
cp -a "$BENCH/oneocr_compat.py" "$OCR/"

python3 "$BENCH/patch_online.py"

echo "--- files ---"
ls -lh "$OCR" | grep -E "oneocr|models|ocr_server.py$"

echo "--- manual smoke (port 5557, oneocr engine) ---"
cd "$OCR"
OCR_ENGINE=oneocr ONEOCR_MODELS="$OCR/models/onnx" \
  "$OCR/venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, "/home/ubuntu/TGBotNet/ocr")
import ocr_server as s
s.ENGINE = s.make_engine()
data = open("/home/ubuntu/ocr_bench/imgs/engine-image-37cc3551854df89ca09104e272018679abdc223c90e0d9b405ef11bff3fcb8cd.jpg", "rb").read()
text, cands = s._recognize_locked(data)
print("lines:", len(text.splitlines()))
print(text[:300])
PY