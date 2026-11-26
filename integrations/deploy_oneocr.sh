#!/usr/bin/env bash
# Deploy oneocr engine into one TGBot-family OCR service (reversible).
# Usage: deploy_oneocr.sh <ocr_dir> [port]
set -euo pipefail
OCR="${1:?usage: deploy_oneocr.sh <ocr_dir> [port]}"
PORT="${2:-5555}"
BENCH=/home/ubuntu/oneocr_deploy
STAMP=$(date +%Y%m%d%H%M%S)

mkdir -p "$BENCH"
tar -xzf "$BENCH/bundle.tar.gz" -C "$BENCH"

cp -a "$OCR/ocr_server.py" "$OCR/ocr_server.py.bak-$STAMP"
cp -a "$BENCH/oneocr_linux" "$OCR/"
mkdir -p "$OCR/models"
if [ ! -d "$OCR/models/onnx" ]; then cp -a "$BENCH/models/onnx" "$OCR/models/"; fi
cp -a "$BENCH/integrations/oneocr_compat.py" "$OCR/"

python3 "$BENCH/integrations/patch_online.py" "$OCR"

UNIT=/etc/systemd/system/ocrserver.service
if ! grep -q "OCR_ENGINE=oneocr" "$UNIT"; then
  sudo cp -a "$UNIT" "$UNIT.bak-$STAMP"
  sudo sed -i "/^\[Service\]\$/a Environment=OCR_ENGINE=oneocr\nEnvironment=ONEOCR_MODELS=$OCR/models/onnx\nEnvironment=OMP_NUM_THREADS=2" "$UNIT"
fi
sudo systemctl daemon-reload
sudo systemctl restart ocrserver
sleep 6
echo "active: $(systemctl is-active ocrserver)"

echo "--- smoke (empty base64 should answer missing base64) ---"
curl -s -m 30 -X POST -H 'Content-Type: application/json' \
  --data '{"type":"startocr","user":"smoke","base64":""}' \
  "http://127.0.0.1:$PORT/" | head -c 200; echo
echo "--- memory ---"
systemctl show ocrserver -p MemoryCurrent -p MemoryMax -p NRestarts