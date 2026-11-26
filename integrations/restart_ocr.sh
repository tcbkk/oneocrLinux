#!/usr/bin/env bash
set -euo pipefail
UNIT=/etc/systemd/system/ocrserver.service
if grep -q "OCR_ENGINE=oneocr" "$UNIT"; then
  echo "env already set"
else
  sudo cp -a "$UNIT" "$UNIT.bak-$(date +%Y%m%d%H%M%S)"
  sudo sed -i '/^\[Service\]$/a Environment=OCR_ENGINE=oneocr\nEnvironment=ONEOCR_MODELS=/home/ubuntu/TGBotNet/ocr/models/onnx\nEnvironment=OMP_NUM_THREADS=2' "$UNIT"
  echo "env injected"
fi
sudo systemctl daemon-reload
sudo systemctl restart ocrserver
sleep 6
systemctl is-active ocrserver
systemctl show ocrserver -p MemoryMax -p NRestarts
ps -o pid,rss,etime,cmd -C python --sort=-rss | head -5