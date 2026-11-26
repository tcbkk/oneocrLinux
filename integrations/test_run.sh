#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/ocr_bench
V=/home/ubuntu/TGBotNet/ocr/venv/bin/python
echo "=== big (846x77) ==="
time "$V" test_http.py imgs/engine-image-37cc3551854df89ca09104e272018679abdc223c90e0d9b405ef11bff3fcb8cd.jpg
echo "=== small (683x46) ==="
time "$V" test_http.py imgs/engine-image-0e0311b5a1b4bf7c5e18f06c5199fc6bb629cc7f67a52981bf99325103c51354.jpg
echo "=== memory ==="
systemctl show ocrserver -p MemoryCurrent -p NRestarts