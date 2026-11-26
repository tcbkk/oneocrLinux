#!/usr/bin/env bash
set -uo pipefail
echo "--- listen ---"
ss -lntp 2>/dev/null | grep -E '5555|5556' || echo "no listener"
echo "--- http ---"
curl -s -m 90 -o /tmp/resp.json -w "http_code=%{http_code} size=%{size_download}\n" \
  -X POST -H 'Content-Type: application/json' \
  --data '{"type":"startocr","user":"t","base64":""}' \
  http://127.0.0.1:5555/ || echo "curl failed"
head -c 300 /tmp/resp.json 2>/dev/null; echo
echo "--- log ---"
journalctl -u ocrserver -n 15 --no-pager 2>/dev/null | tail -15