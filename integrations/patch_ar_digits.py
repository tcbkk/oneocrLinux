"""Normalise Arabic-Indic digits to ASCII in ocr_server output (idempotent)."""
import os
import sys

OCR_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/TGBotNet/ocr"
OCR = os.path.join(OCR_DIR, "ocr_server.py")
src = open(OCR, encoding="utf-8").read()

if "raw.translate(_EA_TRANS)" in src:
    print("already normalised")
    raise SystemExit(0)

marker = 'raw = best_text or "\\n".join(collected_any) or ""'
if marker not in src:
    print("marker not found")
    raise SystemExit(1)

src = src.replace(
    marker,
    marker + '\n    # 阿拉伯-印度数字 → 西文数字（221 头输出 ٣٩٧…，客户端只认 ASCII）\n'
             '    raw = raw.translate(_EA_TRANS)',
    1,
)
open(OCR, "w", encoding="utf-8").write(src)
print("normalised:", OCR)