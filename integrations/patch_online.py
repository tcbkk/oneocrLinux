import os
import re

OCR = "/home/ubuntu/TGBotNet/ocr/ocr_server.py"
src = open(OCR, encoding="utf-8").read()

if "def make_engine(" in src:
    print("already patched")
    raise SystemExit(0)

start = src.index("    global ENGINE\n    if ENGINE is None:")
end_marker = '        log.info("RapidOCR 就绪")'
end = src.index(end_marker, start) + len(end_marker)
src = src[:start] + """    global ENGINE
    if ENGINE is None:
        ENGINE = make_engine()""" + src[end:]

helper = '''
def make_engine():
    """OCR_ENGINE=oneocr 时用 oneocr_linux，否则回退 RapidOCR。"""
    kind = os.environ.get("OCR_ENGINE", "rapid").lower()
    if kind == "oneocr":
        from oneocr_compat import OneOcrEngine
        root = os.environ.get("ONEOCR_MODELS",
                              "/home/ubuntu/TGBotNet/ocr/models/onnx")
        log.info("worker 初始化 oneocr_linux (%s)…", root)
        return OneOcrEngine(root)
    from rapidocr_onnxruntime import RapidOCR
    log.info("worker 初始化 RapidOCR…")
    return RapidOCR(det_model_path="", det_limit_side_len=1280, det_limit_type="max")


'''
src = src.replace("def recognize(image_bytes):", helper + "def recognize(image_bytes):", 1)
open(OCR, "w", encoding="utf-8").write(src)
print("patched:", "def make_engine" in src, "| lines:", len(src.splitlines()))