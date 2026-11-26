import os
import shutil

BASE = "/home/ubuntu/ocr_bench"
src = open(os.path.join(BASE, "..", "TGBotNet", "ocr", "ocr_server.py"), encoding="utf-8").read()

start = src.index("    global ENGINE\n    if ENGINE is None:")
end_marker = '        log.info("RapidOCR 就绪")'
end = src.index(end_marker, start) + len(end_marker)
new_block = """    global ENGINE
    if ENGINE is None:
        ENGINE = make_engine()"""
src = src[:start] + new_block + src[end:]

helper = '''
def make_engine():
    """Engine factory: OCR_ENGINE=oneocr switches to the ONNX oneocr port."""
    kind = os.environ.get("OCR_ENGINE", "rapid").lower()
    if kind == "oneocr":
        from oneocr_compat import OneOcrEngine
        root = os.environ.get("ONEOCR_MODELS", "/home/ubuntu/ocr_bench/models/onnx")
        log.info("worker 初始化 oneocr_linux (%s)…", root)
        return OneOcrEngine(root)
    from rapidocr_onnxruntime import RapidOCR
    log.info("worker 初始化 RapidOCR…")
    return RapidOCR(det_model_path="", det_limit_side_len=1280, det_limit_type="max")


'''
src = src.replace("def recognize(image_bytes):", helper + "def recognize(image_bytes):", 1)
open(os.path.join(BASE, "ocr_server_bench.py"), "w", encoding="utf-8").write(src)

# adapter must be importable from the bench dir
shutil.copyfile(os.path.join(BASE, "integrations", "oneocr_compat.py"),
                os.path.join(BASE, "oneocr_compat.py"))
print("patched, make_engine present:", "def make_engine" in src)