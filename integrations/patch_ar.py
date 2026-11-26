"""Replace ocr_server._arabic_fallback with a local oneocr-221 implementation.

Idempotent: skips when already patched.
Usage: python3 patch_ar.py /home/ubuntu/<project>/ocr
"""
import os
import sys

OCR_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/TGBotNet/ocr"
OCR = os.path.join(OCR_DIR, "ocr_server.py")
src = open(OCR, encoding="utf-8").read()

if "oneocr 阿语头" in src:
    print("already patched")
    raise SystemExit(0)

start = src.index("def _arabic_fallback(image_bytes):")
end = src.index("\ndef ", start + 10) + 1

new = '''def _arabic_fallback(image_bytes):
    """主 OCR 拼不出安装 ID 时，用 oneocr 的阿拉伯语头（221）再读一遍。

    221 头输出的是阿拉伯-印度数字（٣٩٧…），这里统一 translate 成西文数字，
    客户端拿到的仍是 9 组 7 位 ASCII 数字。失败再回落到同机 ar_server。
    """
    try:
        if ENGINE is not None and hasattr(ENGINE, "recognize_forced"):
            import cv2 as _cv2
            import numpy as _np
            _img = _cv2.imdecode(_np.frombuffer(image_bytes, _np.uint8), _cv2.IMREAD_COLOR)
            if _img is not None:
                _res = ENGINE.recognize_forced(_img, 221)
                _text = "\\n".join(l.text for l in _res.lines)
                if _text.strip():
                    log.info("oneocr 阿语头(221) → %d 行", len(_res.lines))
                    return _text.translate(_EA_TRANS)
    except Exception as e:
        log.info("oneocr 阿语头失败：%s", e)

    try:
        import urllib.request
        payload = json.dumps({
            "type": "startocr",
            "user": "arfb",
            "base64": base64.b64encode(image_bytes).decode(),
        }).encode()
        req = urllib.request.Request(AR_FALLBACK_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("rawText") or "").translate(_EA_TRANS)
    except Exception as e:
        log.info("阿拉伯语回退不可用：%s", e)
        return ""


'''
src = src[:start] + new + src[end:]
open(OCR, "w", encoding="utf-8").write(src)
print("patched:", OCR, "| lines:", len(src.splitlines()))