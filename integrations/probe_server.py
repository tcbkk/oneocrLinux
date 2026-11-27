import os
import sys

sys.path.insert(0, "/home/ubuntu/TGBotNet/ocr")
os.environ["OCR_ENGINE"] = "oneocr"
os.environ["ONEOCR_MODELS"] = "/home/ubuntu/TGBotNet/ocr/models/onnx"

import ocr_server as s  # noqa: E402

s.ENGINE = s.make_engine()

for name in sys.argv[1:]:
    data = open(name, "rb").read()
    text, cands = s._recognize_locked(data)
    print("=" * 60)
    print(os.path.basename(name))
    print("iid_quality:", s.iid_quality(text))
    for ln in text.splitlines()[:12]:
        print("   ", repr(ln[:110]))