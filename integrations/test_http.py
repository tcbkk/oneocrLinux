import base64
import json
import sys
import urllib.request

img = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/ubuntu/ocr_bench/imgs/engine-image-37cc3551854df89ca09104e272018679abdc223c90e0d9b405ef11bff3fcb8cd.jpg")
raw = open(img, "rb").read()
payload = json.dumps({
    "type": "startocr",
    "user": "bench",
    "base64": base64.b64encode(raw).decode(),
}).encode()
req = urllib.request.Request("http://127.0.0.1:5555/", data=payload,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as resp:
    body = resp.read().decode("utf-8", "replace")
print(body[:1200])