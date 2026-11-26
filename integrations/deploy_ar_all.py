import os
import sys
import time

import paramiko

HOSTS = [
    ("43.133.102.47", "/home/ubuntu/TGBotNet/ocr"),
    ("43.152.231.22", "/home/ubuntu/TGBotNet/ocr"),
    ("43.173.51.84", "/home/ubuntu/WhatsAppBotNet/ocr"),
    ("119.28.239.124", "/home/ubuntu/PidKeyStation/ocr"),
]
BUNDLE = r"D:\PROGRAM\oneocrLinux\dist\ar_bundle2.tar.gz"
COMPAT = r"D:\PROGRAM\oneocrLinux\integrations\oneocr_compat.py"
PATCH_AR = r"D:\PROGRAM\oneocrLinux\integrations\patch_ar.py"
PATCH_DIG = r"D:\PROGRAM\oneocrLinux\integrations\patch_ar_digits.py"
IMG = r"E:\project\2026-11-26\mwwzrvw2itexnnni\uploads\engine-image-4d6f21b3c85de5cc9104581d7608ce5a80c666bc3e71ee14db66b871a34c20c0.jpg"
TESTER = r"D:\PROGRAM\oneocrLinux\integrations\test_http.py"

for host, ocr in HOSTS:
    print("=" * 70)
    print(host, ocr)
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, 22, username="ubuntu", password="MMSmms760816", timeout=20,
                  allow_agent=False, look_for_keys=False)
        sftp = c.open_sftp()
        sftp.put(BUNDLE, "/tmp/ar_bundle2.tar.gz")
        sftp.put(COMPAT, "/tmp/oneocr_compat.py")
        sftp.put(PATCH_AR, "/tmp/patch_ar.py")
        sftp.put(PATCH_DIG, "/tmp/patch_ar_digits.py")
        sftp.put(IMG, "/tmp/ar_test.jpg")
        sftp.put(TESTER, "/tmp/test_http.py")
        sftp.close()
        cmd = (
            "set -e; cd /tmp && mkdir -p arb && tar -xzf ar_bundle2.tar.gz -C arb && "
            "cp -a arb/oneocr_linux/. " + ocr + "/oneocr_linux/ && "
            "cp -a oneocr_compat.py " + ocr + "/ && "
            "python3 patch_ar.py " + ocr + " && python3 patch_ar_digits.py " + ocr + " && "
            "sudo systemctl restart ocrserver && sleep 7 && systemctl is-active ocrserver && "
            + ocr + "/venv/bin/python /tmp/test_http.py /tmp/ar_test.jpg 2>/dev/null | head -c 300"
        )
        i, o, e = c.exec_command(cmd, timeout=300)
        out = o.read().decode("utf-8", "replace")
        print(out.strip()[:600])
        c.close()
    except Exception as ex:
        print("ERROR", type(ex).__name__, str(ex)[:120])