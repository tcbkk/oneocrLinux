import os
import subprocess

import paramiko

HOSTS = [
    ("43.133.102.47", "/home/ubuntu/TGBotNet/ocr/venv/bin/python", "TGBotNet+WhatsAppBot"),
    ("43.152.231.22", "/home/ubuntu/TGBotNet/ocr/venv/bin/python", "ntqqBot+ServerNet"),
    ("43.173.51.84", "/home/ubuntu/WhatsAppBotNet/ocr/venv/bin/python", "ntqqBot+WhatsAppBotNet"),
    ("119.28.239.124", "/home/ubuntu/PidKeyStation/ocr/venv/bin/python", "PidKeyStation"),
]
IMG = r"E:\project\2026-11-26\mwwzrvw2itexnnni\uploads\engine-image-37cc3551854df89ca09104e272018679abdc223c90e0d9b405ef11bff3fcb8cd.jpg"
SCRIPT = r"D:\PROGRAM\oneocrLinux\integrations\test_http.py"
EXPECT = "5103634 3854336 8033720 7915422 7734252 6519943 3098580 7709425 3129762"

for host, py, note in HOSTS:
    print("=" * 72)
    print(host, note)
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, 22, username="ubuntu", password="MMSmms760816", timeout=20,
                  allow_agent=False, look_for_keys=False)
        sftp = c.open_sftp()
        sftp.put(IMG, "/home/ubuntu/oneocr_deploy/test.jpg")
        sftp.put(SCRIPT, "/home/ubuntu/oneocr_deploy/test_http.py")
        sftp.close()
        cmd = "%s /home/ubuntu/oneocr_deploy/test_http.py /home/ubuntu/oneocr_deploy/test.jpg" % py
        i, o, e = c.exec_command(cmd, timeout=180)
        out = o.read().decode("utf-8", "replace").strip()
        err = e.read().decode("utf-8", "replace").strip()
        print(out[:400])
        ok = EXPECT in out
        print("MATCH" if ok else "MISMATCH")
        if err:
            print("stderr:", err[:200])
        i2, o2, _ = c.exec_command("systemctl show ocrserver -p MemoryCurrent -p NRestarts", timeout=30)
        print(o2.read().decode().strip())
        c.close()
    except Exception as ex:
        print("ERROR", type(ex).__name__, str(ex)[:120])