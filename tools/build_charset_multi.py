"""Rebuild per-head charsets by single-glyph probing.

Each recognizer head has its own private token order; tokens < 95 are ASCII in
every head we tested. This script probes a per-head candidate alphabet and
writes oneocr_linux/data/charset_<classes>.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from oneocr_linux import recognizer  # noqa: E402
from oneocr_linux.models import SessionPool  # noqa: E402

FONT_AR = r"C:\Windows\Fonts\arial.ttf"
FONT_HI = r"C:\Windows\Fonts\Nirmala.ttc"
FONT_RU = r"C:\Windows\Fonts\arial.ttf"
FONT_CJK = r"C:\Windows\Fonts\msyh.ttc"
SIZES = ((26, 4), (32, 8), (40, 4))


def ranges(*pairs):
    out = []
    for a, b in pairs:
        out.extend(chr(c) for c in range(a, b + 1))
    return out


ARABIC = (ranges((0x0621, 0x063A), (0x0641, 0x064A), (0x0660, 0x0669), (0x066E, 0x06D3))
          + list("\u060c\u061b\u061f\u0640\u066d"))
DEVANAGARI = (ranges((0x0905, 0x0939), (0x093E, 0x094C), (0x0966, 0x096F))
              + list("\u0902\u0903\u093C\u094D\u0950\u0964\u0965"))
CYRILLIC = ranges((0x0410, 0x044F)) + list("\u0401\u0451\u0404\u0454\u0406\u0456\u0407\u0457")
KANA = ranges((0x3041, 0x3096), (0x30A1, 0x30FA)) + list("\u30FC\u3002\u3001")

JOBS = {
    221: (ARABIC, FONT_AR),
    237: (DEVANAGARI, FONT_HI),
    415: (CYRILLIC, FONT_RU),
    32632: (KANA, FONT_CJK),
}


def ids_of(sess, ch, size, pad, font):
    im = Image.new("RGB", (size + 2 * pad + 24, size + 2 * pad), "white")
    ImageDraw.Draw(im).text((pad + 12, pad), ch, font=ImageFont.truetype(font, size), fill="black")
    rs = recognizer._resize_height(np.array(im))
    w = rs.shape[1]
    t = np.ascontiguousarray(rs.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    lg = sess.run(None, {"data": t, "seq_lengths": np.array([max(1, w // 8)], dtype=np.int32)})[0]
    lg = lg.reshape(-1, lg.shape[-1])
    idx = lg.argmax(-1)
    probs = np.exp(lg)
    blank = lg.shape[-1] - 1
    col, prev = [], -1
    for pos, i in enumerate(idx):
        if i != prev and i != blank:
            col.append((int(i), float(probs[pos, i])))
        prev = i
    return col


def build(classes, chars, font, pool):
    path = os.path.join(ROOT, "models", "onnx", "rec_%d.onnx" % classes)
    sess = pool.get(path)
    votes = {}
    for ch in chars:
        tally = {}
        for size, pad in SIZES:
            got = ids_of(sess, ch, size, pad, font)
            if len(got) == 1 and got[0][0] >= 95:
                tok, p = got[0]
                cur = tally.get(tok)
                tally[tok] = (cur[0] + 1, cur[1] + p) if cur else (1, p)
        if not tally:
            continue
        tok = max(tally, key=lambda k: (tally[k][0], tally[k][1]))
        votes.setdefault(tok, []).append((ch, tally[tok][0], tally[tok][1] / tally[tok][0]))
    table = {}
    conflicts = 0
    for tok, lst in votes.items():
        best = max(lst, key=lambda x: (x[1], x[2]))[0]
        if len({c for c, _, _ in lst}) > 1:
            conflicts += 1
        table[str(tok)] = best
    out = os.path.join(ROOT, "oneocr_linux", "data", "charset_%d.json" % classes)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "tokens": table}, f, ensure_ascii=False, indent=0)
    print("head %-6d chars=%-4d mapped=%-4d conflicts=%-3d -> %s"
          % (classes, len(chars), len(table), conflicts, os.path.basename(out)))
    return table


if __name__ == "__main__":
    pool = SessionPool()
    for classes, (chars, font) in JOBS.items():
        build(classes, chars, font, pool)