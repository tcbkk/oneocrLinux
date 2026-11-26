"""Rebuild per-head charsets: single glyph + sentinel pairing.

Isolated glyphs that look like Latin letters (ا→1, П→N) are misread, so every
candidate is also rendered next to a known sentinel letter and the non-sentinel
token is taken as the answer (pairing wins over the solo reading).
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

SIZES = ((28, 4), (34, 8), (40, 4))


def ranges(*pairs):
    out = []
    for a, b in pairs:
        out.extend(chr(c) for c in range(a, b + 1))
    return out


ARABIC = (ranges((0x0621, 0x063A), (0x0641, 0x064A), (0x0660, 0x0669), (0x066E, 0x06D3))
          + list("\u060c\u061b\u061f\u0640\u066d\u0622\u0623\u0625\u0627\u0629\u0649"))
DEVANAGARI = (ranges((0x0905, 0x0939), (0x093E, 0x094C), (0x0966, 0x096F))
              + list("\u0902\u0903\u093C\u094D\u0950\u0964\u0965"))
CYRILLIC = ranges((0x0410, 0x044F)) + list("\u0401\u0451\u0404\u0454\u0406\u0456\u0407\u0457")
KANA = ranges((0x3041, 0x3096), (0x30A1, 0x30FA)) + list("\u30FC\u3002\u3001")

# head -> (chars, font, sentinel char, sentinel token)
JOBS = {
    221: (ARABIC, r"C:\Windows\Fonts\arial.ttf", "\u0628", 111),
    237: (DEVANAGARI, r"C:\Windows\Fonts\Nirmala.ttc", "\u0928", 138),
    415: (CYRILLIC, r"C:\Windows\Fonts\arial.ttf", "\u0410", 33),
    32632: (KANA, r"C:\Windows\Fonts\msyh.ttc", "\u4e00", 13),
}


def tokens_of(sess, text, font, size, pad):
    im = Image.new("RGB", (max(240, size * len(text) + 2 * pad + 30), size + 2 * pad), "white")
    ImageDraw.Draw(im).text((pad + 15, pad), text, font=ImageFont.truetype(font, size), fill="black")
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


def probe_char(sess, ch, font, sentinel, sent_tok):
    """Solo evidence first; pairing only fills glyphs the solo pass misses."""
    solo_ev, pair_ev = {}, {}
    for size, pad in SIZES:
        solo = tokens_of(sess, ch, font, size, pad)
        if len(solo) == 1 and solo[0][0] >= 95:
            solo_ev[solo[0][0]] = solo_ev.get(solo[0][0], 0) + 1
        for text in (sentinel + ch, ch + sentinel):
            got = tokens_of(sess, text, font, size, pad)
            others = [t for t, _ in got if t != sent_tok and t >= 95]
            if len(others) == 1:
                pair_ev[others[0]] = pair_ev.get(others[0], 0) + 1
    return solo_ev, pair_ev


def build(classes, chars, font, sentinel, sent_tok, pool):
    sess = pool.get(os.path.join(ROOT, "models", "onnx", "rec_%d.onnx" % classes))
    votes = {}
    for ch in chars:
        solo_ev, pair_ev = probe_char(sess, ch, font, sentinel, sent_tok)
        ev = solo_ev if solo_ev else pair_ev
        weight = 1 if solo_ev else 3   # pairing is only a fallback, but trusted when alone
        for tok, cnt in ev.items():
            votes.setdefault(tok, []).append((ch, cnt, weight))
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
    print("head %-6d chars=%-4d mapped=%-4d conflicts=%-3d" % (classes, len(chars), len(table), conflicts))
    return table


if __name__ == "__main__":
    pool = SessionPool()
    for classes, (chars, font, sentinel, sent_tok) in JOBS.items():
        build(classes, chars, font, sentinel, sent_tok, pool)