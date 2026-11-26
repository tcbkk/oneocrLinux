#!/usr/bin/env python3
"""Recover the 32632-class recognizer charset by single-glyph probing.

The head maps the first 95 tokens to ASCII (chr(i+32)) but everything above is a
private ordering, so `chr()` decoding turns Chinese into gibberish. This script
renders each candidate glyph at several sizes, reads the argmax token with the
real recognizer, and votes, producing `oneocr_linux/data/charset.json`
(token -> char).
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

FONT = r"C:\Windows\Fonts\msyh.ttc"
BLANK = 32631
SIZES = ((24, 2), (28, 6), (32, 2), (36, 6), (40, 2))


def candidates():
    chars = [chr(c) for c in range(0x20, 0x7F)]
    seen = set(chars)
    # GB2312 hanzi + fullwidth forms, in code order
    for hi in range(0xB0, 0xF8):
        for lo in range(0xA1, 0xFF):
            try:
                ch = bytes([hi, lo]).decode("gbk")
            except UnicodeDecodeError:
                continue
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    # GB2312 punctuation / symbols zones
    for hi in range(0xA1, 0xAA):
        for lo in range(0xA1, 0xFF):
            try:
                ch = bytes([hi, lo]).decode("gbk")
            except UnicodeDecodeError:
                continue
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return chars


def ids_of(sess, ch, size, pad):
    im = Image.new("RGB", (size + 2 * pad + 20, size + 2 * pad), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, size)
    d.text((pad + 10, pad), ch, font=f, fill="black")
    rs = recognizer._resize_height(np.array(im))
    w = rs.shape[1]
    t = np.ascontiguousarray(rs.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    lg = sess.run(None, {"data": t, "seq_lengths": np.array([max(1, w // 8)], dtype=np.int32)})[0]
    idx = lg.reshape(-1, lg.shape[-1]).argmax(-1)
    probs = np.exp(lg.reshape(-1, lg.shape[-1]))
    col, prev = [], -1
    for pos, i in enumerate(idx):
        if i != prev and i != BLANK:
            col.append((int(i), float(probs[pos, i])))
        prev = i
    return col


def main():
    pool = SessionPool()
    sess = pool.get(os.path.join(ROOT, "models", "onnx", "rec_32632.onnx"))
    chars = candidates()
    print("candidates:", len(chars))

    votes = {}
    for n, ch in enumerate(chars, 1):
        tally = {}
        for size, pad in SIZES:
            got = ids_of(sess, ch, size, pad)
            if len(got) == 1:
                token, prob = got[0]
                if token < 95:
                    continue
                cur = tally.get(token)
                tally[token] = (cur[0] + 1, cur[1] + prob) if cur else (1, prob)
        if not tally:
            continue
        token = max(tally, key=lambda k: (tally[k][0], tally[k][1]))
        votes.setdefault(token, []).append((ch, tally[token][0], tally[token][1] / tally[token][0]))
        if n % 1000 == 0:
            print("  probed", n, "mapped", len(votes))

    charset = {}
    conflicts = 0
    for token, lst in sorted(votes.items()):
        best = max(lst, key=lambda x: (x[1], x[2]))[0]
        if len({c for c, _, _ in lst}) > 1:
            conflicts += 1
        charset[str(token)] = best

    out = os.path.join(ROOT, "oneocr_linux", "data", "charset.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"classes": 32632, "blank": BLANK, "tokens": charset}, f,
                  ensure_ascii=False, indent=0)
    print("mapped tokens:", len(charset), "conflicts:", conflicts)
    print("written:", out)
    for probe in ("阅", "读", "的", "隐", "私", "声", "明", "免", "费", "电", "话"):
        for token, ch in charset.items():
            if ch == probe:
                print("  %s -> %s" % (probe, token))
                break


if __name__ == "__main__":
    main()