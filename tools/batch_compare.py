#!/usr/bin/env python3
"""Batch OCR comparison: Windows oneocr.dll vs oneocr_linux (pure ONNX).

Usage:
    python tools/batch_compare.py --images <dir> --out samples/batch.json
    python tools/batch_compare.py --images <dir> --engine linux --out ...
"""
import argparse
import ctypes
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- windows
def windows_engine(runtime=None):
    runtime = os.path.abspath(runtime or os.path.join(HERE, "runtime"))
    os.environ.setdefault("ONEOCR_DUMP_DIR", os.path.join(ROOT, "models", "batch_dump"))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(runtime)
    ctypes.windll.kernel32.SetDllDirectoryW(runtime)

    dll = ctypes.CDLL(os.path.join(runtime, "oneocr.dll"))

    class Img(ctypes.Structure):
        _fields_ = [("t", ctypes.c_int), ("col", ctypes.c_int), ("row", ctypes.c_int),
                    ("unk", ctypes.c_int), ("step", ctypes.c_int), ("data", ctypes.c_void_p)]

    dll.CreateOcrInitOptions.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    dll.OcrInitOptionsSetUseModelDelayLoad.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
    dll.CreateOcrPipeline.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p)]
    dll.CreateOcrProcessOptions.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    dll.OcrProcessOptionsSetMaxRecognitionLineCount.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.RunOcrPipeline.argtypes = [ctypes.c_void_p, Img, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrLineCount.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrLine.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrLineContent.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]
    dll.GetOcrLineBoundingBox.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]

    ctx = ctypes.c_void_p()
    assert dll.CreateOcrInitOptions(ctypes.byref(ctx)) == 0
    dll.OcrInitOptionsSetUseModelDelayLoad(ctx, 0)
    key = b'kj)TGtrK>f]b[Piow.gU+nC@s' + b'"' * 6 + b'4'
    pipe = ctypes.c_void_p()
    rc = dll.CreateOcrPipeline(os.path.join(runtime, "oneocr.onemodel").encode(), key, ctx,
                               ctypes.byref(pipe))
    assert rc == 0, rc
    opt = ctypes.c_void_p()
    dll.CreateOcrProcessOptions(ctypes.byref(opt))
    dll.OcrProcessOptionsSetMaxRecognitionLineCount(opt, 1000)

    def run(path):
        from PIL import Image
        im = Image.open(path).convert("RGBA")
        w, h = im.size
        buf = im.tobytes("raw", "BGRA")
        arr = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
        inst = ctypes.c_longlong()
        rc = dll.RunOcrPipeline(pipe, Img(3, w, h, 0, w * 4, ctypes.cast(arr, ctypes.c_void_p)),
                                opt, ctypes.byref(inst))
        if rc != 0:
            return {"error": rc, "lines": []}
        n = ctypes.c_longlong()
        dll.GetOcrLineCount(inst, ctypes.byref(n))
        lines = []
        for i in range(n.value):
            line = ctypes.c_longlong()
            dll.GetOcrLine(inst, i, ctypes.byref(line))
            ptr = ctypes.c_void_p()
            dll.GetOcrLineContent(line, ctypes.byref(ptr))
            text = ctypes.string_at(ptr.value).decode("gbk", "replace")
            bp = ctypes.c_void_p()
            dll.GetOcrLineBoundingBox(line, ctypes.byref(bp))
            box = [round(ctypes.cast(bp, ctypes.POINTER(ctypes.c_float))[k], 1) for k in range(8)]
            lines.append({"text": text, "box": box})
        return {"lines": lines}

    return run


# --------------------------------------------------------------------------- linux
def linux_engine(models_root=None):
    from oneocr_linux.pipeline import OcrEngine

    engine = OcrEngine(models_root or os.path.join(ROOT, "models", "onnx"))

    def run(path):
        r = engine.recognize(path)
        return {"lines": [{"text": ln.text, "box": ln.box} for ln in r.lines]}

    return run


# --------------------------------------------------------------------------- helpers
def sort_lines(lines):
    def key(line):
        b = line["box"]
        return (round(b[1] / 12), b[0])
    return sorted(lines, key=key)


def seven_digit_groups(lines):
    text = " ".join(ln["text"] for ln in lines)
    return re.findall(r"(?<!\d)(\d{7})(?!\d)", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--engine", choices=["windows", "linux", "both"], default="both")
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "samples", "batch.json"))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    engines = {}
    if a.engine in ("windows", "both"):
        engines["windows"] = windows_engine()
    if a.engine in ("linux", "both"):
        engines["linux"] = linux_engine(a.models)

    out = {}
    for path in paths:
        name = os.path.basename(path)
        entry = {"name": name}
        for key, run in engines.items():
            t0 = time.perf_counter()
            res = run(path)
            res["seconds"] = round(time.perf_counter() - t0, 3)
            res["lines"] = sort_lines(res.get("lines", []))
            res["groups"] = seven_digit_groups(res["lines"])
            entry[key] = res
            print("%-46s %-8s %5.2fs  %d lines  %d groups" % (
                name[:46], key, res["seconds"], len(res["lines"]), len(res["groups"])))
        out[name] = entry

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()