#!/usr/bin/env python3
"""Drive oneocr.dll once through the proxy ORT runtime and dump every decrypted
model blob.

Usage:
    python extract_models.py --image ..\\samples\\sample.png [--out ..\\models\\raw]

Requirements: Windows x64, tools\\runtime populated with
    oneocr.dll, oneocr.onemodel, onnxruntime.dll (proxy), onnxruntime_real.dll
"""
import argparse
import ctypes
import os
import struct
import sys

KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s' + b'"' * 6 + b'4'


class Img(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_int),
        ("col", ctypes.c_int),
        ("row", ctypes.c_int),
        ("unk", ctypes.c_int),
        ("step", ctypes.c_int),
        ("data", ctypes.c_void_p),
    ]


def load_png_bgra(path):
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    w, h = im.size
    buf = im.tobytes("raw", "BGRA")
    arr = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
    return arr, w, h, w * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=os.path.join(os.path.dirname(__file__), "runtime"))
    ap.add_argument("--model", default=None, help="path to oneocr.onemodel")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "raw"))
    a = ap.parse_args()

    runtime = os.path.abspath(a.runtime)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.environ["ONEOCR_DUMP_DIR"] = out
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(runtime)
    ctypes.windll.kernel32.SetDllDirectoryW(runtime)

    dll = ctypes.CDLL(os.path.join(runtime, "oneocr.dll"))
    dll.CreateOcrInitOptions.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    dll.CreateOcrInitOptions.restype = ctypes.c_int
    dll.OcrInitOptionsSetUseModelDelayLoad.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
    dll.OcrInitOptionsSetUseModelDelayLoad.restype = ctypes.c_int
    dll.CreateOcrPipeline.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p)]
    dll.CreateOcrPipeline.restype = ctypes.c_int
    dll.CreateOcrProcessOptions.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    dll.CreateOcrProcessOptions.restype = ctypes.c_int
    dll.OcrProcessOptionsSetMaxRecognitionLineCount.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.OcrProcessOptionsSetMaxRecognitionLineCount.restype = ctypes.c_int
    dll.RunOcrPipeline.argtypes = [ctypes.c_void_p, Img, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_longlong)]
    dll.RunOcrPipeline.restype = ctypes.c_int
    dll.GetOcrLineCount.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrLineCount.restype = ctypes.c_int
    dll.GetOcrLine.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrLine.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrLine.restype = ctypes.c_int
    dll.GetOcrLineContent.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_char_p)]
    dll.GetOcrLineContent.restype = ctypes.c_int

    ctx = ctypes.c_void_p()
    rc = dll.CreateOcrInitOptions(ctypes.byref(ctx))
    print("CreateOcrInitOptions ->", rc, hex(ctx.value or 0))
    rc = dll.OcrInitOptionsSetUseModelDelayLoad(ctx, 0)
    print("SetUseModelDelayLoad(0) ->", rc)

    model = os.path.abspath(a.model or os.path.join(runtime, "oneocr.onemodel"))
    pipeline = ctypes.c_void_p()
    rc = dll.CreateOcrPipeline(model.encode("utf-8"), KEY, ctx, ctypes.byref(pipeline))
    print("CreateOcrPipeline ->", rc, hex(pipeline.value or 0))
    if rc != 0:
        print("failed; see proxy_log.txt in", out)
        return 1

    data, w, h, step = load_png_bgra(a.image)
    img = Img(3, w, h, 0, step, ctypes.cast(data, ctypes.c_void_p))
    opt = ctypes.c_void_p()
    print("CreateOcrProcessOptions ->", dll.CreateOcrProcessOptions(ctypes.byref(opt)))
    dll.OcrProcessOptionsSetMaxRecognitionLineCount(opt, 1000)

    inst = ctypes.c_longlong()
    rc = dll.RunOcrPipeline(pipeline, img, opt, ctypes.byref(inst))
    print("RunOcrPipeline ->", rc, inst.value)

    if rc == 0:
        n = ctypes.c_longlong()
        dll.GetOcrLineCount(inst, ctypes.byref(n))
        print("lines:", n.value)
        for i in range(min(n.value, 20)):
            line = ctypes.c_longlong()
            dll.GetOcrLine(inst, i, ctypes.byref(line))
            content = ctypes.c_char_p()
            dll.GetOcrLineContent(line, ctypes.byref(content))
            print("  [%d] %s" % (i, content.value.decode("utf-8", "replace") if content.value else ""))

    print("\ndumped files:")
    for name in sorted(os.listdir(out)):
        p = os.path.join(out, name)
        print("  %-40s %10d" % (name, os.path.getsize(p)))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
