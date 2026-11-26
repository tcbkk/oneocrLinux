#!/usr/bin/env python3
"""Run the original Windows oneocr.dll and save text + boxes as a reference JSON."""
import ctypes
import locale
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_models import Img, KEY, load_png_bgra  # noqa: E402


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "samples", "sample.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "samples", "reference.json")
    locale.setlocale(locale.LC_ALL, "Chinese_China.936")
    runtime = os.path.abspath(os.path.join(os.path.dirname(__file__), "runtime"))

    os.environ.setdefault("ONEOCR_DUMP_DIR", os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "ref")))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(runtime)
    ctypes.windll.kernel32.SetDllDirectoryW(runtime)

    dll = ctypes.CDLL(os.path.join(runtime, "oneocr.dll"))
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
    dll.GetOcrLineWordCount.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrWord.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    dll.GetOcrWordContent.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]
    dll.GetOcrWordBoundingBox.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]

    ctx = ctypes.c_void_p()
    assert dll.CreateOcrInitOptions(ctypes.byref(ctx)) == 0
    dll.OcrInitOptionsSetUseModelDelayLoad(ctx, 0)
    pl = ctypes.c_void_p()
    model = os.path.join(runtime, "oneocr.onemodel")
    rc = dll.CreateOcrPipeline(model.encode(), KEY, ctx, ctypes.byref(pl))
    assert rc == 0, rc
    opt = ctypes.c_void_p()
    dll.CreateOcrProcessOptions(ctypes.byref(opt))
    dll.OcrProcessOptionsSetMaxRecognitionLineCount(opt, 1000)

    buf, w, h, step = load_png_bgra(image)
    img = Img(3, w, h, 0, step, ctypes.cast(buf, ctypes.c_void_p))
    inst = ctypes.c_longlong()
    rc = dll.RunOcrPipeline(pl, img, opt, ctypes.byref(inst))
    assert rc == 0, rc

    def bbox(handle):
        p = ctypes.c_void_p()
        dll.GetOcrLineBoundingBox(handle, ctypes.byref(p))
        arr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float))
        return [round(arr[i], 3) for i in range(8)]

    def wbox(handle):
        p = ctypes.c_void_p()
        dll.GetOcrWordBoundingBox(handle, ctypes.byref(p))
        arr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float))
        return [round(arr[i], 3) for i in range(8)]

    lines = []
    n = ctypes.c_longlong()
    dll.GetOcrLineCount(inst, ctypes.byref(n))
    for i in range(n.value):
        line = ctypes.c_longlong()
        dll.GetOcrLine(inst, i, ctypes.byref(line))
        c = ctypes.c_void_p()
        dll.GetOcrLineContent(line, ctypes.byref(c))
        text = ctypes.string_at(c.value).decode("gbk", "replace")
        wc = ctypes.c_longlong()
        dll.GetOcrLineWordCount(line, ctypes.byref(wc))
        words = []
        for j in range(wc.value):
            wd = ctypes.c_longlong()
            dll.GetOcrWord(line, j, ctypes.byref(wd))
            s = ctypes.c_void_p()
            dll.GetOcrWordContent(wd, ctypes.byref(s))
            words.append({"text": ctypes.string_at(s.value).decode("gbk", "replace"), "box": wbox(wd)})
        lines.append({
            "text": text,
            "box": bbox(line),
            "words": words,
        })

    doc = {"image": os.path.abspath(image), "width": w, "height": h, "lines": lines}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: doc[k] for k in ("width", "height")}, ensure_ascii=False))
    for ln in lines:
        print(repr(ln["text"]), ln["box"])
    print("saved", out_json)


if __name__ == "__main__":
    main()




