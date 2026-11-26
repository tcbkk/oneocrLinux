#!/usr/bin/env python3
"""Run one OCR then scan this process' memory for long CJK strings (charset tables)."""
import ctypes
import ctypes.wintypes as wt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_models import Img, KEY, load_png_bgra  # noqa: E402

k32 = ctypes.windll.kernel32


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wt.DWORD), ("RegionSize", ctypes.c_size_t),
                ("State", wt.DWORD), ("Protect", wt.DWORD), ("Type", wt.DWORD)]


MEM_COMMIT = 0x1000
PAGE_GUARD = 0x100
READABLE = {0x02, 0x04, 0x20, 0x40, 0x80}  # RO, RW, EXEC_RO, EXEC_RW, WRITECOPY


def iter_regions():
    addr = 0
    mbi = MEMORY_BASIC_INFORMATION()
    size = ctypes.sizeof(mbi)
    while k32.VirtualQuery(ctypes.c_void_p(addr), ctypes.byref(mbi), size):
        base = mbi.BaseAddress or 0
        if mbi.State == MEM_COMMIT and (mbi.Protect & 0xFF) in READABLE and not (mbi.Protect & PAGE_GUARD):
            yield base, mbi.RegionSize
        addr = base + mbi.RegionSize
        if addr > 0x7FFFFFFFFFFF:
            break


def main():
    runtime = os.path.abspath(os.path.join(os.path.dirname(__file__), "runtime"))
    os.environ["ONEOCR_DUMP_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "mem"))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(runtime)
    ctypes.windll.kernel32.SetDllDirectoryW(runtime)
    dll = ctypes.CDLL(os.path.join(runtime, "oneocr.dll"))
    dll.CreateOcrInitOptions.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    dll.CreateOcrPipeline.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p)]
    dll.CreateOcrProcessOptions.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    dll.RunOcrPipeline.argtypes = [ctypes.c_void_p, Img, ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
    ctx = ctypes.c_void_p()
    dll.CreateOcrInitOptions(ctypes.byref(ctx))
    pl = ctypes.c_void_p()
    rc = dll.CreateOcrPipeline(os.path.join(runtime, "oneocr.onemodel").encode(), KEY, ctx, ctypes.byref(pl))
    print("pipeline", rc)
    opt = ctypes.c_void_p()
    dll.CreateOcrProcessOptions(ctypes.byref(opt))
    buf, w, h, step = load_png_bgra(os.path.join(os.path.dirname(__file__), "..", "samples", "sample.png"))
    inst = ctypes.c_longlong()
    dll.RunOcrPipeline(pl, Img(3, w, h, 0, step, ctypes.cast(buf, ctypes.c_void_p)), opt, ctypes.byref(inst))
    print("ocr done")

    hits = []
    for base, size in iter_regions():
        try:
            data = ctypes.string_at(base, size)
        except Exception:
            continue
        # UTF-16LE CJK runs: repeating pattern XX 4E-9F
        i = 0
        n = len(data) - 1
        while i < n:
            lo, hi = data[i], data[i + 1]
            if 0x00 <= hi <= 0x9F and (0x4E <= lo <= 0x9F):
                start = i
                cnt = 0
                while i < n:
                    lo, hi = data[i], data[i + 1]
                    if 0x00 <= hi <= 0x9F and (0x4E <= lo <= 0x9F or lo in (0x30, 0x20, 0x00)):
                        cnt += 1
                        i += 2
                    else:
                        break
                if cnt >= 800:
                    hits.append((base + start, cnt, "utf16", data[start:start + cnt * 2]))
            else:
                i += 2
    print("utf16 hits:", len(hits))
    for addr, cnt, kind, raw in sorted(hits, key=lambda x: -x[1])[:5]:
        try:
            s = raw.decode("utf-16-le", "ignore")
        except Exception:
            continue
        print("  addr=0x%X chars=%d head=%r tail=%r" % (addr, cnt, s[:30], s[-30:]))
        with open(os.path.join(os.path.dirname(__file__), "..", "models",
                               "charset_hit_%X.txt" % addr), "w", encoding="utf-8") as f:
            f.write(s)


if __name__ == "__main__":
    main()