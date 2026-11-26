#!/usr/bin/env python3
"""Compare RapidOCR vs oneocr_linux through ocr_server.py's own pipeline.

Both engines run the identical 1400-line post-processing (multi-recipe retry,
tile splitting, 9-group voting, checksum), so the only variable is the engine.
"""
import importlib.util
import json
import os
import re
import sys
import time

BASE = "/home/ubuntu/ocr_bench"
os.chdir(BASE)
sys.path.insert(0, BASE)

TRUTH = {
    "engine-image-db85017be945856732b32b159c4d1a6df0b766ed024a86498048b6919cd8ae26.jpg":
        "379374521079068072354764738073733665405481397759210193028338484",
    "engine-image-53995527c3f0d007a430f7a70aced9c3129ba92e43e0afcb2fd227c306a1b17f.jpg":
        "252155328578852011936428509275443623334841553612070666855426403",
    "engine-image-0e0311b5a1b4bf7c5e18f06c5199fc6bb629cc7f67a52981bf99325103c51354.jpg":
        "352953228746506582652659816591365208339041811179321193344026560",
    "engine-image-2ebfaabfc75834a6266dc5c8626929c56cd4b8a9c91dee7ccd910b33a27e9844.jpg":
        "262998362545537979042388449181073233091551214975198846054798244",
    "engine-image-37cc3551854df89ca09104e272018679abdc223c90e0d9b405ef11bff3fcb8cd.jpg":
        "510363438543368033720791542277342526519943309858077094253129762",
    "engine-image-024a3c7277244e66fd3904c79420bd2a5942057b9f12f37d71796d34c54776a7.jpg":
        "432362078730014484231143017688826246504720226549202972015289681",
    "engine-image-407e385a04fd3b41e1aaa19a44cd02b0eb05d93d35f0be2f9f86efdcfb6fa67f.jpg":
        "203772238221102212382549132332626023270984077669116751326894086",
}


def load(kind, name):
    os.environ["OCR_ENGINE"] = kind
    os.environ.setdefault("ONEOCR_MODELS", os.path.join(BASE, "models", "onnx"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, "ocr_server_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ENGINE = mod.make_engine()
    return mod


def best_window(mod, text):
    best = None
    for line in text.split("\n"):
        for w in mod._nine_windows(line):
            ok = sum(1 for k in range(9) if mod.group_ok(w[k * 7:(k + 1) * 7]))
            if best is None or ok > best[0]:
                best = (ok, w)
    return best


def main():
    imgs = sorted(f for f in os.listdir(os.path.join(BASE, "imgs"))
                  if f.endswith((".jpg", ".png")) and f in TRUTH)
    print("images:", len(imgs))

    engines = {}
    for kind in ("rapid", "oneocr"):
        t0 = time.time()
        engines[kind] = load(kind, "ocr_bench_" + kind)
        print("loaded", kind, "in %.1fs" % (time.time() - t0))

    summary = {}
    for kind, mod in engines.items():
        total_groups = correct_groups = 0
        exact = 0
        rows = []
        t0 = time.time()
        for name in imgs:
            with open(os.path.join(BASE, "imgs", name), "rb") as f:
                data = f.read()
            try:
                text, cands = mod._recognize_locked(data)
            except Exception as e:
                rows.append((name, "ERR " + str(e)[:60], 0, 0))
                continue
            bw = best_window(mod, text)
            truth = TRUTH.get(name)
            if truth:
                if bw is None:
                    got_ok, got = 0, ""
                    rows.append((name, "no-9-group", 0, 9))
                else:
                    got_ok, got = bw
                    per = sum(1 for k in range(9) if got[k * 7:(k + 1) * 7] == truth[k * 7:(k + 1) * 7])
                    total_groups += 9
                    correct_groups += per
                    if got == truth:
                        exact += 1
                    rows.append((name, "groups %d/9 ok" % per, per, 9))
            else:
                rows.append((name, "no-truth" + ("" if bw is None else " (9-group found!)"), 0, 0))
        summary[kind] = {"exact": exact, "group_acc": correct_groups / total_groups if total_groups else 0,
                         "rows": rows, "seconds": round(time.time() - t0, 1)}

    out = {"summary": summary}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(BASE, "bench_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()