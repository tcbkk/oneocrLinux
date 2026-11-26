"""Command line entry point: oneocr-linux IMAGE [IMAGE ...]."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import OcrEngine, default_models_root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="oneocr-linux",
                                 description="Run OneOCR models on Linux/CPU via onnxruntime.")
    ap.add_argument("images", nargs="+", help="image files")
    ap.add_argument("--models", default=None,
                    help="directory with detector.onnx / rec_<classes>.onnx "
                         "(default: <repo>/models/onnx)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--classes", type=int, default=None,
                    help="preferred recognizer class count (default: largest head)")
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = ORT default)")
    ap.add_argument("--score", type=float, default=0.5, help="detector score threshold")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = args.models or default_models_root()

    engine = OcrEngine(root, intra_threads=args.threads,
                       recognizer_classes=args.classes, score_threshold=args.score)
    try:
        for path in args.images:
            result = engine.recognize(path)
            if args.json:
                print(json.dumps({"image": path, **result.as_dict()}, ensure_ascii=False))
            else:
                print("# %s (%dx%d, %d lines)" % (path, result.width, result.height,
                                                  len(result.lines)))
                for line in result.lines:
                    print("%s\t%s" % (line.text, [round(v, 1) for v in line.box]))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())