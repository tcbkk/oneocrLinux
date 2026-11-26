"""RapidOCR-compatible adapter backed by oneocr_linux.

ocr_server.py talks to RapidOCR through two entry points:

* ``ENGINE(bgr) -> ([[quad, text, score], ...], elapsed)``
* ``ENGINE.text_recognizer([crop, ...]) -> [[(text, score)], ...]``

Everything else (multi-recipe retry, tile splitting, 9-group voting, checksum)
stays untouched, so this adapter only has to translate tensors and boxes.
"""
from __future__ import annotations

import numpy as np

from oneocr_linux.pipeline import OcrEngine
from oneocr_linux.recognizer import recognize_crop
from oneocr_linux.models import SessionPool


def _to_rgb(img):
    """Accept BGR / grayscale / BGRA (ocr_server feeds 2-D prepared grays)."""
    a = np.asarray(img)
    if a.ndim == 2:
        return np.ascontiguousarray(np.repeat(a[:, :, None], 3, axis=2))
    if a.shape[2] == 4:
        return np.ascontiguousarray(a[:, :, 2::-1])
    return np.ascontiguousarray(a[:, :, ::-1])


def _quad_from_box(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class OneOcrEngine:
    """Drop-in replacement for ``rapidocr_onnxruntime.RapidOCR``."""

    def __init__(self, models_root=None, threads=0):
        self._engine = OcrEngine(models_root, intra_threads=threads)
        self._pool = self._engine.sessions
        self._rec_path = self._engine.recognizer_path

    def __call__(self, img_bgr, use_det=True, use_cls=True, use_rec=True):
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0
        rgb = _to_rgb(img_bgr)
        result = self._engine.recognize(rgb)
        out = []
        for line in result.lines:
            if not line.text:
                continue
            out.append([_quad_from_box(line.box), line.text, float(line.confidence)])
        return out, 0.0

    def text_recognizer(self, crops):
        """Recognition-only path used by the column-projection splitter."""
        out = []
        for crop in crops:
            if crop is None or crop.size == 0:
                out.append([])
                continue
            rgb = _to_rgb(crop)
            rec = recognize_crop(self._pool, self._rec_path, rgb)
            out.append([(rec.text, float(rec.confidence))] if rec.text else [])
        return out

    # some RapidOCR versions expose the raw session pool
    @property
    def sessions(self):
        return self._pool