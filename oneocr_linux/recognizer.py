"""CRNN recognizer: crop -> height 60 -> CTC head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .charset import decode_ctc
from .models import SessionPool

INPUT_HEIGHT = 60
MIN_WIDTH = 8
MAX_WIDTH = 4096
WIDTH_ALIGN = 8


@dataclass
class Recognition:
    text: str
    confidence: float
    classes: int


def _resize_height(crop: "np.ndarray", height: int = INPUT_HEIGHT) -> "np.ndarray":
    from PIL import Image

    h, w = crop.shape[:2]
    width = max(MIN_WIDTH, int(round(w * height / max(1, h))))
    width = min(MAX_WIDTH, (width + WIDTH_ALIGN - 1) // WIDTH_ALIGN * WIDTH_ALIGN)
    return np.asarray(Image.fromarray(crop).resize((width, height), Image.BILINEAR))


def recognize_crop(session_pool: SessionPool, model_path: str, crop_rgb: np.ndarray) -> Recognition:
    """Recognize a single line crop (uint8 RGB)."""
    resized = _resize_height(crop_rgb)
    width = resized.shape[1]
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    seq_lengths = np.array([max(1, width // WIDTH_ALIGN)], dtype=np.int32)

    sess = session_pool.get(model_path)
    logits = sess.run(None, {"data": tensor, "seq_lengths": seq_lengths})[0]
    logits = np.asarray(logits).reshape(-1, logits.shape[-1])
    classes = logits.shape[-1]
    blank = classes - 1
    ids = logits.argmax(-1)

    text = decode_ctc(ids, blank)
    keep = ids != blank
    if keep.any():
        best = logits[keep].max(axis=-1)
        confidence = float(np.exp(best).mean())
    else:
        confidence = 0.0
    return Recognition(text=text, confidence=confidence, classes=classes)


def pick_recognizer(recognizers, preferred_classes: Optional[int] = None) -> str:
    if preferred_classes is None:
        preferred_classes = max(recognizers)
    if preferred_classes not in recognizers:
        preferred_classes = max(recognizers)
    return recognizers[preferred_classes]