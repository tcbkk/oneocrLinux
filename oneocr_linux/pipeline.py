"""High level OCR pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import detector, models, recognizer


@dataclass
class OcrLine:
    text: str
    box: List[float]
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {"text": self.text, "box": self.box, "confidence": round(self.confidence, 4)}


@dataclass
class OcrResult:
    width: int
    height: int
    lines: List[OcrLine] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "lines": [ln.as_dict() for ln in self.lines]}

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


class OcrEngine:
    """Loads the extracted graphs once and reuses them across calls."""

    def __init__(self, models_root: str, intra_threads: int = 0,
                 recognizer_classes: Optional[int] = None,
                 score_threshold: float = 0.5):
        self.bundle = models.discover(models_root)
        self.sessions = models.SessionPool(intra_threads=intra_threads)
        self.score_threshold = score_threshold
        self.recognizer_path = recognizer.pick_recognizer(
            self.bundle.recognizers, recognizer_classes)

    def recognize(self, image) -> OcrResult:
        """Recognize a path or an RGB uint8 ndarray."""
        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            pil = Image.open(image).convert("RGB")
            array = np.asarray(pil)
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            if array.shape[2] == 4:
                array = array[:, :, :3]
        height, width = array.shape[:2]

        boxes = detector.detect(self.sessions, self.bundle.detector, array,
                                score_threshold=self.score_threshold)
        lines: List[OcrLine] = []
        for box in boxes:
            crop = None
            if box.quad:
                crop = detector.crop_quad(array, box.quad)
            if crop is None:
                x1, y1, x2, y2 = (int(round(v)) for v in box.as_list())
                crop = array[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size == 0:
                continue
            rec = recognizer.recognize_crop(self.sessions, self.recognizer_path, crop)
            if not rec.text:
                continue
            lines.append(OcrLine(text=rec.text, box=[round(v, 2) for v in box.as_list()],
                                 confidence=rec.confidence))
        return OcrResult(width=width, height=height, lines=lines)

    def close(self) -> None:
        self.sessions.close()


def default_models_root() -> str:
    """Locate models for both a source checkout and a PyInstaller one-file build."""
    import sys

    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
        return os.path.join(base, "models", "onnx")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "onnx")


def recognize_image(path: str, models_root: Optional[str] = None, **kwargs) -> OcrResult:
    root = models_root or default_models_root()
    engine = OcrEngine(root, **kwargs)
    try:
        return engine.recognize(path)
    finally:
        engine.close()