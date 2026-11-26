"""High level OCR pipeline with automatic script/head selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import detector, models, recognizer

# script_classifier (model_002) id -> recognizer head
SCRIPT_TO_HEAD = {
    1: 32632,   # Han / kana / hangul
    2: 32632,   # Latin (any head shares ASCII)
    3: 415,     # Cyrillic
    4: 221,     # Arabic
    5: 237,     # Devanagari
}


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
                 score_threshold: float = 0.5,
                 auto_script: bool = True):
        self.bundle = models.discover(models_root)
        self.sessions = models.SessionPool(intra_threads=intra_threads)
        self.score_threshold = score_threshold
        self.auto_script = auto_script and bool(self.bundle.script_classifier)
        self.default_classes = recognizer_classes or max(self.bundle.recognizers)
        self._script_sess = None if not self.auto_script else None

    # ------------------------------------------------------------------ head
    def _pick_head(self, crop: np.ndarray) -> int:
        if not self.auto_script:
            return self.default_classes
        try:
            if self._script_sess is None:
                self._script_sess = self.sessions.get(self.bundle.script_classifier)
            resized = recognizer._resize_height(crop)
            w = resized.shape[1]
            tensor = np.ascontiguousarray(
                resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            outs = self._script_sess.run(None, {"data": tensor})
            names = [o.name for o in self._script_sess.get_outputs()]
            score = None
            for name, val in zip(names, outs):
                if name == "script_id_score":
                    score = np.asarray(val).reshape(-1)
                    break
            if score is None:
                return self.default_classes
            sid = int(np.argmax(score))
            head = SCRIPT_TO_HEAD.get(sid)
            if head and head in self.bundle.recognizers:
                return head
        except Exception:
            pass
        return self.default_classes

    # ------------------------------------------------------------- recognize
    def recognize(self, image, forced_head: Optional[int] = None) -> OcrResult:
        """Recognize a path or an RGB uint8 ndarray.

        ``forced_head`` bypasses script detection (used by the Arabic fallback).
        """
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
            head = forced_head if forced_head in self.bundle.recognizers else self._pick_head(crop)
            rec = recognizer.recognize_crop(self.sessions, self.bundle.recognizers[head], crop)
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