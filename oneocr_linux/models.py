"""Model bundle discovery and session creation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import onnxruntime as ort

DETECTOR = "detector.onnx"
SCRIPT_CLASSIFIER = "script_classifier.onnx"
# recognizer files are named rec_<classes>.onnx
RECOGNIZER_PREFIX = "rec_"
RECOGNIZER_SUFFIX = ".onnx"
DEFAULT_RECOGNIZER = "rec_32632.onnx"


@dataclass
class ModelBundle:
    """Resolved paths of the extracted ONNX graphs."""

    root: str
    detector: str
    script_classifier: Optional[str] = None
    recognizers: Dict[int, str] = None  # noqa: RUF013 - dataclass default below

    def recognizer(self, classes: Optional[int] = None) -> str:
        classes = classes or max(self.recognizers)
        try:
            return self.recognizers[classes]
        except KeyError as exc:
            raise KeyError(
                "no recognizer with %s classes; available: %s"
                % (classes, sorted(self.recognizers))
            ) from exc


def discover(root: str) -> ModelBundle:
    """Locate the extracted graphs inside *root*."""
    root = os.path.abspath(root)
    det = os.path.join(root, DETECTOR)
    if not os.path.isfile(det):
        raise FileNotFoundError(
            "detector model not found: %s\n"
            "Run tools/extract_models.py on Windows first, then copy the "
            "models/onnx directory here." % det
        )
    recs: Dict[int, str] = {}
    for name in os.listdir(root):
        if not (name.startswith(RECOGNIZER_PREFIX) and name.endswith(RECOGNIZER_SUFFIX)):
            continue
        digits = name[len(RECOGNIZER_PREFIX):-len(RECOGNIZER_SUFFIX)]
        if digits.isdigit():
            recs[int(digits)] = os.path.join(root, name)
    if not recs:
        raise FileNotFoundError("no recognizer graphs found in %s" % root)

    script = os.path.join(root, SCRIPT_CLASSIFIER)
    return ModelBundle(
        root=root,
        detector=det,
        script_classifier=script if os.path.isfile(script) else None,
        recognizers=recs,
    )


class SessionPool:
    """Lazily created ORT sessions with a quiet, CPU-oriented default."""

    def __init__(self, intra_threads: int = 0):
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        self._opts = opts
        self._sessions: Dict[str, ort.InferenceSession] = {}

    def get(self, path: str) -> ort.InferenceSession:
        key = os.path.abspath(path)
        sess = self._sessions.get(key)
        if sess is None:
            sess = ort.InferenceSession(key, self._opts, providers=["CPUExecutionProvider"])
            self._sessions[key] = sess
        return sess

    def close(self) -> None:
        self._sessions.clear()


def recognizer_list(bundle: ModelBundle) -> List[int]:
    return sorted(bundle.recognizers)