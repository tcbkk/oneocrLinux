"""OneOCR on Linux — pure ONNX Runtime reimplementation of the Windows OneOCR engine.

The Windows engine ships three closed-source pieces (oneocr.dll, oneocr.onemodel,
onnxruntime.dll). `tools/` extracts the decrypted ONNX graphs once on Windows; the
runtime code in this package only needs onnxruntime + numpy + Pillow.
"""

from .pipeline import OcrEngine, OcrLine, recognize_image  # noqa: F401

__version__ = "0.3.0"