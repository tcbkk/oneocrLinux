"""Text detection: SegLink-style FPN head + connected-component line grouping.

The dumped detector exposes three FPN levels with, per level:

* ``scores_{hori,vert}_fpnN``   [1,1,H,W]   sigmoid segment score
* ``bbox_deltas_{hori,vert}_fpnN`` [1,8,H,W] corner offsets from the cell centre
* ``link_scores_{hori,vert}_fpnN`` [1,8,H,W] neighbour link scores

All values are already post-activation.  Cells above ``score_threshold`` are
grouped with 8-connectivity after a one-cell dilation, which reproduces the
line grouping the engine performs internally: links inside a text line are ~1.0.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np

from .models import SessionPool

# geometry of the preprocessing grid
# 1152 is what the original engine feeds ORT, but its SegLink post-processing
# recovers small ID lines that our connected-component approximation drops.
# 1600 restores them (see 1280x591 Office screenshot: 0 groups -> 9/9).
TARGET_LONG_SIDE = 1600
SIZE_MULTIPLE = 32
STRIDE = 4  # finest FPN level


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    # four corners (tl, tr, br, bl) in original image coordinates, used for
    # perspective cropping of slightly rotated lines
    quad: List[tuple] = None

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def preprocess_size(width: int, height: int) -> tuple[int, int]:
    """Match the engine's letterbox-free resize: long side 1152, sides rounded up to 32."""
    scale = TARGET_LONG_SIDE / max(width, height)
    grid_w = max(SIZE_MULTIPLE, int(np.ceil(width * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE)
    grid_h = max(SIZE_MULTIPLE, int(np.ceil(height * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE)
    return grid_w, grid_h


def prepare_image(image: "np.ndarray", grid_w: int, grid_h: int) -> np.ndarray:
    """uint8 RGB HxWx3 -> float32 NCHW in 0..255 (the detector is not normalised)."""
    from PIL import Image

    if image.shape[0] != grid_h or image.shape[1] != grid_w:
        image = np.asarray(Image.fromarray(image).resize((grid_w, grid_h), Image.BILINEAR))
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None].astype(np.float32))


def _hot_mask(outputs, score_threshold: float) -> np.ndarray:
    base = outputs["scores_hori_fpn2"][0, 0]
    h4, w4 = base.shape
    mask = base > score_threshold
    for level, stride in ((3, 8), (4, 16)):
        key = "scores_hori_fpn%d" % level
        if key not in outputs:
            continue
        scores = outputs[key][0, 0] > score_threshold
        h, w = scores.shape
        ys = np.arange(h) * (stride // STRIDE)
        xs = np.arange(w) * (stride // STRIDE)
        sub = np.zeros((h4, w4), bool)
        sub[np.ix_(ys, xs)] = scores
        mask |= sub
    return mask


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            queue = deque([(y, x)])
            seen[y, x] = True
            ys, xs = [], []
            while queue:
                cy, cx = queue.popleft()
                ys.append(cy)
                xs.append(cx)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            yield xs, ys


def _fit_quad(xs, ys, stride: int, pad: float):
    """Fit a rotated rectangle to the hot-cell cluster (PCA along the text line)."""
    pts = (np.stack([xs, ys], axis=1).astype(np.float64) + 0.5) * stride
    if len(pts) < 6:
        return None
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v

    du = (pts - mean) @ u
    dv = (pts - mean) @ v
    margin = stride * 0.5
    umin, umax = du.min() - margin - pad, du.max() + margin + pad
    vmin, vmax = dv.min() - margin - pad, dv.max() + margin + pad
    if umax - umin < stride or vmax - vmin < stride:
        return None
    tl = mean + umin * u + vmin * v
    tr = mean + umax * u + vmin * v
    br = mean + umax * u + vmax * v
    bl = mean + umin * u + vmax * v
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def detect(session_pool: SessionPool, model_path: str, image_rgb: np.ndarray,
           score_threshold: float = 0.5, min_size: int = 12,
           pad_ratio: float = 0.02) -> List[Box]:
    """Return line boxes in original image coordinates."""
    h, w = image_rgb.shape[:2]
    grid_w, grid_h = preprocess_size(w, h)
    tensor = prepare_image(image_rgb, grid_w, grid_h)
    im_info = np.array([[grid_h, grid_w, 1.0]], dtype=np.float32)

    sess = session_pool.get(model_path)
    names = [o.name for o in sess.get_outputs()]
    raw = sess.run(None, {"data": tensor, "im_info": im_info})
    outputs = dict(zip(names, raw))

    mask = _hot_mask(outputs, score_threshold)
    # Horizontal-only dilation bridges letter gaps without welding neighbouring
    # text lines together on dense UI screenshots.
    dilated = mask.copy()
    for dx in (-1, 0, 1):
        dilated |= np.roll(mask, dx, 1)

    scale_x = w / grid_w
    scale_y = h / grid_h
    boxes: List[Box] = []
    for xs, ys in _components(dilated):
        gx1, gy1 = min(xs), min(ys)
        gx2, gy2 = max(xs) + 1, max(ys) + 1
        bw = (gx2 - gx1) * STRIDE
        bh = (gy2 - gy1) * STRIDE
        if bw < min_size or bh < min_size:
            continue
        px = bw * pad_ratio
        py = bh * pad_ratio
        quad = _fit_quad(xs, ys, STRIDE, pad=1.0)
        if quad is not None:
            quad = [(qx * scale_x, qy * scale_y) for qx, qy in quad]
        boxes.append(Box(
            max(0.0, (gx1 * STRIDE - px) * scale_x),
            max(0.0, (gy1 * STRIDE - py) * scale_y),
            min(float(w), (gx2 * STRIDE + px) * scale_x),
            min(float(h), (gy2 * STRIDE + py) * scale_y),
            quad=quad,
        ))
    boxes.sort(key=lambda b: (round(b.y1 / 10), b.x1))
    return boxes


def crop_quad(image_rgb: "np.ndarray", quad, min_height: int = 4):
    """Perspective-crop a rotated quad back to an upright rectangle."""
    from PIL import Image

    tl, tr, br, bl = quad
    width = max(2, int(round(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))))
    height = max(2, int(round(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))))
    if height < min_height:
        return None
    data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
    return np.asarray(Image.fromarray(image_rgb).transform(
        (width, height), Image.QUAD, data, Image.BICUBIC))
