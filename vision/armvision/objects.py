"""Empty-mat reference differencing in the rectified view -> object contours with colour names."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import VisionConfig

CHROMATIC = {  # same hue ranges as nlp/vision/hsv.py (degrees; red wraps around 0)
    "red": [(0, 15), (345, 360)], "orange": [(15, 40)], "yellow": [(40, 70)], "green": [(70, 170)],
    "blue": [(190, 260)], "purple": [(260, 300)], "pink": [(300, 345)],
}
MIN_SAT, MIN_VAL = 0.35, 0.2


def color_name(bgr_pixels: np.ndarray) -> str:
    """Dominant colour name of (N, 3) BGR pixels."""
    if len(bgr_pixels) == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    scores = {}
    for name, ranges in CHROMATIC.items():
        in_hue = np.zeros(len(h), bool)
        for lo, hi in ranges:
            in_hue |= (h >= lo) & (h <= hi)
        scores[name] = float((in_hue & (s >= MIN_SAT) & (v >= MIN_VAL)).mean())
    scores["brown"] = float(((h >= 10) & (h <= 40) & (s >= 0.3) & (v >= MIN_VAL) & (v < 0.6)).mean()) * 0.9
    scores["black"] = float((v < 0.2).mean())
    scores["white"] = float(((s < 0.15) & (v > 0.8)).mean())
    scores["gray"] = float(((s < 0.15) & (v >= 0.2) & (v <= 0.8)).mean())
    name = max(scores, key=scores.get)
    return name if scores[name] >= 0.3 else "unknown"


@dataclass
class Detection:
    polygon_mm: np.ndarray   # (N, 2) contour in mat mm
    color: str
    area_mm2: float


def foreground_mask(rect: np.ndarray, reference: np.ndarray, cfg: VisionConfig,
                    ignore: np.ndarray | None = None) -> np.ndarray:
    """uint8 0/255 mask of pixels differing from the empty-mat reference (Lab distance).

    Robust to lighting: the frame-wide brightness shift (auto-exposure) is removed, lightness counts for
    cfg.l_weight of a colour change so soft shadows on white paper stay below the threshold, and `ignore`
    (0/255, e.g. the printed markers) is cleared."""
    a = cv2.cvtColor(cv2.GaussianBlur(rect, (5, 5), 0), cv2.COLOR_BGR2Lab).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(reference, (5, 5), 0), cv2.COLOR_BGR2Lab).astype(np.float32)
    d = a - b
    d[..., 0] -= np.median(d[..., 0])
    d[..., 0] *= cfg.l_weight
    mask = (np.linalg.norm(d, axis=2) > cfg.diff_threshold).astype(np.uint8) * 255
    if ignore is not None:
        mask[ignore > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    m = round(cfg.edge_margin_mm * cfg.px_per_mm)
    if m > 0:
        mask[:m], mask[-m:], mask[:, :m], mask[:, -m:] = 0, 0, 0, 0
    return mask


def extract_detections(rect: np.ndarray, obj_mask: np.ndarray, cfg: VisionConfig,
                       mat_height_mm: float) -> list[Detection]:
    s = cfg.px_per_mm
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        area = cv2.contourArea(c) / (s * s)
        if area < cfg.min_area_mm2:
            continue
        fill = np.zeros(obj_mask.shape, np.uint8)
        cv2.drawContours(fill, [c], -1, 255, -1)
        fill = cv2.erode(fill, np.ones((3, 3), np.uint8))
        poly = c.reshape(-1, 2).astype(np.float64)
        poly_mm = np.c_[poly[:, 0] / s, mat_height_mm - poly[:, 1] / s]
        out.append(Detection(poly_mm, color_name(rect[fill > 0]), area))
    return out


def near_half(polygon_mm: np.ndarray, cam_xy) -> np.ndarray:
    """Keep the contour points on the camera-near side (tall objects lean away from the camera)."""
    c = polygon_mm.mean(0)
    d = np.asarray(cam_xy, np.float64) - c
    n = np.linalg.norm(d)
    if n < 1e-6:
        return polygon_mm
    keep = (polygon_mm - c) @ (d / n) >= 0
    return polygon_mm[keep] if keep.sum() >= 3 else polygon_mm
