"""Numpy-only color matching. Frames are RGB uint8 arrays of shape (H, W, 3)."""
from __future__ import annotations

import numpy as np

CHROMATIC = {  # color -> hue ranges in degrees (inclusive); red wraps around 0
    "red": [(0, 15), (345, 360)],
    "orange": [(15, 40)],
    "yellow": [(40, 70)],
    "green": [(70, 170)],
    "blue": [(190, 260)],
    "purple": [(260, 300)],
    "pink": [(300, 345)],
}
MIN_SAT, MIN_VAL = 0.35, 0.2


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hue (degrees 0-360), saturation (0-1), value (0-1)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(-1)
    c = v - x.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0.0)
    cc = np.maximum(c, 1e-6)
    h = np.where(v == r, ((g - b) / cc) % 6, np.where(v == g, (b - r) / cc + 2, (r - g) / cc + 4)) * 60.0
    h = np.where(c == 0, 0.0, h)
    return h, s, v


def color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    """Boolean mask of pixels matching a named color."""
    h, s, v = rgb_to_hsv(rgb)
    if color in CHROMATIC:
        in_hue = np.zeros(h.shape, bool)
        for lo, hi in CHROMATIC[color]:
            in_hue |= (h >= lo) & (h <= hi)
        return in_hue & (s >= MIN_SAT) & (v >= MIN_VAL)
    if color == "brown":
        return (h >= 10) & (h <= 40) & (s >= 0.3) & (v >= MIN_VAL) & (v < 0.6)
    if color == "black":
        return v < 0.2
    if color == "white":
        return (s < 0.15) & (v > 0.8)
    if color in ("gray", "silver"):
        return (s < 0.15) & (v >= 0.2) & (v <= 0.8)
    raise ValueError(f"unknown color {color!r}")


def color_fraction(rgb: np.ndarray, color: str) -> float:
    """Fraction of pixels in the crop matching the color (0 for empty crops)."""
    if rgb.size == 0:
        return 0.0
    return float(color_mask(rgb, color).mean())
