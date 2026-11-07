"""Contour (mat mm) -> long axis, grasp point near the end closest to the arm, gripper yaw. Pure numpy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grasp:
    center: np.ndarray      # centre of the minimum-area rectangle (mm)
    length: float
    width: float
    axis_deg: float         # long-axis direction in (-90, 90]
    point: np.ndarray       # grasp point (mm)
    yaw_deg: float          # gripper yaw, perpendicular to the long axis, in (-90, 90]


def _wrap180(deg: float) -> float:
    """Wrap to (-90, 90]."""
    d = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if d == -90.0 else d


def min_area_rect(points: np.ndarray, step_deg: float = 1.0):
    """Brute-force rotating bounding box. Returns (center, length, width, long_axis_deg)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    angles = np.radians(np.arange(0.0, 180.0, step_deg))
    u = np.stack([np.cos(angles), np.sin(angles)], 1)           # (A, 2) candidate axes
    v = np.stack([-np.sin(angles), np.cos(angles)], 1)
    pu, pv = pts @ u.T, pts @ v.T                               # (N, A)
    ext_u, ext_v = pu.max(0) - pu.min(0), pv.max(0) - pv.min(0)
    k = int(np.argmin(ext_u * ext_v))
    cu, cv_ = (pu[:, k].max() + pu[:, k].min()) / 2, (pv[:, k].max() + pv[:, k].min()) / 2
    center = cu * u[k] + cv_ * v[k]
    a, b = ext_u[k], ext_v[k]
    ang = np.degrees(angles[k])
    if b > a:
        a, b, ang = b, a, ang + 90.0
    return center, float(a), float(b), _wrap180(ang)


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return p.mean(0)
    return np.array([((x + x2) * cross).sum(), ((y + y2) * cross).sum()]) / (6 * a)


def compute_grasp(points: np.ndarray, base_xy, inset_mm: float = 15.0, round_aspect: float = 1.3) -> Grasp:
    center, length, width, axis_deg = min_area_rect(points)
    yaw = _wrap180(axis_deg + 90.0)
    if length / max(width, 1e-9) < round_aspect:
        return Grasp(center, length, width, axis_deg, polygon_centroid(points), yaw)
    d = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))])
    ends = [center + d * length / 2, center - d * length / 2]
    base = np.asarray(base_xy, np.float64)
    end = min(ends, key=lambda e: float(np.linalg.norm(e - base)))
    inward = (center - end) / (length / 2)
    return Grasp(center, length, width, axis_deg, end + inward * min(inset_mm, length / 2), yaw)
