"""Shared drawing: a footprint box per tracked object with a green->red confidence meter and a "green screwdriver" tag."""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def confidence_color(conf: float) -> tuple[int, int, int]:
    """BGR: red at 0 through yellow to green at 1."""
    hue = int(60 * min(max(conf, 0.0), 1.0))          # OpenCV hue: 0 red, 30 yellow, 60 green
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def object_name(color: str, label: str) -> str:
    """"green screwdriver"; no duplicated colour ("red cup" stays "red cup"), and "unknown" colours are dropped."""
    parts = [] if color in ("", "unknown") or label.lower().startswith(color.lower()) else [color]
    return " ".join(parts + [label])


def footprint_img(vs, o) -> np.ndarray:
    """Object footprint (size_mm, oriented along its long axis) as 4 image points."""
    length, width = o.size_mm
    axis = np.radians(o.grasp_yaw_deg - 90.0)       # grasp yaw is perpendicular to the long axis (base frame)
    d, n = np.array([np.cos(axis), np.sin(axis)]), np.array([-np.sin(axis), np.cos(axis)])
    c = np.asarray(o.center_mm[:2], np.float64)
    corners = [c + d * length / 2 + n * width / 2, c + d * length / 2 - n * width / 2,
               c - d * length / 2 - n * width / 2, c - d * length / 2 + n * width / 2]
    return vs.pose.mat_to_img(np.array([vs.frames.base_to_mat(p) for p in corners]))


def _text(img, t, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, t, org, FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, t, org, FONT, scale, color, 2, cv2.LINE_AA)


def draw_object(out: np.ndarray, vs, o) -> None:
    col = confidence_color(o.confidence)
    box = footprint_img(vs, o).astype(np.int32)
    cv2.polylines(out, [box], True, col, 2 if not o.occluded else 1, cv2.LINE_AA)
    x, y = int(box[:, 0].min()), int(box[:, 1].min())
    name = f"#{o.id} {object_name(o.color, o.label)}" + (" (hidden)" if o.occluded else "")
    _text(out, name, (x, max(y - 30, 18)))
    bw, bh, by = 100, 10, max(y - 22, 26)
    cv2.rectangle(out, (x, by), (x + bw, by + bh), (40, 40, 40), -1)
    cv2.rectangle(out, (x, by), (x + int(bw * o.confidence), by + bh), col, -1)
    cv2.rectangle(out, (x, by), (x + bw, by + bh), (255, 255, 255), 1)
    _text(out, f"{o.confidence:.0%}", (x + bw + 8, by + bh), 0.55)
