"""Generate printable ArUco marker PNGs at an exact physical size: python -m armvision.markers --out markers/"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
DPI = 300


def marker_image(marker_id: int, size_mm: float, dpi: int = DPI, margin_mm: float = 6.35):
    """Return (image, actual_size_mm). The black square is snapped to a multiple of 6 px (one module per 1/6)."""
    px = max(6, 6 * round(size_mm / 25.4 * dpi / 6))
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), marker_id, px)
    pad = round(margin_mm / 25.4 * dpi)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255), px / dpi * 25.4


def main(argv=None):
    from .config import load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="markers")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, cfg.mat.marker_size_mm, "mat") for i in sorted(cfg.mat.markers)]
    jobs.append((cfg.arm.gripper_marker_id, cfg.arm.gripper_marker_size_mm, "gripper"))
    for mid, size, kind in jobs:
        img, actual = marker_image(mid, size)
        path = out / f"{kind}_marker_{mid}.png"
        cv2.imwrite(str(path), img)
        note = "" if abs(actual - size) < 0.05 else f"  (!) snapped from {size} mm; set the config to {actual:.2f}"
        print(f"{path}: black square {actual:.2f} mm at {DPI} DPI{note}")
    print(f"Print at 100% scale ({DPI} DPI, no 'fit to page') and measure one marker with a ruler.")


if __name__ == "__main__":
    main()
