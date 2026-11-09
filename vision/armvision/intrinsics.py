"""One-time camera calibration from views of the mat markers; save/load camera_intrinsics.yaml."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import Config
from .mat import camera_matrix, marker_corners_mat

DEFAULT_PATH = Path("camera_intrinsics.yaml")


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]     # (width, height)
    calibrated: bool = True
    rms: float | None = None

    def save(self, path=DEFAULT_PATH):
        Path(path).write_text(yaml.safe_dump({
            "image_size": list(self.image_size), "camera_matrix": self.K.tolist(),
            "dist_coeffs": self.dist.reshape(-1).tolist(), "rms": self.rms}))

    def scaled(self, new_size: tuple[int, int]) -> Intrinsics:
        """Return a copy of Intrinsics scaled to a new image resolution (width, height)."""
        w, h = int(new_size[0]), int(new_size[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid image size {new_size}")
        w0, h0 = self.image_size
        sx, sy = w / w0, h / h0
        K_new = self.K.copy().astype(np.float64)
        K_new[0, 0] *= sx
        K_new[0, 2] *= sx
        K_new[1, 1] *= sy
        K_new[1, 2] *= sy
        return Intrinsics(K_new, self.dist.copy(), (w, h), self.calibrated, self.rms)


def load_intrinsics(path=DEFAULT_PATH) -> Intrinsics | None:
    path = Path(path)
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text())
    return Intrinsics(np.array(raw["camera_matrix"], np.float64), np.array(raw["dist_coeffs"], np.float64),
                      tuple(raw["image_size"]), True, raw.get("rms"))


def fallback_intrinsics(width: int, height: int, fov_deg: float) -> Intrinsics:
    warnings.warn("camera_intrinsics.yaml not found: using a field-of-view estimate. Object positions are "
                  "unaffected, but the gripper's 3D position will be approximate. Run the calibrate mode.",
                  stacklevel=2)
    return Intrinsics(camera_matrix(width, height, fov_deg), np.zeros(5), (width, height), calibrated=False)


def view_points(markers: dict[int, np.ndarray], cfg: Config):
    """(object points, image points) for one view, or None if too few mat markers are visible."""
    ids = [i for i in markers if i in cfg.mat.markers]
    if len(ids) < cfg.mat.min_markers:
        return None
    obj = np.concatenate([marker_corners_mat(cfg.mat.markers[i], cfg.mat.marker_size_mm) for i in ids])
    img = np.concatenate([markers[i] for i in ids])
    return obj.astype(np.float32), img.astype(np.float32)


def calibrate(views: list[tuple[np.ndarray, np.ndarray]], image_size: tuple[int, int]) -> Intrinsics:
    """views: list of view_points() results; needs at least ~5, ideally ~15 from varied angles."""
    if len(views) < 5:
        raise ValueError(f"need at least 5 views, got {len(views)}")
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size {image_size}")
    for i, (obj, img) in enumerate(views):
        if len(obj) != len(img) or len(obj) == 0:
            raise ValueError(f"view {i}: empty or mismatched points")
    K0 = camera_matrix(w, h, 60.0)
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [v[0] for v in views], [v[1] for v in views], image_size, K0, None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
    return Intrinsics(K, dist, image_size, True, float(rms))
