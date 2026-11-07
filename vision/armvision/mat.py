"""Mat markers -> homography, camera pose, rectified view, and mat <-> arm-base transforms.

Mat frame: origin bottom-left corner of the mat, x right, y far, z up (right-handed).
Rectified image: row = (height - y) * px_per_mm, col = x * px_per_mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ArmConfig, Config, MatConfig
from .markers import DICT


class MarkerDetector:
    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """id -> (4, 2) float corners (TL, TR, BR, BL as seen on the marker)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._det.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def marker_corners_mat(center: tuple[float, float], size: float) -> np.ndarray:
    """Mat-frame 3D corners of a flat marker, same order as the detector (TL, TR, BR, BL; y is up)."""
    cx, cy = center
    h = size / 2
    return np.array([[cx - h, cy + h, 0], [cx + h, cy + h, 0], [cx + h, cy - h, 0], [cx - h, cy - h, 0]], np.float64)


def camera_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


@dataclass
class MatPose:
    H_mat2img: np.ndarray
    rvec: np.ndarray        # mat -> camera
    tvec: np.ndarray
    K: np.ndarray
    n_markers: int
    px_per_mm: float
    mat_height_mm: float
    mat_size_mm: tuple[float, float]

    def __post_init__(self):
        self.H_img2mat = np.linalg.inv(self.H_mat2img)
        self.R = cv2.Rodrigues(self.rvec)[0]
        self.cam_center = (-self.R.T @ self.tvec.reshape(3))
        s, h = self.px_per_mm, self.mat_height_mm
        self.M_mat2rect = np.array([[s, 0, 0], [0, -s, h * s], [0, 0, 1]], np.float64)
        self.M_img2rect = self.M_mat2rect @ self.H_img2mat

    @property
    def rect_size(self) -> tuple[int, int]:
        """(width, height) of the rectified image in px."""
        return round(self.mat_size_mm[0] * self.px_per_mm), round(self.mat_size_mm[1] * self.px_per_mm)

    def img_to_mat(self, pts) -> np.ndarray:
        return _apply_h(self.H_img2mat, pts)

    def mat_to_img(self, pts) -> np.ndarray:
        return _apply_h(self.H_mat2img, pts)

    def img_to_rect(self, pts) -> np.ndarray:
        return _apply_h(self.M_img2rect, pts)

    def rect_to_mat(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, np.float64).reshape(-1, 2)
        return np.c_[pts[:, 0] / self.px_per_mm, self.mat_height_mm - pts[:, 1] / self.px_per_mm]

    def mat_to_rect(self, pts) -> np.ndarray:
        return _apply_h(self.M_mat2rect, pts)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M_img2rect, self.rect_size, flags=cv2.INTER_LINEAR)

    def cam_to_mat(self, p_cam) -> np.ndarray:
        return self.R.T @ (np.asarray(p_cam, np.float64).reshape(3) - self.tvec.reshape(3))


def solve_mat(markers: dict[int, np.ndarray], cfg: Config, K: np.ndarray) -> MatPose | None:
    """Mat pose from whichever mat markers are visible, or None if fewer than cfg.mat.min_markers."""
    mat = cfg.mat
    ids = [i for i in markers if i in mat.markers]
    if len(ids) < mat.min_markers:
        return None
    obj = np.concatenate([marker_corners_mat(mat.markers[i], mat.marker_size_mm) for i in ids])
    img = np.concatenate([markers[i] for i in ids])
    H, _ = cv2.findHomography(obj[:, :2], img, 0)
    if H is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return MatPose(H, rvec, tvec, K, len(ids), cfg.vision.px_per_mm, mat.height_mm, (mat.width_mm, mat.height_mm))


class Frames:
    """Mat frame <-> arm base frame (x forward, y left, z up)."""

    def __init__(self, arm: ArmConfig):
        self.base = np.array(arm.base_xy_mm, np.float64)
        self.yaw = np.radians(arm.base_yaw_deg)
        self.yaw_deg = arm.base_yaw_deg
        self.mat_z = arm.mat_z_mm
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self._R = np.array([[c, s], [-s, c]])   # mat vector -> base vector

    def mat_to_base(self, xy) -> np.ndarray:
        return (np.asarray(xy, np.float64) - self.base) @ self._R.T

    def base_to_mat(self, xy) -> np.ndarray:
        return np.asarray(xy, np.float64) @ self._R + self.base

    def yaw_to_base(self, yaw_mat_deg: float) -> float:
        return yaw_mat_deg - self.yaw_deg
