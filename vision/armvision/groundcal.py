"""Ground-plane camera model from 4 clicked points on the sheet (no printed markers).

Frame: arm base frame, mm (x forward towards the camera, y left, z up), the mat surface is z = 0.
The 4 points give a homography ground -> image; the focal length comes out of that homography (principal point at
the image centre, square pixels), and solvePnP then gives the camera centre, which the container finder needs to
undo the perspective lean of tall objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from .pipeline import FrameSizeError

DEFAULT_FOV_DEG = 60.0
HFOV_RANGE_DEG = (25.0, 140.0)


def reference_points(cfg) -> np.ndarray:
    """(4, 2) arm-frame mm of the points to click, in click order: far-left corner, far-right corner, left tick,
    right tick. The far corners are the sheet's camera-end corners; the ticks are pen marks on the long edges,
    `sheet.mark_distance_mm` from the arm-end edge (the arm-end corners can be cut out, so they are never used).
    "Left" is the arm's left, i.e. the right side of the image when the camera faces the arm."""
    if cfg.sheet.points is not None:
        return np.asarray(cfg.sheet.points, np.float64).reshape(4, 2)
    s, a = cfg.sheet, cfg.arm
    x0 = -a.base_from_edge_mm                  # arm-end edge of the sheet
    yc = -a.base_offset_mm                     # sheet centre line (base is base_offset to its left)
    hw = s.width_mm / 2
    pts = np.array([[x0 + s.length_mm, yc + hw], [x0 + s.length_mm, yc - hw],
                    [x0 + s.mark_distance_mm, yc + hw], [x0 + s.mark_distance_mm, yc - hw]])
    t = math.radians(a.base_yaw_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return pts @ rot.T


def estimate_focal(H: np.ndarray, size: tuple[int, int]) -> float | None:
    """Focal length (px) from a ground -> image homography, principal point at the image centre.

    The first two columns of K^-1 H must be orthogonal and of equal length. Each constraint alone is ill-conditioned
    for some views; minimising both together is stable to a couple of pixels of click noise."""
    w, h = size
    M = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]]) @ np.asarray(H, np.float64)

    def cost(f):
        f = np.asarray(f, np.float64)[:, None]
        h1, h2 = M[:, 0], M[:, 1]
        a1 = np.hstack([h1[:2] / f, np.full_like(f, h1[2])])
        a2 = np.hstack([h2[:2] / f, np.full_like(f, h2[2])])
        n1, n2 = np.sum(a1 * a1, 1), np.sum(a2 * a2, 1)
        g = np.sum(a1 * a2, 1) / np.sqrt(n1 * n2)
        n = (n1 - n2) / (n1 + n2)
        return g * g + n * n

    grid = np.geomspace(0.2 * w, 5 * w, 600)
    c = cost(grid)
    i = int(np.argmin(c))
    if not np.isfinite(c[i]) or i in (0, len(grid) - 1):
        return None
    fine = np.linspace(grid[i - 1], grid[i + 1], 200)
    return float(fine[int(np.argmin(cost(fine)))])


def _hfov(f: float, width: int) -> float:
    return math.degrees(2 * math.atan(width / (2 * f)))


@dataclass
class GroundCalibration:
    K: np.ndarray
    R: np.ndarray            # ground (arm frame) -> camera
    t: np.ndarray
    H_base2img: np.ndarray   # (x, y, 1) mm -> pixels
    size: tuple[int, int]    # (width, height) the calibration is valid for
    reproj_rms: float = 0.0
    fov_source: str = "estimated"

    def __post_init__(self):
        self.K, self.R = np.asarray(self.K, np.float64), np.asarray(self.R, np.float64)
        self.t = np.asarray(self.t, np.float64).reshape(3)
        self.H_base2img = np.asarray(self.H_base2img, np.float64)
        self._Hinv = np.linalg.inv(self.H_base2img)
        c = -self.R.T @ self.t
        self.camera_center = c
        self.nadir_xy = c[:2].copy()
        self.cam_height = float(c[2])

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    @property
    def hfov_deg(self) -> float:
        return _hfov(self.focal, self.size[0])

    @classmethod
    def from_points(cls, img_pts, base_pts, size, fov_deg: float | None = None) -> "GroundCalibration":
        img = np.asarray(img_pts, np.float64).reshape(-1, 2)
        base = np.asarray(base_pts, np.float64).reshape(-1, 2)
        if len(img) != len(base) or len(img) < 4:
            raise ValueError("need at least 4 matching points")
        size = (int(size[0]), int(size[1]))
        H, _ = cv2.findHomography(base, img, 0)
        if H is None:
            raise ValueError("cannot compute a homography from these points (are some of them collinear?)")
        f = None if fov_deg is not None else estimate_focal(H, size)
        source = "estimated"
        if f is None or not (HFOV_RANGE_DEG[0] <= _hfov(f, size[0]) <= HFOV_RANGE_DEG[1]):
            fov = fov_deg if fov_deg is not None else DEFAULT_FOV_DEG
            f = size[0] / (2 * math.tan(math.radians(fov) / 2))
            source = "config" if fov_deg is not None else "fallback"
        K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
        obj = np.c_[base, np.zeros(len(base))]
        best = None
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
            rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))
            if best is None or (math.isfinite(rms) and not rms >= best[0]):   # a NaN fit never wins
                best = (rms, R, tv.reshape(3))
        if best is None or not (-best[1].T @ best[2])[2] > 0:   # NaN, or the camera is below the sheet
            raise ValueError("the clicked points put the camera below the sheet; "
                             "check the click order (far-left corner, far-right corner, left tick, right tick)")
        return cls(K, best[1], best[2], H, size, best[0], source)

    def project(self, xyz) -> np.ndarray:
        """(N, 3) arm-frame mm -> (N, 2) pixels."""
        p = np.asarray(xyz, np.float64).reshape(-1, 3)
        c = p @ self.R.T + self.t
        uv = c @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def img_to_ground(self, pts) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) arm-frame mm on the z = 0 plane."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        q = np.c_[p, np.ones(len(p))] @ self._Hinv.T
        return q[:, :2] / q[:, 2:3]

    def rescaled(self, size) -> "GroundCalibration":
        w, h = int(size[0]), int(size[1])
        if (w, h) == self.size:
            return self
        if abs(w / h - self.size[0] / self.size[1]) > 0.01 * self.size[0] / self.size[1]:
            raise FrameSizeError(f"frame is {w}x{h} but the calibration was made at "
                                 f"{self.size[0]}x{self.size[1]}; recalibrate or fix the camera aspect ratio")
        s = w / self.size[0]
        S = np.diag([s, s, 1.0])
        return GroundCalibration(S @ self.K, self.R, self.t, S @ self.H_base2img, (w, h),
                                 self.reproj_rms * s, self.fov_source)

    def save(self, path) -> None:
        data = {"size": list(self.size), "K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist(),
                "H_base2img": self.H_base2img.tolist(), "reproj_rms": float(self.reproj_rms),
                "fov_source": self.fov_source}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path) -> "GroundCalibration":
        try:
            d = yaml.safe_load(Path(path).read_text())
            return cls(np.array(d["K"]), np.array(d["R"]), np.array(d["t"]), np.array(d["H_base2img"]),
                       tuple(d["size"]), float(d.get("reproj_rms", 0.0)), d.get("fov_source", "estimated"))
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot read calibration {path}: {exc}") from exc
