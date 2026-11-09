"""Markerless arm pose: fit the EEZYbot Mk2's forward-kinematics skeleton (q1, q2, q3) to the arm silhouette.

The arm moves in one vertical plane, so its whole shape is three angles. Candidate skeletons are projected through the
mat-calibrated camera into the image and scored by chamfer distance against the arm silhouette, which resolves the
gripper's height as well as its position. The result is the same end-effector point inverseKinematics() uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .mat import Frames, MatPose

# Mk2 joint limits from the firmware servo limits (firmware/comms/combined_only/servo_bridge.h), mapped as in
# Kinematics/kinematic_model.py: servo1 = -2.0497*q1 + 91.726, servo2 = 180 - q2, servo3 = q2 + q3 + 135.
SERVO_LIMITS_DEG = ((0, 180), (20, 160), (15, 130))
Q1_RANGE = ((91.726 - SERVO_LIMITS_DEG[0][1]) / 2.0497, (91.726 - SERVO_LIMITS_DEG[0][0]) / 2.0497)
Q2_RANGE = (180.0 - SERVO_LIMITS_DEG[1][1], 180.0 - SERVO_LIMITS_DEG[1][0])
CAP_MM = 80.0
MIN_MASK_PX = 40
MAX_COST_MM = 40.0


def q3_limits(q2):
    """Mk2 limits for joint 3 given joint 2 (degrees), from the elbow servo limits."""
    return SERVO_LIMITS_DEG[2][0] - 135 - q2, SERVO_LIMITS_DEG[2][1] - 135 - q2


def skeleton(q: np.ndarray, links, n: int = 10) -> np.ndarray:
    """(C, 3) joint angles in degrees -> (C, P, 3) points along base, shoulder, elbow, wrist, end effector (base frame)."""
    L1, L2, L3, L4 = links
    q = np.atleast_2d(np.radians(q))
    c1, s1 = np.cos(q[:, 0]), np.sin(q[:, 0])
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    c23, s23 = np.cos(q[:, 1] + q[:, 2]), np.sin(q[:, 1] + q[:, 2])
    C = len(q)
    zero = np.zeros(C)
    base = np.stack([zero, zero, zero], 1)
    sh = np.stack([zero, zero, np.full(C, L1)], 1)
    el = sh + L2 * np.stack([c1 * c2, s1 * c2, s2], 1)
    wr = el + L3 * np.stack([c1 * c23, s1 * c23, s23], 1)
    ee = wr + L4 * np.stack([c1, s1, zero], 1)
    t = np.linspace(0, 1, n, endpoint=False)[None, :, None]
    segs = [a[:, None] + (b - a)[:, None] * t for a, b in [(base, sh), (sh, el), (el, wr), (wr, ee)]]
    return np.concatenate(segs + [ee[:, None]], 1)


def forward(q, links) -> np.ndarray:
    """End-effector (x, y, z) in the base frame; identical to EEZYbotARM_Mk2 forward kinematics."""
    return skeleton(np.asarray(q, float), links, 1)[0, -1]


def project_base_points(p_base: np.ndarray, pose: MatPose, frames: Frames) -> np.ndarray:
    """(..., 3) points in the arm base frame -> (..., 2) pixels in the (undistorted) camera image."""
    p_base = np.asarray(p_base, np.float64)
    xy = frames.base_to_mat(p_base[..., :2])
    pm = np.concatenate([xy, (p_base[..., 2] - frames.mat_z)[..., None]], -1)
    pc = pm @ pose.R.T + pose.tvec.reshape(3)
    uv = pc @ pose.K.T
    return uv[..., :2] / uv[..., 2:3]


def shoulder_scale(pose: MatPose, frames: Frames, links):
    """Shoulder pixel and the image scale (px per mm) at its depth."""
    p = np.array([0.0, 0.0, links[0]])
    px = project_base_points(p, pose, frames)
    xy = frames.base_to_mat(p[:2])
    depth = (np.array([xy[0], xy[1], p[2] - frames.mat_z]) @ pose.R.T + pose.tvec.reshape(3))[2]
    return px, float(pose.K[0, 0] / depth)


@dataclass
class ArmFit:
    q_deg: np.ndarray
    ee_mm: np.ndarray       # base frame
    cost_mm: float
    visible: bool


class ArmFitter:
    def __init__(self, links):
        self.links, self.s = tuple(links), 1.0
        self.prev: np.ndarray | None = None

    def reset(self):
        self.prev = None

    def _grid(self, centers, half, step):
        axes = []
        for c, h, lim in zip(centers, half, (Q1_RANGE, Q2_RANGE, (-180.0, 180.0))):
            lo, hi = max(c - h, lim[0]), min(c + h, lim[1])
            axes.append(np.arange(lo, hi + 1e-6, step))
        q = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        lo3, hi3 = q3_limits(q[:, 1])
        return q[(q[:, 2] >= lo3) & (q[:, 2] <= hi3)]

    def _cost(self, q, pose, frames, dist, pts_px, shape):
        out = np.empty(len(q))
        h, w = shape
        for i in range(0, len(q), 400):
            sk = skeleton(q[i:i + 400], self.links)
            rc = project_base_points(sk, pose, frames)
            ci, ri = np.round(rc[..., 0]).astype(int), np.round(rc[..., 1]).astype(int)
            inb = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
            d1 = np.where(inb, dist[np.clip(ri, 0, h - 1), np.clip(ci, 0, w - 1)], 0.0)
            d1 = np.minimum(d1 / self.s, CAP_MM)
            t1 = d1.sum(1) / np.maximum(inb.sum(1), 1)
            t1 = np.where(inb.sum(1) == 0, CAP_MM, t1)
            diff = pts_px[None, :, None, :] - rc[:, None, :, :].astype(np.float32)   # (C, K, P, 2)
            d2 = np.sqrt((diff ** 2).sum(-1)).min(2) / self.s
            out[i:i + 400] = t1 + np.minimum(d2, CAP_MM).mean(1)
        return out

    def fit(self, mask: np.ndarray, px_per_mm: float, pose: MatPose, frames: Frames) -> ArmFit | None:
        nz = cv2.findNonZero(mask)
        if nz is None or len(nz) < MIN_MASK_PX:
            self.prev = None
            return None
        self.s = px_per_mm
        pts = nz.reshape(-1, 2)
        pts_px = pts[np.linspace(0, len(pts) - 1, min(len(pts), 100)).astype(int)].astype(np.float32)
        dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)

        def best(q):
            c = self._cost(q, pose, frames, dist, pts_px, mask.shape)
            k = int(np.argmin(c))
            return q[k], float(c[k])

        if self.prev is None:
            qb, cb = best(self._grid((0.0, 79.5, -70.0), (30.0, 41.0, 110.0), 6.0))
            windows = [(6.0, 3.0), (3.0, 1.5), (1.5, 0.75)]
        else:
            qb, cb = best(self._grid(self.prev, (8.0, 8.0, 10.0), 2.5))
            windows = [(3.0, 1.5), (1.5, 0.75)]
        for half, step in windows:
            q, c = best(self._grid(qb, (half,) * 3, step))
            if c <= cb:
                qb, cb = q, c
        ok = cb < MAX_COST_MM
        self.prev = qb if ok else None
        return ArmFit(qb, forward(qb, self.links), cb, ok)
