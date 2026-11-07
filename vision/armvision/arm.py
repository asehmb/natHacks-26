"""Gripper marker 3D pose and arm silhouette."""
from __future__ import annotations

import cv2
import numpy as np


def marker_pose_cam(corners: np.ndarray, size_mm: float, K: np.ndarray):
    """Pose of a flat square marker in the camera frame: (R, t) with t = marker centre (mm), or None."""
    h = size_mm / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float64).reshape(4, 1, 2), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def arm_mask(fg_mask: np.ndarray, base_center_px, base_radius_px: float,
             marker_px=None, marker_radius_px: float = 0.0) -> np.ndarray:
    """Foreground blobs touching the base-zone circle or the gripper marker's rectified position."""
    seed = np.zeros(fg_mask.shape, np.uint8)
    cv2.circle(seed, (int(round(base_center_px[0])), int(round(base_center_px[1]))), int(round(base_radius_px)), 255, -1)
    if marker_px is not None:
        cv2.circle(seed, (int(round(marker_px[0])), int(round(marker_px[1]))),
                   max(1, int(round(marker_radius_px))), 255, -1)
    # bridge small gaps so the arm's links join into one blob
    joined = cv2.dilate(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels = cv2.connectedComponents(joined)
    hit = np.unique(labels[(seed > 0) & (joined > 0)])
    out = np.isin(labels, hit[hit > 0]) & (fg_mask > 0)
    return out.astype(np.uint8) * 255
