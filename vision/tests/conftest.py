import cv2
import numpy as np
import pytest

from armvision.config import ArmConfig, Config, MatConfig, VisionConfig, validate

SCALE = 3  # canvas px per mm


@pytest.fixture
def cfg():
    return validate(Config(
        MatConfig(400, 300, 40.0, {0: (30, 30), 1: (370, 30), 2: (370, 270), 3: (30, 270)}, 2),
        ArmConfig((-60, 150), 0, 0, 100, 10, 30.0),
        VisionConfig(px_per_mm=2.0, min_area_mm2=100.0, min_age=1)))


# canvas (mat at SCALE px/mm, y flipped) -> camera image, a trapezoid like an oblique view
SRC = np.float32([[0, 0], [400 * SCALE, 0], [400 * SCALE, 300 * SCALE], [0, 300 * SCALE]])
DST = np.float32([[250, 150], [1030, 150], [1230, 900], [50, 900]])
H_CANVAS2IMG = cv2.getPerspectiveTransform(SRC, DST)
IMG_SIZE = (1280, 960)


def mat_canvas(cfg):
    img = np.full((300 * SCALE, 400 * SCALE, 3), 190, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    px = int(cfg.mat.marker_size_mm * SCALE)
    for mid, (x, y) in cfg.mat.markers.items():
        m = cv2.aruco.generateImageMarker(d, mid, px)
        pad = 12
        m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        r0, c0 = int((300 - y) * SCALE - px / 2) - pad, int(x * SCALE - px / 2) - pad
        img[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = m[..., None]
    return img


def draw_rect(img, center_mm, size_mm, angle_deg, color):
    """Filled rotated rectangle; center in mat mm, angle of the long (first) side counter-clockwise."""
    a = np.radians(angle_deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array(center_mm, float)
    pts = [c + sx * size_mm[0] / 2 * u + sy * size_mm[1] / 2 * v for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    px = np.array([[p[0] * SCALE, (300 - p[1]) * SCALE] for p in pts], np.int32)
    cv2.fillPoly(img, [px], color)


def to_camera(canvas):
    return cv2.warpPerspective(canvas, H_CANVAS2IMG, IMG_SIZE, borderValue=(90, 90, 90))
