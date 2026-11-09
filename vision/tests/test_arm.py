import cv2
import numpy as np
import pytest

from armvision.arm import arm_mask, marker_pose_cam
from armvision.mat import MarkerDetector, camera_matrix


def test_blob_touching_base_zone_is_arm():
    fg = np.zeros((300, 400), np.uint8)
    cv2.rectangle(fg, (0, 140), (150, 160), 255, -1)      # arm link from the base
    cv2.rectangle(fg, (300, 50), (340, 90), 255, -1)      # separate object
    m = arm_mask(fg, (0, 150), 40)
    assert m[150, 100] > 0 and m[70, 320] == 0


def test_blob_touching_gripper_marker_is_arm():
    fg = np.zeros((300, 400), np.uint8)
    cv2.rectangle(fg, (300, 50), (340, 90), 255, -1)
    m = arm_mask(fg, (0, 150), 40, marker_px=(320, 70), marker_radius_px=10)
    assert m[70, 320] > 0


def test_gripper_pose_from_solvepnp():
    K = camera_matrix(640, 480, 60)
    size = 30.0
    px = 180
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 10, px)
    img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    # marker frame is y-up with z towards the viewer, i.e. camera z pointing into the page: flip about x, then tilt
    R = cv2.Rodrigues(np.array([0.4, -0.3, 0.1]))[0] @ cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    rvec = cv2.Rodrigues(R)[0].reshape(3)
    tvec = np.array([20.0, -10.0, 150.0])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    pad = 30 * size / px
    outer = np.array([[-h - pad, h + pad, 0], [h + pad, h + pad, 0], [h + pad, -h - pad, 0], [-h - pad, -h - pad, 0]])
    po, _ = cv2.projectPoints(outer, rvec, tvec, K, None)
    src = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    M = cv2.getPerspectiveTransform(src, po.reshape(4, 2).astype(np.float32))
    frame = cv2.warpPerspective(img, M, (640, 480), borderValue=255)
    found = MarkerDetector().detect(frame)
    assert 10 in found
    R, t = marker_pose_cam(found[10], size, K)
    assert np.linalg.norm(t - tvec) < 5.0
