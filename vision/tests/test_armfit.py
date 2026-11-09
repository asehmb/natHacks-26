import cv2
import numpy as np
import pytest

from armvision.armfit import forward, project_base_points, q3_limits, skeleton
from armvision.intrinsics import Intrinsics
from armvision.pipeline import VisionSystem

from conftest import IMG_SIZE, PIN_K, mat_canvas, pin_camera

LINKS = (92, 135, 147, 87)


def test_forward_matches_mk2_equations():
    q1, q2, q3 = np.radians([12.0, 70.0, -100.0])
    L1, L2, L3, L4 = LINKS
    exp = [np.cos(q1) * (np.cos(q2 + q3) * L3 + np.cos(q2) * L2 + L4),
           np.sin(q1) * (np.cos(q2 + q3) * L3 + np.cos(q2) * L2 + L4),
           L1 + np.sin(q2) * L2 + np.sin(q2 + q3) * L3]
    assert np.allclose(forward([12.0, 70.0, -100.0], LINKS), exp)


def render_arm(cfg, vs, q, canvas_img):
    pts = project_base_points(skeleton(np.array([q], float), LINKS, 20)[0], vs.pose, vs.frames)
    cv2.polylines(canvas_img, [pts.astype(np.int32)], False, (40, 40, 40), 28)
    return canvas_img


@pytest.mark.xfail(reason="armfit is not wired into VisionSystem.process yet: the arm is only found via the "
                          "gripper ArUco marker", strict=False)
@pytest.mark.parametrize("q", [(10.0, 70.0, -100.0), (-15.0, 95.0, -120.0), (0.0, 55.0, -90.0)])
def test_markerless_arm_pose(cfg, q):
    assert q3_limits(q[1])[0] <= q[2] <= q3_limits(q[1])[1]
    vs = VisionSystem(cfg, Intrinsics(PIN_K, np.zeros(5), IMG_SIZE))
    assert vs.capture_reference(pin_camera(mat_canvas(cfg)))
    frame = render_arm(cfg, vs, q, pin_camera(mat_canvas(cfg)))
    state = vs.process(frame)
    assert state.arm.visible and state.objects == []
    truth = forward(q, LINKS)
    err = np.linalg.norm(state.arm.gripper_mm - truth)
    assert err < 25.0, (state.arm.gripper_mm, truth, state.arm.joints_deg)


def test_no_arm_when_mat_empty(cfg):
    vs = VisionSystem(cfg, Intrinsics(PIN_K, np.zeros(5), IMG_SIZE))
    vs.capture_reference(pin_camera(mat_canvas(cfg)))
    state = vs.process(pin_camera(mat_canvas(cfg)))
    assert not state.arm.visible and state.arm.gripper_mm is None
