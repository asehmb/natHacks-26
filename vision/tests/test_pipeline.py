import json

import numpy as np
import pytest

from armvision.mat import MarkerDetector, solve_mat, camera_matrix
from armvision.pipeline import VisionSystem

from conftest import H_CANVAS2IMG, IMG_SIZE, SCALE, draw_rect, mat_canvas, to_camera


def canvas_to_mat(px):
    return np.array([px[0] / SCALE, 300 - px[1] / SCALE])


def test_homography_recovers_known_points(cfg):
    frame = to_camera(mat_canvas(cfg))
    K = camera_matrix(*IMG_SIZE, 60)
    pose = solve_mat(MarkerDetector().detect(frame), cfg, K)
    assert pose is not None and pose.n_markers == 4
    for mat_pt in [(200, 150), (50, 250), (350, 60)]:
        canvas_px = np.array([[mat_pt[0] * SCALE, (300 - mat_pt[1]) * SCALE, 1.0]])
        img = canvas_px @ H_CANVAS2IMG.T
        img = img[:, :2] / img[:, 2:]
        assert np.allclose(pose.img_to_mat(img)[0], mat_pt, atol=2.0)


def test_too_few_markers_gives_none(cfg):
    assert solve_mat({}, cfg, camera_matrix(640, 480, 60)) is None


def make_system(cfg):
    vs = VisionSystem(cfg)
    empty = to_camera(mat_canvas(cfg))
    assert vs.capture_reference(empty)
    return vs


def test_object_found_with_grasp_in_base_frame(cfg):
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    draw_rect(canvas, (200, 150), (60, 20), 0, (0, 0, 220))   # red bar along x
    state = vs.process(to_camera(canvas))
    assert state.mat_visible and len(state.objects) == 1
    o = state.objects[0]
    assert o.color == "red"
    # base frame: base at (-60, 150) with yaw 0 -> centre (260, 0)
    assert np.allclose(o.center_mm[:2], [260, 0], atol=2.5)
    assert o.size_mm[0] == pytest.approx(60, abs=4) and o.size_mm[1] == pytest.approx(20, abs=4)
    # grasp at the end nearest the base (x smaller), inset 15 from the end at x=230
    assert np.allclose(o.grasp_mm[:2], [245, 0], atol=3.5)
    assert o.grasp_mm[2] == 15
    json.dumps(state.to_dict())


def test_arm_blob_not_reported_as_object(cfg):
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    draw_rect(canvas, (60, 150), (120, 25), 0, (60, 60, 60))   # link touching the base zone (x<40)
    draw_rect(canvas, (300, 80), (40, 40), 0, (200, 80, 0))    # blue block
    state = vs.process(to_camera(canvas))
    assert [o.color for o in state.objects] == ["blue"]
    assert vs.last_arm_mask.any()


def test_no_reference_means_no_objects(cfg):
    vs = VisionSystem(cfg)
    state = vs.process(to_camera(mat_canvas(cfg)))
    assert state.mat_visible and state.objects == []


def test_find_helper(cfg):
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    draw_rect(canvas, (150, 80), (40, 40), 0, (0, 0, 220))
    draw_rect(canvas, (300, 220), (40, 40), 0, (200, 80, 0))
    state = vs.process(to_camera(canvas))
    assert [o.color for o in state.find(color="blue")] == ["blue"]
    assert state.find(position="left")[0].color == "blue"     # y base = mat_y - 150 -> blue is +70, left
    assert state.find(position="far")[0].color == "blue"
    with pytest.raises(ValueError):
        state.find(position="sideways")
