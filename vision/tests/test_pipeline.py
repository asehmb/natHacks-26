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


def test_stale_object_not_returned_by_find(cfg):
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    draw_rect(canvas, (200, 150), (40, 40), 0, (0, 0, 220))   # red block
    state1 = vs.process(to_camera(canvas))
    assert len(state1.objects) == 1 and state1.objects[0].seen is True
    assert len(state1.find(color="red")) == 1

    # Remove the object in next frame (empty canvas with mat markers)
    empty_canvas = mat_canvas(cfg)
    state2 = vs.process(to_camera(empty_canvas))
    assert len(state2.objects) == 1                      # track kept briefly for continuity
    assert state2.objects[0].seen is False               # marked not seen this frame
    assert state2.objects[0].missed == 1
    assert state2.find(color="red") == []                # excluded from pickup selection by default!
    assert len(state2.find(color="red", allow_stale=True)) == 1


def test_capture_reference_requires_current_markers(cfg):
    vs = VisionSystem(cfg)
    # 1. capture_reference on blank frame (no markers) must fail
    blank = np.zeros((*IMG_SIZE[::-1], 3), np.uint8)
    assert not vs.capture_reference(blank)

    # 2. Establish a pose
    valid = to_camera(mat_canvas(cfg))
    vs.process(valid)
    assert vs.pose is not None

    # 3. Now attempt capture_reference with blank frame again (no markers visible now)
    # Must FAIL and NOT reuse the old pose!
    assert not vs.capture_reference(blank)


def test_lost_markers_tracked_and_expires(cfg):
    cfg.vision.max_pose_age = 2
    vs = make_system(cfg)
    valid = to_camera(mat_canvas(cfg))
    s1 = vs.process(valid)
    assert s1.mat_visible is True and s1.pose_age == 0 and s1.pose_fresh is True
    assert vs.pose is not None

    # Frame with no markers visible
    blank = np.zeros((*IMG_SIZE[::-1], 3), np.uint8)
    s2 = vs.process(blank)
    assert s2.mat_visible is False and s2.pose_age == 1 and s2.pose_fresh is False
    assert vs.pose is not None  # within grace period

    s3 = vs.process(blank)
    assert s3.mat_visible is False and s3.pose_age == 2 and s3.pose_fresh is False
    assert vs.pose is not None  # still at limit

    # Exceeds max_pose_age -> pose must expire
    s4 = vs.process(blank)
    assert s4.mat_visible is False and s4.pose_age == 3 and s4.pose_fresh is False
    assert vs.pose is None


def test_min_age_hides_flicker(cfg):
    import dataclasses
    cfg.vision = dataclasses.replace(cfg.vision, min_age=3)
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    draw_rect(canvas, (200, 150), (60, 20), 0, (0, 0, 220))
    frame = to_camera(canvas)
    assert [len(vs.process(frame).objects) for _ in range(4)] == [0, 0, 1, 1]


def test_exposure_shift_and_soft_shadow_are_not_objects(cfg):
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    dim = (canvas.astype(np.float32) * 0.85).astype(np.uint8)       # auto-exposure drift
    assert vs.process(to_camera(dim)).objects == []
    shadow = canvas.copy()
    shadow[200:400, 300:500] = (shadow[200:400, 300:500].astype(np.float32) * 0.92).astype(np.uint8)
    assert vs.process(to_camera(shadow)).objects == []


def test_marker_misalignment_is_not_an_object(cfg):
    vs = make_system(cfg)
    canvas = mat_canvas(cfg)
    shifted = np.roll(canvas, 3, axis=1)                             # printed markers sit 3 px off the reference
    assert vs.process(to_camera(shifted)).objects == []
