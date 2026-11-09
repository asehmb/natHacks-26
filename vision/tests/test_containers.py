"""Container finder on synthetic 3D scenes (no GUI, no camera)."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from armvision.config import ConfigError
from armvision.containers import (ContainerConfig, ContainerFinder, ContainersConfig, TrackingConfig, ZoneConfig,
                                  circle_fit, load_container_config, validate_container_config)
from armvision.groundcal import GroundCalibration, reference_points
from armvision.pipeline import FrameSizeError

SIZE = (1280, 720)
F = 900.0
RED, BLUE, GREY = (40, 40, 200), (200, 80, 30), (120, 120, 120)


def true_camera(center=(520.0, 0.0, 350.0), target=(200.0, 0.0, 40.0), f=F, size=SIZE):
    c, t = np.array(center), np.array(target)
    fwd = (t - c) / np.linalg.norm(t - c)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    pts = reference_points(ContainerConfig())
    H, _ = cv2.findHomography(pts, project(R, -R @ c, K, np.c_[pts, np.zeros(4)]))
    return GroundCalibration(K, R, -R @ c, H, size)


def project(R, t, K, xyz):
    q = (np.asarray(xyz, float) @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:3]


def paint(img, cam, pts3d, color):
    hull = cv2.convexHull(cam.project(pts3d).astype(np.float32)).reshape(-1, 2)
    cv2.fillConvexPoly(img, np.round(hull).astype(np.int32), color)


def frustum(c, r_base, r_top, h, n=60):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    return np.vstack([np.c_[c[0] + r_base * ring[:, 0], c[1] + r_base * ring[:, 1], np.zeros(n)],
                      np.c_[c[0] + r_top * ring[:, 0], c[1] + r_top * ring[:, 1], np.full(n, h)]])


def box(c, length, width, h, yaw_deg):
    t = math.radians(yaw_deg)
    u, v = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    xy = [np.array(c) + sx * length / 2 * u + sy * width / 2 * v for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return np.array([[*p, z] for z in (0, h) for p in xy])


def bowl(c, r, h, n=60):
    """Sphere-cap bowl: widest at the rim, narrow base."""
    out = [frustum(c, 0.45 * r, 0.45 * r, 0.0, n)]
    for z in np.linspace(0, h, 6)[1:]:
        out.append(frustum(c, (0.45 + 0.55 * (z / h) ** 0.5) * r, (0.45 + 0.55 * (z / h) ** 0.5) * r, z, n)[n:])
    return np.vstack(out)


def scene(cam, solids):
    img = np.full((SIZE[1], SIZE[0], 3), 215, np.uint8)
    for pts, color in solids:
        paint(img, cam, pts, color)
    return img


def make_finder(cam, heights=None, **kw):
    cfg = validate_container_config(ContainerConfig(
        containers=ContainersConfig(heights_mm=heights or {}),
        tracking=TrackingConfig(min_age=kw.pop("min_age", 1))))
    return ContainerFinder(cfg, cam)


def run(finder, img, frames=3):
    for _ in range(frames):
        state = finder.process(img)
    return state


@pytest.fixture(scope="module")
def cam():
    return true_camera()


# ------------------------------------------------------------------------------------------ calibration
def test_reference_points_match_sheet_and_arm_config():
    pts = reference_points(ContainerConfig())
    np.testing.assert_allclose(pts, [[300, 215], [300, -215], [150, 215], [150, -215]])
    cfg = ContainerConfig()
    cfg.arm.base_offset_mm = 30
    np.testing.assert_allclose(reference_points(cfg)[:, 1], [185, -245, 185, -245])
    cfg = ContainerConfig()
    cfg.arm.base_yaw_deg = 90
    np.testing.assert_allclose(reference_points(cfg)[2], [-215, 150], atol=1e-9)
    cfg.sheet.points = [[1, 2], [3, 4], [5, 6], [7, 8]]
    np.testing.assert_allclose(reference_points(cfg)[3], [7, 8])


def calib_from(cam, noise=0.0, seed=0):
    base = reference_points(ContainerConfig())
    img = cam.project(np.c_[base, np.zeros(4)]) + np.random.default_rng(seed).normal(0, noise, (4, 2))
    return GroundCalibration.from_points(img, base, SIZE)


def test_focal_and_camera_centre_without_noise(cam):
    est = calib_from(cam)
    assert est.focal == pytest.approx(F, rel=0.01)
    np.testing.assert_allclose(est.camera_center, cam.camera_center, rtol=0.02, atol=2)
    assert est.cam_height == pytest.approx(350, rel=0.02)
    assert est.fov_source == "estimated"


def test_focal_with_one_pixel_noise(cam):
    est = [calib_from(cam, 1.0, s) for s in range(20)]
    assert np.median([abs(e.focal / F - 1) for e in est]) < 0.04
    assert np.median([np.linalg.norm(e.camera_center - cam.camera_center) for e in est]) < 0.04 * 520


def test_save_load_and_rescale(cam, tmp_path):
    est = calib_from(cam)
    est.save(tmp_path / "c.yaml")
    back = GroundCalibration.load(tmp_path / "c.yaml")
    np.testing.assert_allclose(back.camera_center, est.camera_center)
    big = est.rescaled((1920, 1080))
    p = np.array([[200.0, 50.0, 0.0]])
    np.testing.assert_allclose(big.project(p), est.project(p) * 1.5, atol=1e-6)
    np.testing.assert_allclose(big.camera_center, est.camera_center, atol=1e-6)
    np.testing.assert_allclose(big.img_to_ground(big.project(p)), [[200, 50]], atol=1e-2)
    with pytest.raises(FrameSizeError):
        est.rescaled((720, 1280))
    with pytest.raises(ValueError):
        GroundCalibration.load(tmp_path / "missing.yaml")


def test_swapped_left_right_clicks_are_rejected(cam):
    base = reference_points(ContainerConfig())
    img = cam.project(np.c_[base, np.zeros(4)])
    with pytest.raises(ValueError, match="below the sheet"):
        GroundCalibration.from_points(img[[1, 0, 3, 2]], base, SIZE)


def test_known_fov_makes_the_pose_robust_to_noise(cam):
    base = reference_points(ContainerConfig())
    for seed in range(10):
        img = cam.project(np.c_[base, np.zeros(4)]) + np.random.default_rng(seed).normal(0, 1.5, (4, 2))
        est = GroundCalibration.from_points(img, base, SIZE, fov_deg=math.degrees(2 * math.atan(SIZE[0] / 2 / F)))
        assert est.fov_source == "config"
        assert np.linalg.norm(est.camera_center - cam.camera_center) < 0.03 * 520


def test_circle_fit():
    a = np.linspace(0, 2.0, 40)
    c, r, rms = circle_fit(np.c_[50 + 30 * np.cos(a), -20 + 30 * np.sin(a)])
    np.testing.assert_allclose(c, [50, -20], atol=1e-6)
    assert r == pytest.approx(30) and rms < 1e-6


# ------------------------------------------------------------------------------------------ fitting
SHAPES = {   # name -> (solid points, true centre, true size, shape, height)
    "cylinder": (frustum((240, 40), 55, 55, 80), (240, 40), (110, 110), "round", 80),
    "cup": (frustum((250, -50), 35, 48, 95), (250, -50), (96, 96), "round", 95),
    "box30": (box((235, 30), 130, 80, 70, 30), (235, 30), (130, 80), "box", 70),
    "box0": (box((260, -70), 110, 90, 90, 0), (260, -70), (110, 90), "box", 90),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_configured_height_gives_accurate_centre_and_size(cam, name):
    pts, centre, size, shape, h = SHAPES[name]
    state = run(make_finder(cam, {"red": h}), scene(cam, [(pts, RED)]))
    (c,) = state.containers
    assert c.shape == shape and c.height_source == "config"
    assert np.linalg.norm(c.center_mm - centre) < 4
    assert sorted(c.size_mm) == pytest.approx(sorted(size), abs=5)
    assert c.top_mm[2] == pytest.approx(h)


def test_strongly_leaning_box_is_not_mistaken_for_round():
    near = true_camera(center=(520.0, 0.0, 330.0), target=(100.0, 0.0, 30.0))   # box far from the camera: long lean
    pts = box((50, -90), 70, 60, 70, 20)
    f = make_finder(near, {"blue": 70})
    f.cfg.zone = ZoneConfig(x_mm=(0.0, 100.0), y_mm=(-165.0, 165.0), max_angle_deg=180.0)
    f = ContainerFinder(f.cfg, near)
    (c,) = run(f, scene(near, [(pts, BLUE)])).containers
    assert c.shape == "box" and np.linalg.norm(c.center_mm - (50, -90)) < 5


def test_box_yaw(cam):
    pts, _, _, _, h = SHAPES["box30"]
    (c,) = run(make_finder(cam, {"red": h}), scene(cam, [(pts, RED)])).containers
    assert abs(((c.yaw_deg - 30) + 90) % 180 - 90) < 3


def test_wide_bowl_with_configured_height(cam):
    pts = bowl((245, 20), 85, 60)
    (c,) = run(make_finder(cam, {"red": 60}), scene(cam, [(pts, RED)])).containers
    assert c.shape == "round" and np.linalg.norm(c.center_mm - (245, 20)) < 4
    assert c.size_mm[0] == pytest.approx(170, abs=5)


@pytest.mark.parametrize("name", ["cylinder", "cup", "box30", "box0"])
def test_auto_height(cam, name):
    pts, centre, _, _, h = SHAPES[name]
    (c,) = run(make_finder(cam), scene(cam, [(pts, RED)])).containers
    assert c.height_source == "auto", name
    assert abs(c.height_mm - h) < 8
    assert np.linalg.norm(c.center_mm - centre) < 6


def test_hidden_base_bowl_falls_back_to_default(cam):
    # a very wide bowl hides its base behind the rim: the automatic height is rejected (set it in the config)
    (c,) = run(make_finder(cam), scene(cam, [(bowl((250, 10), 110, 50), RED)])).containers
    assert c.height_source == "default" and c.height_mm == 80


# ------------------------------------------------------------------------------------------ finding
def test_find_picks_the_right_colour(cam):
    img = scene(cam, [(frustum((240, 60), 50, 50, 80), RED), (frustum((250, -70), 50, 50, 80), BLUE)])
    state = run(make_finder(cam, {"red": 80, "blue": 80}), img)
    assert {c.color for c in state.containers} == {"red", "blue"}
    assert state.find("red").color == "red" and state.find("Blue").center_mm[1] < 0
    assert state.find("green") is None
    assert state.to_dict("red")["target"]["color"] == "red"
    json.dumps(state.to_dict("red"))


def test_out_of_zone_is_flagged_and_not_found(cam):
    img = scene(cam, [(frustum((240, 200), 40, 40, 80), RED)])
    state = run(make_finder(cam, {"red": 80}), img)
    assert len(state.containers) == 1 and not state.containers[0].in_zone
    assert state.find("red") is None


def test_empty_sheet_and_grey_shadow_give_nothing(cam):
    assert run(make_finder(cam), scene(cam, [])).containers == []
    assert run(make_finder(cam), scene(cam, [(frustum((240, 0), 60, 60, 60), GREY)])).containers == []


def test_min_age_gates_reporting_and_ids_are_stable(cam):
    img = scene(cam, [(frustum((240, 0), 50, 50, 80), RED)])
    f = make_finder(cam, {"red": 80})
    f.cfg.tracking.min_age = 3
    assert f.process(img).containers == [] and f.process(img).containers == []
    first = f.process(img).containers
    assert [c.age for c in first] == [3]
    assert f.process(img).containers[0].id == first[0].id
    assert f.process(scene(cam, [])).find("red") is None      # not seen this frame


def test_frame_of_other_size_same_aspect_is_rescaled(cam):
    img = cv2.resize(scene(cam, [(frustum((240, 0), 50, 50, 80), RED)]), (1920, 1080))
    (c,) = run(make_finder(cam, {"red": 80}), img).containers
    assert np.linalg.norm(c.center_mm - (240, 0)) < 5
    with pytest.raises(FrameSizeError):
        make_finder(cam).process(np.zeros((1280, 720, 3), np.uint8))


# ------------------------------------------------------------------------------------------ config
def test_default_config_file_loads():
    cfg = load_container_config()
    assert cfg.sheet.width_mm == 430 and cfg.zone.x_mm == (200.0, 300.0) and cfg.arm.base_from_edge_mm == 100 \
        and cfg.arm.base_offset_mm == 0


@pytest.mark.parametrize("bad", [
    {"sheet": {"width_mm": -1}}, {"sheet": {"mark_distance_mm": 500}}, {"zone": {"x_mm": [5, 1]}}, {"containers": {"heights_mm": {"red": 0}}},
    {"containers": {"default_height_mm": 500}}, {"tracking": {"ema": 0}}, {"camera": {"fov_deg": 200}},
    {"sheet": {"points": [[0, 0]]}}, {"bogus": {}}, {"zone": {"nope": 1}}, {"tracking": {"min_age": -1}}])
def test_config_validation_errors(bad, tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError):
        load_container_config(p)


def test_heights_colour_names_are_normalised(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("containers: {heights_mm: {Grey: 50}}")
    assert load_container_config(p).containers.heights_mm == {"gray": 50.0}


# ------------------------------------------------------------------------------------------ CLI
def test_still_image_through_the_cli_prints_json(cam, tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import find_containers
    cam.save(tmp_path / "cal.yaml")
    cv2.imwrite(str(tmp_path / "s.png"), scene(cam, [(frustum((240, 40), 55, 55, 80), RED)]))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("containers: {heights_mm: {red: 80}}")
    assert find_containers.main(["--source", str(tmp_path / "s.png"), "--calibration", str(tmp_path / "cal.yaml"),
                                 "--config", str(cfg), "--no-gui", "--json", "--target", "red",
                                 "--save", str(tmp_path / "o.png")]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["calibrated"] and line["target"]["color"] == "red"
    assert np.linalg.norm(np.array(line["target"]["center_mm"]) - (240, 40)) < 4
    assert line["target"]["height_source"] == "config" and (tmp_path / "o.png").exists()


def test_cli_without_calibration_reports_it(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import find_containers
    cv2.imwrite(str(tmp_path / "s.png"), np.zeros((72, 128, 3), np.uint8))
    with pytest.raises(SystemExit) as e:
        find_containers.main(["--source", str(tmp_path / "s.png"), "--calibration", str(tmp_path / "none.yaml"),
                              "--no-gui", "--json"])
    assert e.value.code == 1 and json.loads(capsys.readouterr().out)["calibrated"] is False


def test_calibrate_ui_shows_live_video_then_freezes_and_saves(cam, tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import find_containers
    base = reference_points(ContainerConfig())
    clicks = [tuple(p) for p in cam.project(np.c_[base, np.zeros(4)])]
    frames = [0]
    img = scene(cam, [])

    def grab():
        frames[0] += 1
        return img.copy()

    handler, calls = {}, [0]

    def wait_key(ms):
        calls[0] += 1
        if calls[0] == 1:
            return ord(" ")                                  # freeze after one live frame
        if calls[0] == 2:
            for x, y in clicks:
                handler["cb"](cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)
            return 255
        return ord("s")

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda w, cb: handler.update(cb=cb))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    out = tmp_path / "cal.yaml"
    result = find_containers.calibrate(grab, ContainerConfig(), out, None)
    assert result is not None and out.exists()
    assert frames[0] >= 2                                    # frames kept coming until the freeze
    assert result.focal == pytest.approx(F, rel=0.02)


# ------------------------------------------------------------------------------------------ false positives
ORANGE = (0, 130, 255)


def test_orange_container_is_named_orange_and_found(cam):
    state = run(make_finder(cam, {"orange": 80}), scene(cam, [(frustum((240, 40), 50, 50, 80), ORANGE)]))
    assert [c.color for c in state.containers] == ["orange"] and state.find("orange") is not None


def test_a_container_missed_for_a_frame_is_not_reported_as_a_ghost(cam):
    f = make_finder(cam, {"red": 80})
    run(f, scene(cam, [(frustum((240, 40), 50, 50, 80), RED)]))
    assert f.process(scene(cam, [])).containers == []


def test_jumping_height_estimates_do_not_spawn_new_containers(cam):
    f = make_finder(cam)
    ids = set()
    for i in range(30):                      # same container, slightly different silhouette every frame
        r = 50 + (i % 3)
        ids |= {c.id for c in f.process(scene(cam, [(frustum((240, 40), r, r, 80 + (i % 4) * 2), RED)])).containers}
    assert len(ids) == 1


def test_colour_outside_the_sheet_is_ignored(cam):
    img = scene(cam, [(box((240, -200), 120, 50, 10, 0), ORANGE)])        # a low orange strip on the table
    cfg = make_finder(cam).cfg
    cfg.sheet.width_mm = 200                                              # sheet edge at y = -100, margin 60
    assert run(ContainerFinder(cfg, cam), img).containers == []
    cfg.containers.sheet_margin_mm = None
    assert len(run(ContainerFinder(cfg, cam), img).containers) == 1


def test_colour_filter_and_blob_debug_info(cam):
    img = scene(cam, [(frustum((240, 60), 50, 50, 80), RED), (frustum((250, -70), 50, 50, 80), BLUE)])
    f = make_finder(cam, {"red": 80, "blue": 80})
    f.cfg.containers.colors = ["blue"]
    s = run(ContainerFinder(f.cfg, cam), img)
    assert [c.color for c in s.containers] == ["blue"]
    g = make_finder(cam, {"red": 80, "blue": 80})
    g.detect(img)
    assert {b["color"] for b in g.last_blobs} == {"red", "blue"} and all(b["status"] == "kept" for b in g.last_blobs)


def test_unknown_colour_in_config_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("containers: {colors: [mauve]}")
    with pytest.raises(ConfigError):
        load_container_config(p)


def test_pale_shadow_tint_next_to_a_container_is_not_part_of_it(cam):
    pale = (150, 175, 215)           # warm, desaturated: shadow / glow through translucent plastic
    img = scene(cam, [(box((335, 20), 90, 60, 1, 0), pale), (frustum((345, -60), 40, 40, 70), ORANGE)])
    state = run(make_finder(cam, {"orange": 70}), img)
    assert [c.color for c in state.containers] == ["orange"]
    assert np.linalg.norm(state.containers[0].center_mm - (345, -60)) < 5


def test_hue_range_override_keeps_a_shaded_orange_side_in_one_container(cam):
    shaded = (20, 60, 220)           # hue ~ 8 deg: "red" by default
    solids = [(frustum((340, 0), 45, 45, 70), (0, 140, 255)), (box((340, 0), 60, 40, 30, 0), shaded)]
    cfg = make_finder(cam, {"orange": 70}).cfg
    cfg.containers.hue_ranges = {"orange": [(0.0, 40.0), (350.0, 360.0)]}
    cfg.containers.colors = ["orange"]
    (c,) = run(ContainerFinder(validate_container_config(cfg), cam), scene(cam, solids)).containers
    assert c.color == "orange"


@pytest.mark.parametrize("bad", [{"containers": {"min_saturation": 2}}, {"containers": {"hue_ranges": {"mauve": [[0, 10]]}}},
                                 {"containers": {"hue_ranges": {"orange": [[50, 10]]}}}, {"containers": {"hue_ranges": []}}])
def test_new_container_options_are_validated(bad, tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError):
        load_container_config(p)
