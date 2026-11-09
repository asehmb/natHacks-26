import numpy as np
import pytest

from armvision.config import load_config
from armvision.intrinsics import Intrinsics, calibrate, fallback_intrinsics, view_points
from armvision.pipeline import FrameSizeError, VisionSystem


def test_fallback_intrinsics():
    intr = fallback_intrinsics(640, 480, 60.0)
    assert not intr.calibrated
    assert intr.image_size == (640, 480)
    assert intr.K.shape == (3, 3)
    assert intr.dist.shape == (5,)


def test_view_points_min_markers():
    cfg = load_config()
    # markers has 0, 1, 2, 3
    # empty markers
    assert view_points({}, cfg) is None
    # only one marker when min_markers=2
    one_marker = {0: np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)}
    assert view_points(one_marker, cfg) is None

    # two markers
    two_markers = {
        0: np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32),
        1: np.array([[100, 10], [110, 10], [110, 20], [100, 20]], dtype=np.float32),
    }
    vp = view_points(two_markers, cfg)
    assert vp is not None
    obj, img = vp
    assert len(obj) == 8
    assert len(img) == 8


def test_calibrate_validation():
    with pytest.raises(ValueError, match="need at least 5 views"):
        calibrate([], (640, 480))

    # Invalid image size
    dummy_view = (np.zeros((8, 3), np.float32), np.zeros((8, 2), np.float32))
    views = [dummy_view] * 5
    with pytest.raises(ValueError, match="invalid image size"):
        calibrate(views, (0, 480))


def test_intrinsics_setter_resets_reference_and_pose(cfg):
    vs = VisionSystem(cfg)
    vs.reference = np.zeros((100, 100, 3), dtype=np.uint8)
    vs.pose = "mock_pose"

    new_intr = Intrinsics(np.eye(3), np.zeros(5), (640, 480))
    vs.intrinsics = new_intr

    assert vs.intrinsics is new_intr
    assert vs.reference is None
    assert vs.pose is None


def test_same_aspect_resolution_is_rescaled(cfg):
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    vs = VisionSystem(cfg, intrinsics=Intrinsics(K, np.zeros(5), (640, 480)))
    vs.process(np.zeros((960, 1280, 3), dtype=np.uint8))
    assert vs.intrinsics.image_size == (1280, 960)
    assert vs.intrinsics.K[0, 0] == pytest.approx(1000.0)


def test_rotated_frame_rejected(cfg):
    # a phone flipping to portrait changes the aspect ratio: can't be fixed by scaling
    vs = VisionSystem(cfg, intrinsics=Intrinsics(np.eye(3), np.zeros(5), (640, 480)))
    with pytest.raises(FrameSizeError, match="aspect ratio"):
        vs.process(np.zeros((640, 480, 3), dtype=np.uint8))


def test_intrinsics_scaling_and_explicit_adjustment(cfg):
    # Base intrinsics at 640x480
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
    intr = Intrinsics(K, dist, (640, 480), calibrated=True, rms=0.42)

    # Scale to 1280x960
    scaled = intr.scaled((1280, 960))
    assert scaled.image_size == (1280, 960)
    assert scaled.K[0, 0] == pytest.approx(1000.0)
    assert scaled.K[1, 1] == pytest.approx(1000.0)
    assert scaled.K[0, 2] == pytest.approx(640.0)
    assert scaled.K[1, 2] == pytest.approx(480.0)
    assert np.array_equal(scaled.dist, dist)

    # Test VisionSystem adjust_intrinsics_for_frame
    vs = VisionSystem(cfg, intrinsics=intr)
    frame_1280 = np.zeros((960, 1280, 3), dtype=np.uint8)
    vs.adjust_intrinsics_for_frame(frame_1280)
    assert vs.intrinsics.image_size == (1280, 960)
    # Processing now proceeds without ValueError
    state = vs.process(frame_1280)
    assert isinstance(state.mat_visible, bool)


def test_vision_standalone_does_not_import_kinematics_stack():
    import subprocess
    import sys
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import armvision; "
            "from armvision import validate_target; "
            "leaked = [m for m in sys.modules if 'matplotlib' in m or 'scipy' in m or 'Kinematics' in m]; "
            "assert not leaked, f'Leaked: {leaked}'"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "vision"})
    assert res.returncode == 0, f"Import check failed: {res.stderr}"
