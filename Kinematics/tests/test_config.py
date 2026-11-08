from copy import deepcopy
import math

import numpy as np
import pytest

from Kinematics.config import (
    ConfigError, DEFAULT_PATH, JOINT_NAMES, get_T_base_cam, get_joint_limits,
    load_config, save_config, validate_config,
)


@pytest.fixture
def cfg():
    return load_config()


def at(config, path):
    for part in path:
        config = config[part]
    return config


def label(path):
    result = ""
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else ("." if result else "") + part
    return result


# Every required field, including containers, must produce its full key path.
REQUIRED_PATHS = [
    ("schema_version",), ("units",), ("frame",), ("links",),
    ("links", "lengths"), ("links", "offsets"), ("joints",),
    ("servos",), ("gripper",), ("workspace",), ("camera",), ("safety",),
    *[("units", key) for key in ("length", "angle", "time", "pulse")],
    *[("frame", key) for key in ("handedness", "origin", "x", "y", "z")],
    *[("links", "lengths", key) for key in ("upper_arm", "forearm", "tool")],
    *[("links", "offsets", key) for key in ("shoulder_xyz", "wrist_xyz", "tool_xyz")],
    *[("joints", i, key) for i in range(5)
      for key in ("name", "sign", "zero_offset", "min", "max", "max_velocity")],
    *[("servos", name) for name in JOINT_NAMES],
    *[("servos", name, key) for name in JOINT_NAMES
      for key in ("pulse_min_us", "pulse_max_us", "angle_at_min", "angle_at_max", "calibration")],
    *[("servos", name, "calibration", "coefficients") for name in JOINT_NAMES],
    *[("gripper", key) for key in ("open_pulse_us", "closed_pulse_us")],
    *[("workspace", axis) for axis in "xyz"],
    *[("workspace", axis, key) for axis in "xyz" for key in ("min", "max")],
    *[("camera", key) for key in ("intrinsics", "T_base_cam", "table_homography", "resolution")],
    ("safety", "max_speed"),
]


@pytest.mark.parametrize("path", REQUIRED_PATHS, ids=label)
def test_missing_required_key_names_path(cfg, path):
    del at(cfg, path[:-1])[path[-1]]
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert label(path) in str(error.value)


def test_example_and_helpers(cfg):
    assert validate_config(cfg) is cfg
    assert tuple(j["name"] for j in cfg["joints"]) == JOINT_NAMES
    limits = get_joint_limits(cfg)
    assert limits.shape == (5, 2)
    assert limits.dtype == float
    np.testing.assert_allclose(limits[0], [-1, 1])
    transform = get_T_base_cam(cfg)
    np.testing.assert_array_equal(transform, np.eye(4))
    limits[0, 0] = 999
    transform[0, 0] = 999
    assert cfg["joints"][0]["min"] == -1
    assert cfg["camera"]["T_base_cam"][0][0] == 1
    np.testing.assert_array_equal(get_joint_limits(), get_joint_limits(cfg))
    np.testing.assert_array_equal(get_T_base_cam(), get_T_base_cam(cfg))


@pytest.mark.parametrize("path,value", [
    (("units", "length"), "cm"), (("units", "angle"), "deg"),
    (("units", "time"), "ms"), (("units", "pulse"), "s"),
    (("frame", "handedness"), "left"), (("frame", "x"), "backward"),
    (("frame", "y"), "right"), (("frame", "z"), "down"),
    (("frame", "origin"), "shoulder"), (("schema_version",), True),
    (("joints", 0, "name"), "shoulder"), (("joints", 0, "sign"), 0),
    (("joints", 0, "max_velocity"), 0), (("safety", "max_speed"), -1),
    (("links", "lengths", "upper_arm"), 0),
    (("servos", "base", "pulse_min_us"), -1),
    (("servos", "base", "calibration", "coefficients"), [1000, 0]),
    (("servos", "base", "calibration", "coefficients"), [1000, 300]),
    (("servos", "base", "calibration", "coefficients"), [1000, 318, 0]),
    (("camera", "resolution", 0), 0), (("camera", "resolution", 1), 1080.0),
    (("camera", "resolution", 1), True),
    (("gripper", "closed_pulse_us"), 2500),
    # Physically inside the servo pulse range, but outside gripper joint limits.
    (("gripper", "closed_pulse_us"), 1800),
])
def test_invalid_values_name_key(cfg, path, value):
    at(cfg, path[:-1])[path[-1]] = value
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert label(path) in str(error.value)


@pytest.mark.parametrize("path", [
    ("joints", 0, "min"), ("joints", 2, "zero_offset"),
    ("servos", "wrist", "pulse_min_us"),
    ("servos", "elbow", "calibration", "coefficients", 1),
    ("links", "offsets", "shoulder_xyz", 2),
    ("links", "lengths", "tool"), ("workspace", "z", "max"),
    ("camera", "T_base_cam", 0, 3), ("camera", "table_homography", 0, 0),
    ("safety", "max_speed"),
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.3"])
def test_bad_numbers_are_rejected_with_path(cfg, path, value):
    at(cfg, path[:-1])[path[-1]] = value
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert label(path) in str(error.value)


@pytest.mark.parametrize("path,low,high", [
    (("joints", 0), "min", "max"),
    (("workspace", "x"), "min", "max"),
    (("servos", "base"), "pulse_min_us", "pulse_max_us"),
])
@pytest.mark.parametrize("delta", [0, 1])
def test_empty_and_reversed_limits(cfg, path, low, high, delta):
    node = at(cfg, path)
    node[low] = node[high] + delta
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert label(path + (low,)) in str(error.value)


@pytest.mark.parametrize("sign,key,value", [
    (1, "min", -2), (1, "max", 2), (-1, "min", -2), (-1, "max", 2),
])
def test_joint_limits_checked_after_sign_and_offset(cfg, sign, key, value):
    cfg["joints"][0]["sign"] = sign
    cfg["joints"][0][key] = value
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert f"joints[0].{key}" in str(error.value)
    assert "servos.base" in str(error.value)


def test_negative_servo_slope_is_supported(cfg):
    servo = cfg["servos"]["base"]
    servo["angle_at_min"], servo["angle_at_max"] = math.pi, 0.0
    servo["calibration"]["coefficients"] = [2000, -1000 / math.pi]
    validate_config(cfg)


def test_different_per_joint_and_global_speed_caps_are_allowed(cfg):
    cfg["joints"][0]["max_velocity"] = 0.5
    cfg["safety"]["max_speed"] = 0.1
    validate_config(cfg)  # Consumers use min(joint cap, global cap).


@pytest.mark.parametrize("path,value", [
    (("units",), []), (("joints",), []), (("servos",), None),
    (("links", "offsets", "shoulder_xyz"), [0, 0]),
    (("camera", "T_base_cam"), [[1, 0], [0, 1]]),
    (("camera", "T_base_cam", 3), [0, 0, 0, 0]),
    (("camera", "T_base_cam", 0), [2, 0, 0, 0]),
    (("camera", "T_base_cam", 0), [-1, 0, 0, 0]),  # reflection is not right-handed
    (("camera", "table_homography"), [[0, 0, 0]] * 3),
    (("camera", "table_homography"), [[1, 0, 0], [1, 0, 0], [0, 0, 1]]),
    (("camera", "intrinsics"), [[0, 0, 960], [0, 1000, 540], [0, 0, 1]]),
    (("camera", "intrinsics"), [[1000, 0, 960], [0, 1000, 540], [1, 0, 1]]),
    (("camera", "intrinsics"), [[1000, 0, 960], [3, 1000, 540], [0, 0, 1]]),
])
def test_invalid_shapes_and_transforms(cfg, path, value):
    at(cfg, path[:-1])[path[-1]] = value
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    # Rotation validity is a property of the whole matrix, not one bad row.
    expected = ("camera", "T_base_cam") if path[:2] == ("camera", "T_base_cam") else path
    assert label(expected) in str(error.value)


def test_valid_nonidentity_camera(cfg):
    cfg["camera"]["intrinsics"] = [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
    cfg["camera"]["T_base_cam"] = [[0, -1, 0, 0.1], [1, 0, 0, -0.2],
                                       [0, 0, 1, 0.5], [0, 0, 0, 1]]
    # Homographies are defined only up to an arbitrary nonzero scale.
    cfg["camera"]["table_homography"] = [[1e-20, 0, 0], [0, 1e-20, 0], [0, 0, 1e-20]]
    validate_config(cfg)
    p_base = get_T_base_cam(cfg) @ [1, 0, 0, 1]
    np.testing.assert_allclose(p_base, [0.1, 0.8, 0.5, 1])


def test_all_finite_numbers_in_extra_fields(cfg):
    cfg["future_calibration"] = {"samples": [float("nan")]}
    with pytest.raises(ConfigError, match=r"future_calibration.samples\[0\]"):
        validate_config(cfg)


@pytest.mark.parametrize("bad_root", [None, [], "hello", 42])
def test_bad_root(bad_root):
    with pytest.raises(ConfigError, match="config"):
        validate_config(bad_root)


def test_comments_and_values_survive_calibration_save(cfg, tmp_path):
    path = tmp_path / "arm_config.yaml"
    cfg["links"]["lengths"]["upper_arm"] = 0.140
    save_config(cfg, path)
    text = path.read_text()
    assert "# EXAMPLE ONLY" in text
    assert "# CAD NOMINAL: shoulder pivot to elbow pivot, meters" in text
    assert load_config(path) == cfg
    assert load_config(path)["links"]["lengths"]["upper_arm"] == 0.140


def test_invalid_save_keeps_original_file(cfg, tmp_path):
    path = tmp_path / "arm_config.yaml"
    path.write_text("keep the original bytes\n")
    cfg["joints"][0]["min"] = 2
    with pytest.raises(ConfigError):
        save_config(cfg, path)
    assert path.read_text() == "keep the original bytes\n"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_keeps_original_and_cleans_temp(cfg, tmp_path, monkeypatch):
    path = tmp_path / "arm_config.yaml"
    path.write_text("original\n")

    def fail_replace(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr("config.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_config(cfg, path)
    assert path.read_text() == "original\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("text", ["camera: [", "units: {angle: rad, angle: deg}", ""])
def test_bad_yaml_and_duplicate_keys(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_file_error_names_path(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert str(path) in str(error.value)


def test_working_directory_does_not_change_default_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_PATH.is_absolute()
    assert load_config()["units"]["angle"] == "rad"


def test_validation_does_not_modify_config(cfg):
    before = deepcopy(cfg)
    validate_config(cfg)
    assert cfg == before
