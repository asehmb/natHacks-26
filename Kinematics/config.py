"""Load and validate arm_config.yaml; retain comments for calibration writeback.

No hardware I/O and no import-time file reads. All model angles are radians.
See README.md for the joint/servo mapping and helper return types.
"""

from collections.abc import Mapping
import math
from pathlib import Path
import tempfile
import os

import numpy as np
try:
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError
except ImportError:
    import yaml
    YAMLError = yaml.YAMLError

    class YAML:
        def __init__(self, typ=None):
            self.preserve_quotes = True
            self.allow_duplicate_keys = False

        def indent(self, mapping=2, sequence=4, offset=2):
            pass

        def load(self, stream):
            return yaml.safe_load(stream)

        def dump(self, data, stream):
            yaml.dump(data, stream, sort_keys=False)

DEFAULT_PATH = Path(__file__).with_name("arm_config.yaml")
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist", "gripper")


class ConfigError(ValueError):
    """A missing or invalid configuration value, with its key path."""


def _fail(path, message):
    raise ConfigError(f"{path}: {message}")


def _mapping(value, path):
    if not isinstance(value, Mapping):
        _fail(path, "expected a mapping")
    return value


def _required(mapping, key, path):
    if key not in mapping:
        _fail(f"{path}.{key}" if path else key, "required key is missing")
    return mapping[key]


def _number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a finite number (not a string or boolean)")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, "number must be finite")
    return float(value)


def _num(mapping, key, path):
    return _number(_required(mapping, key, path), f"{path}.{key}")


def _finite_tree(value, path="config"):
    # Also catch nonfinite numbers in future/extra calibration fields.
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _number(value, path)


def _sequence(value, length, path):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        _fail(path, f"expected a sequence of length {length}")
    return value


def _matrix(value, size, path):
    rows = _sequence(value, size, path)
    return np.array([
        [_number(v, f"{path}[{i}][{j}]")
         for j, v in enumerate(_sequence(row, size, f"{path}[{i}]"))]
        for i, row in enumerate(rows)
    ], dtype=float)


def _bounds(mapping, low_key, high_key, path):
    low = _num(mapping, low_key, path)
    high = _num(mapping, high_key, path)
    if low >= high:
        _fail(f"{path}.{low_key}", f"must be < {path}.{high_key}")
    return low, high


def validate_config(config):
    """Validate in place without modifying values; return the original mapping.

    Extra fields are allowed for team extensions. Exactly five ordered actuator
    slots are required. Validation proves schema consistency, not hardware safety.
    """
    _mapping(config, "config")
    _finite_tree(config)
    version = _required(config, "schema_version", "")
    if type(version) is not int or version != 1:
        _fail("schema_version", "expected integer 1")

    units = _mapping(_required(config, "units", ""), "units")
    for key, expected in {"length": "m", "angle": "rad", "time": "s", "pulse": "us"}.items():
        if _required(units, key, "units") != expected:
            _fail(f"units.{key}", f"must be {expected!r}")
    frame = _mapping(_required(config, "frame", ""), "frame")
    for key, expected in {
        "handedness": "right", "origin": "base_rotation_axis_at_table_height",
        "x": "forward", "y": "left", "z": "up",
    }.items():
        if _required(frame, key, "frame") != expected:
            _fail(f"frame.{key}", f"must be {expected!r}")

    links = _mapping(_required(config, "links", ""), "links")
    lengths = _mapping(_required(links, "lengths", "links"), "links.lengths")
    for name in ("upper_arm", "forearm", "tool"):
        if _num(lengths, name, "links.lengths") <= 0:
            _fail(f"links.lengths.{name}", "must be > 0")
    offsets = _mapping(_required(links, "offsets", "links"), "links.offsets")
    for name in ("shoulder_xyz", "wrist_xyz", "tool_xyz"):
        path = f"links.offsets.{name}"
        for i, value in enumerate(_sequence(_required(offsets, name, "links.offsets"), 3, path)):
            _number(value, f"{path}[{i}]")

    joints = _sequence(_required(config, "joints", ""), 5, "joints")
    servos = _mapping(_required(config, "servos", ""), "servos")
    gripper_pulse_bounds = None
    for i, name in enumerate(JOINT_NAMES):
        jp = f"joints[{i}]"
        joint = _mapping(joints[i], jp)
        if _required(joint, "name", jp) != name:
            _fail(f"{jp}.name", f"expected {name!r}; joint order is {JOINT_NAMES}")
        sign = _num(joint, "sign", jp)
        if sign not in (-1, 1):
            _fail(f"{jp}.sign", "must be +1 or -1")
        zero = _num(joint, "zero_offset", jp)
        low, high = _bounds(joint, "min", "max", jp)
        if _num(joint, "max_velocity", jp) <= 0:
            _fail(f"{jp}.max_velocity", "must be > 0 rad/s")

        sp = f"servos.{name}"
        servo = _mapping(_required(servos, name, "servos"), sp)
        pmin, pmax = _bounds(servo, "pulse_min_us", "pulse_max_us", sp)
        if pmin <= 0:
            _fail(f"{sp}.pulse_min_us", "must be > 0")
        amin = _num(servo, "angle_at_min", sp)
        amax = _num(servo, "angle_at_max", sp)
        if amin == amax:
            _fail(f"{sp}.angle_at_max", "must differ from angle_at_min")
        cp = f"{sp}.calibration"
        calibration = _mapping(_required(servo, "calibration", sp), cp)
        coeff_path = f"{cp}.coefficients"
        coeffs = _sequence(_required(calibration, "coefficients", cp), 2, coeff_path)
        c0, c1 = [_number(value, f"{coeff_path}[{j}]") for j, value in enumerate(coeffs)]
        if c1 == 0:
            _fail(f"{coeff_path}[1]", "slope must be nonzero")
        for angle, pulse, key in ((amin, pmin, "angle_at_min"), (amax, pmax, "angle_at_max")):
            fitted = _number(c0 + c1 * angle, coeff_path)
            if not math.isclose(fitted, pulse, rel_tol=0, abs_tol=1e-6):
                _fail(coeff_path, f"does not map {sp}.{key} to its pulse endpoint")

        # Affine maps attain their extrema at endpoints, including negative signs.
        pulses = []
        for key, q in (("min", low), ("max", high)):
            angle = _number(sign * q + zero, f"{jp}.{key}")
            if not min(amin, amax) <= angle <= max(amin, amax):
                _fail(f"{jp}.{key}", f"mapped angle {angle:g} is outside {sp} angular range")
            pulse = _number(c0 + c1 * angle, f"{jp}.{key}")
            if not pmin - 1e-6 <= pulse <= pmax + 1e-6:
                _fail(f"{jp}.{key}", f"mapped pulse is outside {sp} pulse range")
            pulses.append(pulse)
        if name == "gripper":
            gripper_pulse_bounds = (min(pulses), max(pulses))

    gripper = _mapping(_required(config, "gripper", ""), "gripper")
    for key in ("open_pulse_us", "closed_pulse_us"):
        pulse = _num(gripper, key, "gripper")
        if not gripper_pulse_bounds[0] - 1e-6 <= pulse <= gripper_pulse_bounds[1] + 1e-6:
            _fail(f"gripper.{key}", "outside the gripper joint's mapped pulse range")
    if gripper["open_pulse_us"] == gripper["closed_pulse_us"]:
        _fail("gripper.closed_pulse_us", "must differ from open_pulse_us")

    workspace = _mapping(_required(config, "workspace", ""), "workspace")
    for axis in "xyz":
        path = f"workspace.{axis}"
        _bounds(_mapping(_required(workspace, axis, "workspace"), path), "min", "max", path)

    camera = _mapping(_required(config, "camera", ""), "camera")
    intrinsics = _required(camera, "intrinsics", "camera")
    if intrinsics is not None:
        k = _matrix(intrinsics, 3, "camera.intrinsics")
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            _fail("camera.intrinsics", "fx and fy must be > 0")
        if not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-8) or abs(k[1, 0]) > 1e-8:
            _fail("camera.intrinsics", "expected pinhole K = [[fx, skew, cx], [0, fy, cy], [0, 0, 1]]")
    transform = _matrix(_required(camera, "T_base_cam", "camera"), 4, "camera.T_base_cam")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        _fail("camera.T_base_cam[3]", "expected [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        proper_rotation = np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
    if not proper_rotation or not math.isclose(np.linalg.det(rotation), 1, rel_tol=0, abs_tol=1e-6):
        _fail("camera.T_base_cam", "rotation must be orthonormal with determinant +1")
    homography = _matrix(_required(camera, "table_homography", "camera"), 3, "camera.table_homography")
    scale = np.max(np.abs(homography))
    if scale == 0 or np.linalg.matrix_rank(homography / scale) < 3:
        _fail("camera.table_homography", "must be invertible")
    resolution = _sequence(_required(camera, "resolution", "camera"), 2, "camera.resolution")
    for i, value in enumerate(resolution):
        _number(value, f"camera.resolution[{i}]")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _fail(f"camera.resolution[{i}]", "expected a positive integer pixel count")

    safety = _mapping(_required(config, "safety", ""), "safety")
    if _num(safety, "max_speed", "safety") <= 0:
        _fail("safety.max_speed", "must be > 0 rad/s")
    return config


def _yaml():
    yaml = YAML(typ="rt")  # round-trip parser retains comments and key ordering
    yaml.preserve_quotes = True
    yaml.allow_duplicate_keys = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def load_config(path=DEFAULT_PATH):
    """Return a validated, editable ruamel CommentedMap. No global cache."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            config = _yaml().load(stream)
    except YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read configuration: {exc}") from exc
    return validate_config(config)


def get_joint_limits(config=None):
    """Return a new float ndarray, shape (5, 2), columns [min, max] radians."""
    config = load_config() if config is None else validate_config(config)
    return np.array([[joint["min"], joint["max"]] for joint in config["joints"]], dtype=float)


def get_T_base_cam(config=None):
    """Return a new float (4, 4) camera-to-base homogeneous transform."""
    config = load_config() if config is None else validate_config(config)
    return np.array(config["camera"]["T_base_cam"], dtype=float, copy=True)


def save_config(config, path=DEFAULT_PATH):
    """Validate then atomically replace path, retaining loaded YAML comments.

    Pass the same path used to load if it wasn't DEFAULT_PATH. Do not convert the
    returned CommentedMap to a plain dict if you want to retain its comments.
    Atomic replacement prevents partial files, but does not merge concurrent edits.
    """
    validate_config(config)
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            _yaml().dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
