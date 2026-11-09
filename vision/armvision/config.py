"""Load and validate mat_config.yaml into dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "mat_config.yaml"


class ConfigError(ValueError):
    pass


@dataclass
class MatConfig:
    width_mm: float = 400.0
    height_mm: float = 300.0
    marker_size_mm: float = 38.1
    markers: dict[int, tuple[float, float]] = field(default_factory=lambda: {
        0: (30.0, 30.0), 1: (370.0, 30.0), 2: (370.0, 270.0), 3: (30.0, 270.0)})
    min_markers: int = 2


@dataclass
class ArmConfig:
    base_xy_mm: tuple[float, float] = (-60.0, 150.0)
    base_yaw_deg: float = 0.0
    mat_z_mm: float = 0.0
    base_zone_radius_mm: float = 100.0
    gripper_marker_id: int = 10
    gripper_marker_size_mm: float = 30.48


@dataclass
class VisionConfig:
    px_per_mm: float = 2.0
    diff_threshold: float = 18.0
    min_area_mm2: float = 150.0
    edge_margin_mm: float = 4.0
    grasp_inset_mm: float = 15.0
    grasp_z_mm: float = 15.0
    round_aspect: float = 1.3
    use_near_half: bool = False
    gate_mm: float = 40.0
    max_missed: int = 15
    ema: float = 0.5
    fov_deg: float = 60.0
    max_pose_age: int = 5


@dataclass
class Config:
    mat: MatConfig = field(default_factory=MatConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


def _section(cls, raw, name):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{name}: expected a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}")
    return cls(**raw)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name}: must be a finite number, got {value!r}")
    return float(value)


def _positive(value, name):
    _finite(value, name)
    if value <= 0:
        raise ConfigError(f"{name}: must be a positive number, got {value!r}")


def _non_negative(value, name):
    _finite(value, name)
    if value < 0:
        raise ConfigError(f"{name}: must be non-negative, got {value!r}")


def _non_negative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name}: must be a non-negative integer, got {value!r}")


def _pair(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise ConfigError(f"{name}: expected finite [x, y] numbers, got {value!r}")
    return float(value[0]), float(value[1])


def validate(cfg: Config) -> Config:
    m, a, v = cfg.mat, cfg.arm, cfg.vision
    for val, name in [(m.width_mm, "mat.width_mm"), (m.height_mm, "mat.height_mm"),
                      (m.marker_size_mm, "mat.marker_size_mm"),
                      (a.gripper_marker_size_mm, "arm.gripper_marker_size_mm"),
                      (a.base_zone_radius_mm, "arm.base_zone_radius_mm"),
                      (v.px_per_mm, "vision.px_per_mm"), (v.diff_threshold, "vision.diff_threshold"),
                      (v.min_area_mm2, "vision.min_area_mm2"), (v.gate_mm, "vision.gate_mm")]:
        _positive(val, name)

    _finite(v.fov_deg, "vision.fov_deg")
    if not (0 < v.fov_deg < 180):
        raise ConfigError(f"vision.fov_deg: must be between 0 and 180 degrees, got {v.fov_deg!r}")

    if not isinstance(m.markers, dict) or not m.markers:
        raise ConfigError("mat.markers: need at least one marker id -> [x, y]")

    valid_markers = {}
    for k, c in m.markers.items():
        if isinstance(k, bool) or not (isinstance(k, int) or (isinstance(k, str) and k.isdigit())):
            raise ConfigError(f"mat.markers: marker ID must be a non-negative integer, got {k!r}")
        mid = int(k)
        if mid < 0:
            raise ConfigError(f"mat.markers: marker ID must be a non-negative integer, got {mid!r}")
        valid_markers[mid] = _pair(c, f"mat.markers[{mid}]")
    m.markers = valid_markers

    if isinstance(m.min_markers, bool) or not isinstance(m.min_markers, int) or not (1 <= m.min_markers <= len(m.markers)):
        raise ConfigError(f"mat.min_markers: must be an integer between 1 and {len(m.markers)}")

    _non_negative_int(a.gripper_marker_id, "arm.gripper_marker_id")
    if a.gripper_marker_id in m.markers:
        raise ConfigError("arm.gripper_marker_id collides with a mat marker id")

    for mid, (x, y) in m.markers.items():
        h = m.marker_size_mm / 2
        if not (h <= x <= m.width_mm - h and h <= y <= m.height_mm - h):
            raise ConfigError(f"mat.markers[{mid}]: marker does not fit inside the mat")

    a.base_xy_mm = _pair(a.base_xy_mm, "arm.base_xy_mm")
    _finite(a.base_yaw_deg, "arm.base_yaw_deg")
    _finite(a.mat_z_mm, "arm.mat_z_mm")

    _finite(v.ema, "vision.ema")
    if not (0 < v.ema <= 1):
        raise ConfigError("vision.ema: must be in (0, 1]")

    _finite(v.round_aspect, "vision.round_aspect")
    if v.round_aspect < 1:
        raise ConfigError("vision.round_aspect: must be >= 1")

    _non_negative_int(v.max_missed, "vision.max_missed")
    _non_negative_int(v.max_pose_age, "vision.max_pose_age")
    _non_negative(v.grasp_inset_mm, "vision.grasp_inset_mm")
    _non_negative(v.edge_margin_mm, "vision.edge_margin_mm")
    _finite(v.grasp_z_mm, "vision.grasp_z_mm")

    return cfg


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    unknown = set(raw) - {"mat", "arm", "vision"}
    if unknown:
        raise ConfigError(f"unknown top-level keys {sorted(unknown)}")
    return validate(Config(_section(MatConfig, raw.get("mat"), "mat"),
                           _section(ArmConfig, raw.get("arm"), "arm"),
                           _section(VisionConfig, raw.get("vision"), "vision")))
