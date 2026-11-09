import pytest
import yaml

from armvision.config import ConfigError, load_config


def test_default_loads():
    cfg = load_config()
    assert set(cfg.mat.markers) == {0, 1, 2, 3}


@pytest.mark.parametrize("patch,msg", [
    ({"mat": {"width_mm": -1}}, "width_mm"),
    ({"mat": {"width_mm": float("nan")}}, "width_mm"),
    ({"mat": {"width_mm": float("inf")}}, "width_mm"),
    ({"mat": {"markers": {0: [1, 1]}, "min_markers": 1}}, "fit inside"),
    ({"mat": {"markers": {-1: [100, 100]}}}, "non-negative integer"),
    ({"mat": {"min_markers": 9}}, "min_markers"),
    ({"mat": {"min_markers": 2.5}}, "min_markers"),
    ({"arm": {"gripper_marker_id": 1}}, "collides"),
    ({"arm": {"gripper_marker_id": -1}}, "gripper_marker_id"),
    ({"arm": {"gripper_marker_id": 1.5}}, "gripper_marker_id"),
    ({"arm": {"base_xy_mm": [float("nan"), 100.0]}}, "base_xy_mm"),
    ({"arm": {"base_yaw_deg": float("nan")}}, "base_yaw_deg"),
    ({"arm": {"mat_z_mm": float("nan")}}, "mat_z_mm"),
    ({"vision": {"grasp_z_mm": float("nan")}}, "grasp_z_mm"),
    ({"vision": {"grasp_z_mm": float("inf")}}, "grasp_z_mm"),
    ({"vision": {"edge_margin_mm": -1.0}}, "edge_margin_mm"),
    ({"vision": {"edge_margin_mm": float("nan")}}, "edge_margin_mm"),
    ({"vision": {"max_missed": -1}}, "max_missed"),
    ({"vision": {"max_pose_age": -1}}, "max_pose_age"),
    ({"vision": {"fov_deg": 0.0}}, "fov_deg"),
    ({"vision": {"fov_deg": 190.0}}, "fov_deg"),
    ({"vision": {"ema": 0}}, "ema"),
    ({"vision": {"ema": float("nan")}}, "ema"),
    ({"vision": {"bogus": 1}}, "unknown keys"),
    ({"nope": {}}, "top-level"),
])
def test_invalid(tmp_path, patch, msg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(patch))
    with pytest.raises(ConfigError, match=msg):
        load_config(p)
