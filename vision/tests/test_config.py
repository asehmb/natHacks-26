import pytest
import yaml

from armvision.config import ConfigError, load_config


def test_default_loads():
    cfg = load_config()
    assert set(cfg.mat.markers) == {0, 1, 2, 3}


@pytest.mark.parametrize("patch,msg", [
    ({"mat": {"width_mm": -1}}, "width_mm"),
    ({"mat": {"markers": {0: [1, 1]}, "min_markers": 1}}, "fit inside"),
    ({"mat": {"min_markers": 9}}, "min_markers"),
    ({"arm": {"gripper_marker_id": 1}}, "collides"),
    ({"vision": {"ema": 0}}, "ema"),
    ({"vision": {"bogus": 1}}, "unknown keys"),
    ({"nope": {}}, "top-level"),
])
def test_invalid(tmp_path, patch, msg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(patch))
    with pytest.raises(ConfigError, match=msg):
        load_config(p)
