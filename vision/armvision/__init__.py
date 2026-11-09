from .config import Config, ConfigError, load_config
from .pipeline import VisionSystem, WorldState

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "VisionSystem",
    "WorldState",
    "TargetValidationError",
    "validate_target",
    "target_to_planner_coords",
    "check_reachability",
    "execute_pick_and_place",
    "COLOR_DICT",
    "normalize_color",
]

_BRIDGE_EXPORTS = {
    "TargetValidationError",
    "validate_target",
    "target_to_planner_coords",
    "check_reachability",
    "execute_pick_and_place",
    "COLOR_DICT",
    "normalize_color",
}


def __getattr__(name: str):
    if name in _BRIDGE_EXPORTS:
        from . import planner_bridge

        return getattr(planner_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
