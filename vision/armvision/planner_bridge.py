"""Bridge between vision WorldState observations and the Kinematics motion planner.

Converts validated object grasp points (in base-frame millimeters) into planner-ready
Cartesian targets (in base-frame meters). Enforces localization freshness, tracking
validity, occlusion, finite coordinates, and kinematic reachability before motion.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .pipeline import TrackedObject, WorldState


class TargetValidationError(ValueError):
    """Raised when target localization, tracking, occlusion, or reachability checks fail."""


# Canonical dictionary of supported colors and common aliases
COLOR_DICT: dict[str, str] = {
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "blue": "blue",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "gray",
    "violet": "purple",
}


def normalize_color(color: Optional[str]) -> Optional[str]:
    """Normalize a color string using COLOR_DICT. Returns None if color is None or empty."""
    if not color:
        return None
    c = color.strip().lower()
    return COLOR_DICT.get(c, c)


def _get_kinematics():
    """Lazily import Kinematics and motion planner modules when reachability or motion is needed."""
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    _KINEMATICS_DIR = _REPO_ROOT / "Kinematics"
    if str(_KINEMATICS_DIR) not in sys.path:
        sys.path.insert(0, str(_KINEMATICS_DIR))
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    try:
        from Kinematics.kinematics import ik, is_reachable
        from Kinematics.planner import Planner, PlannerError
    except ImportError:
        try:
            from kinematics import ik, is_reachable
            from planner import Planner, PlannerError
        except ImportError as exc:
            raise ImportError(
                "Kinematics motion planner modules are required for arm motion and reachability checks. "
                "Ensure Kinematics dependencies are installed (Kinematics/requirements.txt)."
            ) from exc
    return ik, is_reachable, Planner, PlannerError


def _is_finite_coords(target: TrackedObject) -> bool:
    for coords in (target.grasp_mm, target.center_mm):
        if coords is None or len(coords) < 3:
            return False
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coords):
            return False
    return True


def _validate_coords(target: TrackedObject) -> None:
    for name, coords in [("grasp_mm", target.grasp_mm), ("center_mm", target.center_mm)]:
        if coords is None or len(coords) < 3:
            raise TargetValidationError(f"Target #{target.id} has invalid {name}: {coords}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coords):
            raise TargetValidationError(f"Target #{target.id} has non-finite {name}: {coords}")


def validate_target(
    state: WorldState,
    color: Optional[str] = None,
    position: Optional[str] = None,
    obj_id: Optional[int] = None,
    require_fresh_mat: bool = True,
) -> TrackedObject:
    """Validate mat localization, select target object, and enforce tracking and coordinate validity.

    Validation rules:
      1. Fresh mat localization: state.pose_fresh must be True (pose_age == 0 and mat_visible == True).
      2. If obj_id is specified: the requested object must exist, match color (if color is also provided),
         be fresh, non-occluded, and have finite coordinates. Conflicting criteria (e.g. asking for a blue
         object by a red object's ID) are strictly rejected with TargetValidationError.
      3. If selecting by color/position: filter usable candidates (seen, not occluded, finite) BEFORE ranking
         so that stale or occluded objects never block selection of an otherwise valid object.
      4. Target must be seen in the current frame (target.seen is True and target.missed == 0).
      5. Target must not be occluded by the arm (not target.occluded).
      6. All coordinates in grasp_mm and center_mm must be finite numbers.
    """
    if require_fresh_mat and not state.pose_fresh:
        raise TargetValidationError(
            f"Mat localization is not fresh (mat_visible={state.mat_visible}, pose_age={state.pose_age} frames)"
        )

    req_color = normalize_color(color)

    # 1. Explicit object ID requested: check that specific object directly
    if obj_id is not None:
        matching = [o for o in state.objects if o.id == obj_id]
        if not matching:
            raise TargetValidationError(f"No object found matching id={obj_id}")
        target = matching[0]

        # Enforce consistency: if color was also specified, it must match the target object's color
        if req_color is not None and target.color != req_color:
            raise TargetValidationError(
                f"Conflicting criteria: target #{target.id} has color {target.color!r}, "
                f"which does not match requested color {color!r}"
            )

        if not target.seen or target.missed > 0:
            raise TargetValidationError(
                f"Target #{target.id} ({target.color}) was not seen in current frame (missed={target.missed} frames)"
            )
        if target.occluded:
            raise TargetValidationError(f"Target #{target.id} ({target.color}) is occluded by the arm")
        _validate_coords(target)
        return target

    # 2. General selection (by color, position, or nearest): filter candidates by color
    candidates = state.objects
    if req_color is not None:
        candidates = [o for o in candidates if o.color == req_color]

    if not candidates:
        criteria = []
        if color:
            criteria.append(f"color={color!r}")
        if position:
            criteria.append(f"position={position!r}")
        crit_str = ", ".join(criteria) or "any"
        raise TargetValidationError(f"No object found matching {crit_str}")

    # Filter usable candidates BEFORE ranking so stale objects never block valid objects
    usable = [
        o for o in candidates
        if o.seen and o.missed == 0 and not o.occluded and _is_finite_coords(o)
    ]

    if usable:
        if position is not None:
            ranked = state.find(color=req_color, position=position, allow_stale=False, allow_occluded=False)
            matching_ranked = [o for o in ranked if o in usable]
            target = matching_ranked[0] if matching_ranked else usable[0]
        else:
            target = sorted(usable, key=lambda o: float(np.hypot(o.center_mm[0], o.center_mm[1])))[0]
        return target

    # No usable candidate exists: pick the candidate that best matches criteria to report exact failure
    if position is not None:
        ranked = state.find(color=req_color, position=position, allow_stale=True, allow_occluded=True)
        matching_ranked = [o for o in ranked if o in candidates]
        target = matching_ranked[0] if matching_ranked else candidates[0]
    else:
        target = sorted(candidates, key=lambda o: float(np.hypot(o.center_mm[0], o.center_mm[1])))[0]

    if not target.seen or target.missed > 0:
        raise TargetValidationError(
            f"Target #{target.id} ({target.color}) was not seen in current frame (missed={target.missed} frames)"
        )
    if target.occluded:
        raise TargetValidationError(f"Target #{target.id} ({target.color}) is occluded by the arm")
    _validate_coords(target)
    return target


def target_to_planner_coords(
    target: TrackedObject, grasp_z_m: Optional[float] = None
) -> Tuple[float, float, float]:
    """Convert target.grasp_mm to meters (x, y, z) for the planner.

    Vision already supplies base-frame coordinates—do NOT transform them again!
    Only scale from millimeters to meters. An explicit grasp_z_m overrides the Z height.
    """
    x_m = float(target.grasp_mm[0]) / 1000.0
    y_m = float(target.grasp_mm[1]) / 1000.0
    z_m = float(grasp_z_m) if grasp_z_m is not None else float(target.grasp_mm[2]) / 1000.0
    return (x_m, y_m, z_m)


def check_reachability(xyz_m: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Verify that target Cartesian coordinates in meters are reachable using K1's ik(x, y, z).

    Raises TargetValidationError if ik returns None. Returns joint angles (rad) on success.
    """
    ik, is_reachable, _, _ = _get_kinematics()
    sol = ik(*xyz_m)
    if sol is None:
        raise TargetValidationError(
            f"Target at ({xyz_m[0]:.3f}, {xyz_m[1]:.3f}, {xyz_m[2]:.3f}) m is unreachable (IK returned None)"
        )
    return sol


def execute_pick_and_place(
    planner: Planner,
    state: WorldState,
    destination_m: Tuple[float, float, float] = (0.24, -0.05, 0.12),
    color: Optional[str] = None,
    position: Optional[str] = None,
    obj_id: Optional[int] = None,
    grasp_z_m: Optional[float] = None,
    require_fresh_mat: bool = True,
) -> Tuple[TrackedObject, Tuple[float, float, float]]:
    """Validate target, convert coordinates, verify reachability, and command planner.pick_and_place.

    Returns:
        (target_object, grasp_coordinates_meters)
    """
    _get_kinematics()
    target = validate_target(
        state, color=color, position=position, obj_id=obj_id, require_fresh_mat=require_fresh_mat
    )
    grasp_m = target_to_planner_coords(target, grasp_z_m=grasp_z_m)
    check_reachability(grasp_m)
    check_reachability(destination_m)
    planner.pick_and_place(grasp_m, destination_m)
    return target, grasp_m
