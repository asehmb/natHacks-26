"""Pick and place primitives: hover, descend, grasp, lift, place, retract.

Each primitive returns a list of Steps (poses). Nothing here talks to an arm; the planner
executes the steps. Units: METERS. Frame: x forward, y left, z up, origin on the base axis
at table height (same as kinematics.py).

A Step is "go to this tip position, then do this gripper action":
    Step(name, xyz, gripper, speed, dwell_s)
Straight vertical moves (descend, lift) and the travel moves between poses are expanded
into small waypoints, so the arm follows a straight line in Cartesian space instead of the
arc a single joint-space move would take.

The sequence builders (pick_sequence, place_sequence, pick_and_place_sequence) run
validate_sequence() before returning and raise SequenceError if ANY pose is unreachable,
so the arm never executes half of a sequence.

Object / destination positions are the tip position at which the gripper should close / open.
All constants live in PickPlaceConfig; move them into arm_config.yaml once K2's loader exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from kinematics import is_reachable, line_points

Vec3 = Tuple[float, float, float]

OPEN = "open"
CLOSE = "close"
_GRIPPER_ACTIONS = (OPEN, CLOSE, None)  # None = leave the gripper as it is


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Step:
    """One pose in a sequence."""

    name: str                       # hover, descend, grasp, lift, place, retract, move
    xyz: Vec3                       # tip position to reach (meters)
    gripper: Optional[str] = None   # OPEN / CLOSE to act at this pose, None = unchanged
    speed: float = 1.0              # fraction of the maximum speed, 0 < speed <= 1
    dwell_s: float = 0.0            # wait this long after arriving (e.g. for the gripper)


@dataclass(frozen=True)
class PickPlaceConfig:
    """Constants shared by all primitives (meters, seconds). Replace with arm_config.yaml."""

    hover_clearance: float = 0.04   # height above the grasp / place point to hover at
    line_step: float = 0.01         # waypoint spacing for descend and lift
    transit_step: float = 0.03      # waypoint spacing for travel moves
    home: Vec3 = (0.20, 0.0, 0.15)  # safe pose to start from and retract to
    min_z: float = 0.0              # never go below the table plane
    approach_speed: float = 0.3     # descend / lift (slow, near the object)
    travel_speed: float = 0.7       # moves between poses
    grasp_dwell_s: float = 0.5      # wait for the gripper to close
    release_dwell_s: float = 0.3    # wait for the gripper to open


DEFAULT_CONFIG = PickPlaceConfig()


class SequenceError(ValueError):
    """A sequence contains an unreachable or malformed step and must not be executed."""

    def __init__(self, result: "ValidationResult"):
        super().__init__(result.reason)
        self.result = result


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    index: Optional[int] = None     # index of the first bad step
    step: Optional[Step] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------- validation


def validate_sequence(steps: Sequence[Step], cfg: PickPlaceConfig = DEFAULT_CONFIG) -> ValidationResult:
    """Check every step; return the FIRST problem (never raises).

    A step is bad if its pose is not finite, is below the table (cfg.min_z), is not
    reachable (kinematics.is_reachable runs IK, so reach, singularity margin and joint
    limits are all covered), or has a bad gripper action, speed or dwell time.
    """
    if not steps:
        return ValidationResult(False, None, None, "sequence is empty")
    for i, s in enumerate(steps):
        def bad(reason: str) -> ValidationResult:
            return ValidationResult(False, i, s, f"step {i} ({s.name}): {reason}")

        if len(s.xyz) != 3 or not all(math.isfinite(v) for v in s.xyz):
            return bad(f"pose {s.xyz} is not a finite (x, y, z)")
        if s.xyz[2] < cfg.min_z:
            return bad(f"z={s.xyz[2]:.3f} m is below the table limit {cfg.min_z:.3f} m")
        if not is_reachable(*s.xyz):
            return bad(f"pose {tuple(round(v, 3) for v in s.xyz)} m is unreachable")
        if s.gripper not in _GRIPPER_ACTIONS:
            return bad(f"unknown gripper action {s.gripper!r}")
        if not (math.isfinite(s.speed) and 0.0 < s.speed <= 1.0):
            return bad(f"speed {s.speed} must be in (0, 1]")
        if not (math.isfinite(s.dwell_s) and s.dwell_s >= 0.0):
            return bad(f"dwell_s {s.dwell_s} must be >= 0")
    return ValidationResult(True)


def _require_valid(steps: List[Step], cfg: PickPlaceConfig) -> List[Step]:
    result = validate_sequence(steps, cfg)
    if not result:
        raise SequenceError(result)
    return steps


def _check_xyz(name: str, xyz: Sequence[float]) -> Vec3:
    """Guard against a malformed position or millimeters passed as meters."""
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        raise ValueError(f"{name} must be a finite (x, y, z), got {xyz!r}")
    if max(abs(v) for v in xyz) > 2.0:
        raise ValueError(f"{name}={tuple(xyz)} is larger than 2 m: positions are in meters, not mm")
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _above(xyz: Vec3, height: float) -> Vec3:
    return (xyz[0], xyz[1], xyz[2] + height)


def _line_steps(name: str, p0: Vec3, p1: Vec3, step: float, speed: float) -> List[Step]:
    """Steps along the straight line p0 -> p1, EXCLUDING p0 (the arm is already there)."""
    return [Step(name, p, None, speed) for p in line_points(p0, p1, step)[1:]]


# -------------------------------------------------------------------- primitives


def hover(xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG, gripper: Optional[str] = None) -> List[Step]:
    """Move to cfg.hover_clearance above xyz (one move; gripper optionally set on arrival)."""
    p = _check_xyz("xyz", xyz)
    return [Step("hover", _above(p, cfg.hover_clearance), gripper, cfg.travel_speed)]


def descend(xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG) -> List[Step]:
    """Straight vertical line from the hover pose above xyz down to xyz (hover pose excluded)."""
    p = _check_xyz("xyz", xyz)
    return _line_steps("descend", _above(p, cfg.hover_clearance), p, cfg.line_step, cfg.approach_speed)


def grasp(xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG) -> List[Step]:
    """Close the gripper at xyz and wait for it."""
    p = _check_xyz("xyz", xyz)
    return [Step("grasp", p, CLOSE, cfg.approach_speed, cfg.grasp_dwell_s)]


def lift(xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG) -> List[Step]:
    """Straight vertical line from xyz up to the hover height (xyz excluded)."""
    p = _check_xyz("xyz", xyz)
    return _line_steps("lift", p, _above(p, cfg.hover_clearance), cfg.line_step, cfg.approach_speed)


def place(xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG) -> List[Step]:
    """Open the gripper at xyz and wait for it."""
    p = _check_xyz("xyz", xyz)
    return [Step("place", p, OPEN, cfg.approach_speed, cfg.release_dwell_s)]


def retract(xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG) -> List[Step]:
    """Straight up from xyz to hover height, then straight to cfg.home (xyz excluded)."""
    p = _check_xyz("xyz", xyz)
    up = _above(p, cfg.hover_clearance)
    steps = _line_steps("retract", p, up, cfg.line_step, cfg.approach_speed)
    steps += _line_steps("retract", up, cfg.home, cfg.transit_step, cfg.travel_speed)
    return steps


def _approach(target: Vec3, from_xyz: Optional[Sequence[float]], cfg: PickPlaceConfig,
              gripper: Optional[str]) -> List[Step]:
    """Get to the hover pose above target: straight line from from_xyz, or one move if unknown."""
    hover_pose = hover(target, cfg, gripper)
    if from_xyz is None:
        return hover_pose
    start = _check_xyz("from_xyz", from_xyz)
    steps = _line_steps("move", start, hover_pose[0].xyz, cfg.transit_step, cfg.travel_speed)
    if steps:  # the last travel waypoint IS the hover pose: turn it into the hover step
        steps[-1] = hover_pose[0]
    else:      # already at the hover pose
        steps = hover_pose
    return steps


def move_sequence(target_xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG,
                  from_xyz: Optional[Sequence[float]] = None) -> List[Step]:
    """Straight line from from_xyz (the tip's current position) to target_xyz. Validated.

    Without from_xyz it is a single move. Use this to send the arm to any chosen point.
    """
    p = _check_xyz("target_xyz", target_xyz)
    steps: List[Step] = []
    if from_xyz is not None:
        steps = _line_steps("move", _check_xyz("from_xyz", from_xyz), p, cfg.transit_step, cfg.travel_speed)
    if not steps:  # no start given, or already there: keep one step so execution is uniform
        steps = [Step("move", p, None, cfg.travel_speed)]
    return _require_valid(steps, cfg)


# --------------------------------------------------------------------- sequences


def pick_sequence(object_xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG,
                  from_xyz: Optional[Sequence[float]] = None) -> List[Step]:
    """Hover (gripper open), descend, grasp, lift. Validated; raises SequenceError if unreachable.

    from_xyz is where the tip is now (e.g. cfg.home); the approach is then a straight line.
    """
    p = _check_xyz("object_xyz", object_xyz)
    steps = _approach(p, from_xyz, cfg, OPEN) + descend(p, cfg) + grasp(p, cfg) + lift(p, cfg)
    return _require_valid(steps, cfg)


def place_sequence(dest_xyz: Sequence[float], cfg: PickPlaceConfig = DEFAULT_CONFIG,
                   from_xyz: Optional[Sequence[float]] = None) -> List[Step]:
    """Hover, descend, place (gripper opens), retract to home. Validated like pick_sequence."""
    p = _check_xyz("dest_xyz", dest_xyz)
    steps = _approach(p, from_xyz, cfg, None) + descend(p, cfg) + place(p, cfg) + retract(p, cfg)
    return _require_valid(steps, cfg)


def pick_and_place_sequence(object_xyz: Sequence[float], dest_xyz: Sequence[float],
                            cfg: PickPlaceConfig = DEFAULT_CONFIG,
                            start_xyz: Optional[Sequence[float]] = None) -> List[Step]:
    """Full move: pick at object_xyz, carry at hover height, place at dest_xyz, retract home.

    start_xyz is where the tip is now (default cfg.home). The whole sequence is validated
    once at the end, so a single unreachable pose rejects everything.
    """
    obj = _check_xyz("object_xyz", object_xyz)
    dst = _check_xyz("dest_xyz", dest_xyz)
    start = cfg.home if start_xyz is None else _check_xyz("start_xyz", start_xyz)

    steps = _approach(obj, start, cfg, OPEN) + descend(obj, cfg) + grasp(obj, cfg) + lift(obj, cfg)
    carry_from = _above(obj, cfg.hover_clearance)              # where lift() ended
    steps += _approach(dst, carry_from, cfg, None)             # level carry to above the destination
    steps += descend(dst, cfg) + place(dst, cfg) + retract(dst, cfg)
    return _require_valid(steps, cfg)
