"""Team-facing kinematics interface (the stubs agreed in the brief).

Thin adapter over kinematic_model.EEZYbotARM_Mk2, which works in mm and degrees.
All unit conversion happens HERE, so nobody else ever sees mm or degrees.

Public interface
    fk(q)                    joints (rad)        -> (x, y, z) in meters
    ik(x, y, z)              meters              -> (q1, q2, q3) in rad, or None
    is_reachable(x, y, z)    meters              -> bool (authoritative: IK accepts it)
    joint_to_servo(q)        joints (rad)        -> servo angles (deg), raises JointLimitError
    servo_to_joint(s)        servo angles (deg)  -> joints (rad)
    workspace_bounds()       -> dict in meters / radians (quick-reject box, see docstring)
    plan_move(q0, q1, vmax, amax, dt) -> list of joint tuples (rad), all joints finish together
    line_points(p0, p1, step_m) -> waypoints (m) along a straight line
    cartesian_line(p0, p1, step_m) -> joint tuples (rad) for a straight line, or None
    joint_positions(q)       joints (rad) -> 5 points (m): base, shoulder, elbow, wrist, tip
    configure_arm(config)    reconfigure model link lengths from arm_config.yaml or dict

Frame: right-handed, origin on the base axis at table height, x forward, y left, z up.
Joint angles are the geometric joint angles of kinematic_model (q1 base yaw,
q2 main-arm elevation, q3 horarm angle, negative = elbow up), NOT servo angles.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

try:
    from Kinematics.config import load_config
except ImportError:
    try:
        from config import load_config
    except ImportError:
        load_config = None

from kinematic_model import EEZYbotARM_Mk2, line_points, plan_move as _plan_move

Vec3 = Tuple[float, float, float]

_MM_PER_M = 1000.0

# One model instance. Only its constants and pure functions are used; its stored
# q1/q2/q3 state is never touched.
_ARM = EEZYbotARM_Mk2(0, 80, -90)


def configure_arm(config: Optional[dict] = None) -> None:
    """Update model link dimensions from arm_config.yaml or a passed dict.

    Maps:
      links.offsets.shoulder_xyz[2] -> L1 (base height to shoulder pivot, mm)
      links.lengths.upper_arm       -> L2 (main arm length, mm)
      links.lengths.forearm         -> L3 (horarm length, mm)
      links.lengths.tool            -> L4 (tool/gripper radial offset, mm)
    """
    global _ARM, _RAW_CACHE
    if config is None and load_config is not None:
        try:
            config = load_config()
        except Exception:
            return
    if not isinstance(config, dict):
        return

    try:
        links = config.get("links", {})
        lengths = links.get("lengths", {})
        offsets = links.get("offsets", {})
        shoulder_xyz = offsets.get("shoulder_xyz")
        if shoulder_xyz is not None and len(shoulder_xyz) >= 3 and shoulder_xyz[2] is not None:
            _ARM.L1 = float(shoulder_xyz[2]) * _MM_PER_M
        if "upper_arm" in lengths and lengths["upper_arm"] is not None:
            _ARM.L2 = float(lengths["upper_arm"]) * _MM_PER_M
        if "forearm" in lengths and lengths["forearm"] is not None:
            _ARM.L3 = float(lengths["forearm"]) * _MM_PER_M
        if "tool" in lengths and lengths["tool"] is not None:
            _ARM.L4 = float(lengths["tool"]) * _MM_PER_M
        _RAW_CACHE = None  # invalidate bounds cache
    except Exception:
        pass


# Initialize with geometry from arm_config.yaml
configure_arm()

# Per-servo commanded limits (deg) for servos 1, 2, 3: the firmware's, see kinematic_model SERVO_LIMITS_DEG.
SERVO_LIMITS_DEG = EEZYbotARM_Mk2.SERVO_LIMITS_DEG


class JointLimitError(ValueError):
    """Joint angles, or the servo angles they map to, are out of range."""


def _deg(q: Sequence[float]) -> Vec3:
    return tuple(math.degrees(v) for v in q)  # type: ignore[return-value]


def _rad(q: Sequence[float]) -> Vec3:
    return tuple(math.radians(v) for v in q)  # type: ignore[return-value]


# --------------------------------------------------------------------- FK / IK


def fk(q: Sequence[float]) -> Vec3:
    """Joint angles (rad) -> tip position (x, y, z) in meters. Pure geometry, no limit check."""
    q1, q2, q3 = _deg(q)
    x, y, z = _ARM.forwardKinematics(q1=q1, q2=q2, q3=q3)
    return x / _MM_PER_M, y / _MM_PER_M, z / _MM_PER_M


def ik(x: float, y: float, z: float) -> Optional[Vec3]:
    """Tip position in meters -> (q1, q2, q3) in radians, or None.

    None means unreachable, too close to the elbow singularity, or outside the
    joint limits. Never raises. Elbow-up only.
    """
    sol = _ARM.inverseKinematics(x * _MM_PER_M, y * _MM_PER_M, z * _MM_PER_M)
    return None if sol is None else _rad(sol)


def is_reachable(x: float, y: float, z: float) -> bool:
    """True only if ik() accepts the target. Use this as the final reachability check."""
    return ik(x, y, z) is not None


# ---------------------------------------------------------------------- servos


def joint_to_servo(q: Sequence[float]) -> Vec3:
    """Joint angles (rad) -> servo angles (deg) for servos 1, 2, 3.

    Uses the model's current servo mapping (to be replaced by K2's calibration fit).
    Raises JointLimitError if the joints or resulting servo angles are out of range,
    so a bad command fails loudly instead of reaching a servo. Servo angles come back
    rounded to 0.01 deg by the underlying model.
    """
    q1, q2, q3 = _deg(q)
    try:
        servos = _ARM.map_kinematicsToServoAngles(q1=q1, q2=q2, q3=q3)
    except Exception as exc:  # the model raises a bare Exception on limit violations
        raise JointLimitError(f"{exc} (angles are in radians here; got {tuple(q)})") from exc
    for name, s, (lo, hi) in zip(("servo1", "servo2", "servo3"), servos, SERVO_LIMITS_DEG):
        if not lo <= s <= hi:
            raise JointLimitError(f"{name}={s} deg is outside the servo range ({lo}, {hi})")
    return servos


def servo_to_joint(servo_deg: Sequence[float]) -> Vec3:
    """Servo angles (deg) -> joint angles (rad). Inverse of joint_to_servo.

    MUST stay in sync with map_kinematicsToServoAngles (a test guards this):
        servo1 = -2.0497 * q1 + 91.726
        servo2 = 180 - q2
        servo3 = q2 + q3 + 135
    """
    s1, s2, s3 = servo_deg
    q1 = (s1 - 91.726) / -2.0497
    q2 = 180.0 - s2
    q3 = s3 - 135.0 - q2
    return _rad((q1, q2, q3))


# ------------------------------------------------------------------- workspace

_RAW_CACHE: Optional[dict] = None


def _raw_bounds() -> dict:
    """Unshrunk bounds from the model's in-memory sampling, converted to meters / radians."""
    global _RAW_CACHE
    if _RAW_CACHE is None:
        raw = _ARM.workspaceBounds(margin_mm=0.0, margin_deg=0.0)  # mm / degrees
        _RAW_CACHE = {
            k: (math.radians(v) if k.startswith("q1") else v / _MM_PER_M) for k, v in raw.items()
        }
    return _RAW_CACHE


def workspace_bounds(margin_m: float = 0.01, margin_rad: float = math.radians(2.0)) -> dict:
    """Bounding box of the reachable workspace, shrunk by a safety margin.

    Keys: r_*, x_*, y_*, z_* in meters (r = distance from the base axis), q1_* in radians.

    This is a cheap NECESSARY check: anything outside the box is never reachable,
    but a point inside it may still be (the real envelope is not a box). The
    validator should use is_reachable() / ik() as the final check, and the box
    only for fast rejection. Pass margin_m=0, margin_rad=0 for the raw bounds.
    """
    out = {}
    for key, value in _raw_bounds().items():
        shrink = margin_rad if key.startswith("q1") else margin_m
        out[key] = value + shrink if key.endswith("_min") else value - shrink
    return out


# ------------------------------------------------------------------ trajectory


def plan_move(
    q_start: Sequence[float],
    q_goal: Sequence[float],
    vmax,
    amax,
    dt: float,
) -> List[Vec3]:
    """Joint-space move from q_start to q_goal (rad), trapezoidal velocity profile.

    vmax in rad/s and amax in rad/s^2: one number for all joints, or one per joint.
    All joints start and stop together; the slowest joint sets the duration. The first
    sample is q_start, the last is exactly q_goal, samples are dt seconds apart.

    Raises JointLimitError if q_start or q_goal is outside the joint limits (a straight
    line in joint space between two valid poses stays inside the limits).
    Raises ValueError for bad dt / limits / mismatched lengths.
    """
    for label, q in (("q_start", q_start), ("q_goal", q_goal)):
        try:
            _ARM.checkErrorJointLimits(q1=math.degrees(q[0]), q2=math.degrees(q[1]), q3=math.degrees(q[2]))
        except (IndexError, TypeError) as exc:
            raise ValueError(f"{label} must be three joint angles (rad)") from exc
        except Exception as exc:
            raise JointLimitError(f"{label}: {exc} (angles are in radians here)") from exc
    return _plan_move(q_start, q_goal, vmax, amax, dt)


# ------------------------------------------------------------ straight lines


def cartesian_line(p_start: Sequence[float], p_goal: Sequence[float], step_m: float = 0.005) -> Optional[List[Vec3]]:
    """Joint angles (rad) for a straight Cartesian line from p_start to p_goal (meters).

    Waypoints are at most step_m apart. Returns None if any waypoint is unreachable.
    Use for vertical descend / lift moves; raises ValueError for malformed input.
    """
    pts = line_points(p_start, p_goal, step_m)
    joints: List[Vec3] = []
    for p in pts:
        q = ik(*p)
        if q is None:
            return None
        joints.append(q)
    return joints


# ---------------------------------------------------------------- drawing helper


def joint_positions(q: Sequence[float]) -> List[Vec3]:
    """Joint angles (rad) -> 3D points (m) of base, shoulder, elbow, wrist and tip.

    For drawing the arm. The last point equals fk(q): the gripper stays level, so the
    tip is the wrist plus L4 straight outward along the arm's vertical plane.
    """
    q1, q2, q3 = q
    L1, L2, L3, L4 = (v / _MM_PER_M for v in (_ARM.L1, _ARM.L2, _ARM.L3, _ARM.L4))
    elbow_r, elbow_z = L2 * math.cos(q2), L1 + L2 * math.sin(q2)
    wrist_r, wrist_z = elbow_r + L3 * math.cos(q2 + q3), elbow_z + L3 * math.sin(q2 + q3)
    planar = [(0.0, 0.0), (0.0, L1), (elbow_r, elbow_z), (wrist_r, wrist_z), (wrist_r + L4, wrist_z)]
    c, s = math.cos(q1), math.sin(q1)
    return [(r * c, r * s, z) for r, z in planar]
