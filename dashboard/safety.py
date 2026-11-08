"""Safety Layer: Visible boundary enforcement, collision prevention, and E-STOP latching."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dashboard.events import emit

# Safety boundary limits
WORKSPACE_RADIUS_MIN = 0.08   # meters from base
WORKSPACE_RADIUS_MAX = 0.30   # meters max reach
WORKSPACE_Z_MIN = 0.00        # meters (table level - collision prevention)
WORKSPACE_Z_MAX = 0.26        # meters max safe height

# Joint limits in radians (from arm_config.yaml)
JOINT_LIMITS_RAD = {
    "base": (-1.0, 1.0),
    "shoulder": (-0.8, 0.8),
    "elbow": (-0.8, 0.8),
    "wrist": (-0.8, 0.8),
    "gripper": (0.0, 1.0),
}

# Joint limits in degrees (for firmware comparison)
JOINT_LIMITS_DEG = {
    "base": (30, 150),
    "shoulder": (60, 120),
    "elbow": (45, 135),
    "gripper": (70, 110),
}


@dataclass
class SafetyResult:
    allowed: bool
    status: str            # "accepted" | "rejected" | "needs_clarification"
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SafetyManager:
    def __init__(self):
        self._estop_active: bool = False
        self._last_rejection_reason: Optional[str] = None
        self._rejection_count: int = 0

    @property
    def is_stopped(self) -> bool:
        return self._estop_active

    @property
    def last_rejection(self) -> Optional[str]:
        return self._last_rejection_reason

    def trigger_estop(self, reason: str = "Operator pressed STOP") -> None:
        """Trigger software emergency stop. The physical kill switch remains real safety."""
        self._estop_active = True
        emit("safety:estop", {
            "triggered_by": "operator",
            "reason": reason,
            "warning": "Software motion halted. Physical kill switch remains the primary hardware safety.",
        })

    def reset_estop(self) -> None:
        """Reset software emergency stop back to safe ready state."""
        self._estop_active = False
        self._last_rejection_reason = None
        emit("safety:reset", {
            "reset_by": "operator",
            "state": "ready",
        })

    def reject(self, reason: str, details: Optional[Dict[str, Any]] = None) -> SafetyResult:
        """Helper to record and emit a safety rejection."""
        self._rejection_count += 1
        self._last_rejection_reason = reason
        d = details or {}
        emit("safety:rejected", {
            "reason": reason,
            "details": d,
            "rejection_number": self._rejection_count,
        })
        return SafetyResult(allowed=False, status="rejected", reason=reason, details=d)

    def validate_cartesian_target(self, x: float, y: float, z: float) -> SafetyResult:
        """Verify (x, y, z) target against workspace boundaries and reachability."""
        if self._estop_active:
            return self.reject("E-STOP is latched active. Release/reset before commanding motion.")

        # Check table collision
        if z < WORKSPACE_Z_MIN:
            return self.reject(
                f"Table collision hazard: commanded height z={z*1000:.1f}mm is below table plane (min {WORKSPACE_Z_MIN*1000:.0f}mm)",
                {"target": [x, y, z], "boundary": "table_plane", "z_min": WORKSPACE_Z_MIN},
            )

        if z > WORKSPACE_Z_MAX:
            return self.reject(
                f"Height limit breach: commanded height z={z*1000:.1f}mm exceeds safe ceiling (max {WORKSPACE_Z_MAX*1000:.0f}mm)",
                {"target": [x, y, z], "boundary": "ceiling", "z_max": WORKSPACE_Z_MAX},
            )

        # Check radial reach
        radius = math.hypot(x, y)
        if radius < WORKSPACE_RADIUS_MIN:
            return self.reject(
                f"Singularity / Base collision hazard: radial distance {radius*1000:.1f}mm is too close to base (min {WORKSPACE_RADIUS_MIN*1000:.0f}mm)",
                {"target": [x, y, z], "radius": radius, "radius_min": WORKSPACE_RADIUS_MIN},
            )

        if radius > WORKSPACE_RADIUS_MAX:
            return self.reject(
                f"Out of workspace reach: distance {radius*1000:.1f}mm exceeds maximum arm envelope ({WORKSPACE_RADIUS_MAX*1000:.0f}mm)",
                {"target": [x, y, z], "radius": radius, "radius_max": WORKSPACE_RADIUS_MAX},
            )

        # Kinematic IK reachability check
        try:
            from kinematics import is_reachable
            if not is_reachable(x, y, z):
                return self.reject(
                    f"Kinematic singularity / unreachable posture for target ({x:.3f}, {y:.3f}, {z:.3f})m",
                    {"target": [x, y, z], "ik_reachable": False},
                )
        except Exception:
            pass  # Fall back to radius and z bounds

        return SafetyResult(allowed=True, status="accepted", details={"target": [x, y, z], "radius": radius})

    def validate_joint_target(self, joint_name: str, angle_rad: float) -> SafetyResult:
        """Verify commanded joint angle against physical joint limits."""
        if self._estop_active:
            return self.reject("E-STOP is latched active. Release/reset before commanding motion.")

        if joint_name not in JOINT_LIMITS_RAD:
            return self.reject(f"Unknown joint name {joint_name!r}")

        min_rad, max_rad = JOINT_LIMITS_RAD[joint_name]
        if not (min_rad <= angle_rad <= max_rad):
            deg = math.degrees(angle_rad)
            min_deg, max_deg = math.degrees(min_rad), math.degrees(max_rad)
            return self.reject(
                f"Joint safety limit violation: {joint_name} angle {deg:.1f}° ({angle_rad:.2f} rad) is outside allowed range [{min_deg:.1f}°, {max_deg:.1f}°]",
                {"joint": joint_name, "angle_rad": angle_rad, "limits": [min_rad, max_rad]},
            )

        return SafetyResult(allowed=True, status="accepted", details={"joint": joint_name, "angle_rad": angle_rad})

    def evaluate_action(self, action_dict: Dict[str, Any], context_targets: Optional[List[Dict[str, Any]]] = None) -> SafetyResult:
        """Evaluate a parsed NLP action against safety rules."""
        if self._estop_active:
            return self.reject("E-STOP is latched active. Motion halted. Reset safety before executing commands.")

        act = action_dict.get("action")
        if not act:
            return self.reject("Invalid command: action payload missing")

        # Gestures and dialogue replies are safe
        if act in ("reply", "shake", "home", "return_to_zero", "open_hand", "close_hand"):
            return SafetyResult(allowed=True, status="accepted", details={"action": act})

        # Relative movements
        if act == "move_relative":
            direction = action_dict.get("direction", "")
            distance_mm = action_dict.get("distance_mm") or 20.0
            if distance_mm > 150.0:
                return self.reject(
                    f"Relative move step {distance_mm:.1f}mm exceeds single-step safety cap (max 150mm)",
                    {"direction": direction, "distance_mm": distance_mm},
                )
            return SafetyResult(allowed=True, status="accepted", details={"action": act, "direction": direction})

        # Joint rotation
        if act == "rotate_joint":
            joint_idx = action_dict.get("joint", 0)
            angle = action_dict.get("angle")
            joint_names = ["base", "shoulder", "elbow", "wrist", "gripper"]
            if joint_idx < 0 or joint_idx >= len(joint_names):
                return self.reject(f"Joint index {joint_idx} is out of range [0..4]")
            jname = joint_names[joint_idx]
            if angle is not None:
                # angle given in degrees or radians
                angle_rad = math.radians(angle) if abs(angle) > 3.15 else angle
                return self.validate_joint_target(jname, angle_rad)
            return SafetyResult(allowed=True, status="accepted", details={"action": act})

        # Pick and place or Cartesian move actions
        if act in ("move_cartesian", "goto", "pick_up", "put_in", "put_down", "give"):
            # If coordinates are directly supplied
            xyz = action_dict.get("xyz")
            if xyz and len(xyz) == 3:
                return self.validate_cartesian_target(xyz[0], xyz[1], xyz[2])

            target_ref = action_dict.get("target") or {}

            # If an object target is specified, check against detected targets if available
            noun = target_ref.get("noun")
            color = target_ref.get("color")
            if context_targets is not None:
                matches = [
                    t for t in context_targets
                    if (not color or t.get("color") == color) and (not noun or noun in t.get("label", ""))
                ]
                if not matches and (noun or color):
                    return SafetyResult(
                        allowed=False,
                        status="needs_clarification",
                        reason=f"Target object '{color or ''} {noun or 'object'}'.strip() was not detected in camera view.",
                        details={"target_ref": target_ref, "detected_count": len(context_targets)},
                    )
                if len(matches) > 1:
                    return SafetyResult(
                        allowed=False,
                        status="needs_clarification",
                        reason=f"Ambiguous target: found {len(matches)} matching objects. Please specify position (e.g. 'left' or 'right').",
                        details={"matches": matches},
                    )

            # Default safe nominal coordinates for pick & place
            return SafetyResult(allowed=True, status="accepted", details={"action": act, "target": target_ref})

        return SafetyResult(allowed=True, status="accepted", details={"action": act})


# Global singleton safety manager
safety = SafetyManager()
