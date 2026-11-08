"""ArmInterface and the adapter that lets the planner drive K2's MockArm.

The planner speaks in OUR three joint angles (q1 base, q2 main arm, q3 horarm; radians,
the geometric angles of kinematic_model) plus a gripper open/close command. K2's MockArm
speaks in FIVE model-space radians. MockArmAdapter translates between the two:

    mock_value = sign * q + offset        (per mapped joint)

Joints that are not mapped (for example a wrist joint) keep their current target.
Anything the mock does not accept raises ValueError, because MockArm rejects out-of-limit
targets instead of clamping. A wrong mapping therefore fails loudly.

!! Confirm with K2 (see JointMap): which of the five joints are base / shoulder / elbow /
!! gripper, and whether the limits in arm_config.yaml use the same zero and sign as the
!! geometric angles in kinematic_model. If not, set signs / offsets here, do not edit the planner.

No sockets or real-time threads: time advances only through wait().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, Tuple

Vec3 = Tuple[float, float, float]


class ArmError(RuntimeError):
    """The arm (or its adapter) cannot do what was asked."""


class ArmInterface(Protocol):
    """What the planner needs from an arm. The mock, and later the real link, implement it."""

    def move_to(self, q: Sequence[float]) -> None:
        """Command joint angles (q1, q2, q3) in radians."""

    def set_gripper(self, state: str) -> None:
        """Command the gripper: "open" or "close"."""

    def get_q(self) -> Vec3:
        """Current joint angles (q1, q2, q3) in radians."""

    def wait(self, seconds: float) -> None:
        """Let time pass. Simulation: advance the model. Real arm: sleep."""

    def now(self) -> float:
        """Seconds since start (simulated or real)."""

    def speed_limits(self) -> Optional[Tuple[float, float, float]]:
        """Per-joint speed caps (rad/s) for q1, q2, q3, or None if unknown."""

    def wait_for_gripper(self, timeout_s: float = 10.0) -> None:
        """Advance time until the gripper reaches target, or raise ArmError."""


@dataclass(frozen=True)
class JointMap:
    """How our joints map onto the mock's five.

    Default offsets align K1's geometric angles (where shoulder/elbow are ~1.43 rad
    and ~-2.02 rad) with the standard centered joint limits [-0.8, 0.8] in arm_config.yaml.
    """

    arm_indices: Tuple[int, int, int] = (0, 1, 2)       # mock joints for q1, q2, q3
    signs: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # +1 or -1 per joint
    offsets: Tuple[float, float, float] = (0.0, -1.43345, 2.02065)  # radians, added after the sign
    gripper_index: Optional[int] = 4                    # None if the mock has no gripper joint
    gripper_open: Optional[float] = 1.0                 # mock value when open (rad)
    gripper_close: Optional[float] = 0.0                # mock value when closed (rad)

    def __post_init__(self):
        if len(set(self.arm_indices)) != 3:
            raise ValueError("arm_indices must be three different joint indices")
        if any(s not in (1.0, -1.0) for s in self.signs):
            raise ValueError("signs must be +1 or -1")



class MockArmAdapter:
    """Wraps K2's MockArm as an ArmInterface. Deterministic: time moves only in wait()."""

    def __init__(self, mock, joint_map: JointMap = JointMap(), tick: float = 0.01):
        if not (math.isfinite(tick) and tick > 0):
            raise ValueError("tick must be > 0")
        self._mock = mock
        self._map = joint_map
        self._tick = tick
        self._time = 0.0
        self._gripper_cmd: Optional[str] = None

    # ------------------------------------------------------------------ mapping

    def _target_vector(self):
        return list(self._mock.get_state()["target_q"])

    def _to_mock(self, q: Sequence[float]):
        vec = self._target_vector()
        for k, idx in enumerate(self._map.arm_indices):
            vec[idx] = self._map.signs[k] * q[k] + self._map.offsets[k]
        return vec

    def _from_mock(self, q5: Sequence[float]) -> Vec3:
        m = self._map
        return tuple((q5[idx] - m.offsets[k]) * m.signs[k] for k, idx in enumerate(m.arm_indices))  # type: ignore

    # ---------------------------------------------------------------- interface

    def move_to(self, q: Sequence[float]) -> None:
        q = [float(v) for v in q]
        if len(q) != 3 or not all(math.isfinite(v) for v in q):
            raise ValueError("q must be three finite joint angles (rad)")
        self._mock.move(self._to_mock(q))  # raises ValueError if outside the mock's limits

    def set_gripper(self, state: str) -> None:
        m = self._map
        if state not in ("open", "close"):
            raise ValueError(f"gripper state must be 'open' or 'close', got {state!r}")
        if m.gripper_index is None or m.gripper_open is None or m.gripper_close is None:
            raise ArmError("gripper is not configured: set gripper_index, gripper_open and "
                           "gripper_close in JointMap")
        vec = self._target_vector()
        vec[m.gripper_index] = m.gripper_open if state == "open" else m.gripper_close
        self._mock.move(vec)
        self._gripper_cmd = state

    def get_q(self) -> Vec3:
        return self._from_mock(self._mock.get_state()["q"])

    def wait(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not (math.isfinite(seconds) and seconds >= 0):
            raise ValueError("seconds must be a finite number >= 0")
        remaining = float(seconds)
        while remaining > 1e-12:
            step = min(self._tick, remaining)
            self._mock.advance(step)
            self._time += step
            remaining -= step

    def now(self) -> float:
        return self._time

    def speed_limits(self) -> Optional[Tuple[float, float, float]]:
        """Caps from the mock. Reads its private _velocity (ask K2 to expose a public one)."""
        velocity = getattr(self._mock, "_velocity", None)
        if velocity is None:
            return None
        return tuple(float(velocity[idx]) for idx in self._map.arm_indices)  # type: ignore

    # ------------------------------------------------------- extras (not required)

    def gripper_value(self) -> Optional[float]:
        """Current gripper joint value (mock radians), or None if not mapped."""
        idx = self._map.gripper_index
        return None if idx is None else float(self._mock.get_state()["q"][idx])

    def gripper_travel_time(self, state: Optional[str] = None) -> float:
        """Estimated or exact time in seconds for the gripper to travel to state."""
        m = self._map
        if m.gripper_index is None:
            return 0.0
        state = state or self._gripper_cmd
        if state is None:
            return 0.0
        target = m.gripper_open if state == "open" else m.gripper_close
        if target is None:
            return 0.0
        current = self.gripper_value()
        if current is None:
            return 0.0
        velocity = getattr(self._mock, "_velocity", None)
        if velocity is not None and m.gripper_index < len(velocity):
            v = float(velocity[m.gripper_index])
            if v > 0:
                return abs(target - current) / v
        return 0.0

    def wait_for_gripper(
        self,
        timeout_s: float = 10.0,
        tol: float = 1e-3,
        check_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Advance time until the gripper reaches its commanded target.

        Raises ArmError if the target is not reached within timeout_s seconds.
        """
        m = self._map
        if m.gripper_index is None:
            return
        if self._gripper_cmd is None:
            return
        target = m.gripper_open if self._gripper_cmd == "open" else m.gripper_close
        if target is None:
            return

        waited = 0.0
        while waited < timeout_s - 1e-9:
            curr = self.gripper_value()
            if curr is not None and abs(curr - target) <= tol:
                return
            if check_stop is not None:
                check_stop()
            step = min(self._tick, timeout_s - waited)
            self.wait(step)
            waited += step

        curr = self.gripper_value()
        if curr is None or abs(curr - target) > tol:
            curr_str = f"{curr:.3f}" if curr is not None else "unknown"
            raise ArmError(
                f"gripper did not reach {self._gripper_cmd} target ({target:.3f} rad) "
                f"within {timeout_s:.1f} s (current: {curr_str} rad)"
            )

    @property
    def gripper_command(self) -> Optional[str]:
        return self._gripper_cmd
