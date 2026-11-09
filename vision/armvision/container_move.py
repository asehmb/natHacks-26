"""Send the arm to hover above a container found by armvision.containers.

    arm = make_arm(port="/dev/cu.usbserial-10")      # real arm over USB; port=None = Kinematics MockArm simulation
    container, xyz_m = move_above_container(arm, state, "red")    # 20 mm above the rim
    arm.close()

Vision already reports positions in the arm base frame (x forward, y left, z up, mm); the only conversion here is
mm -> m. Every check (container found, finite, IK-reachable) runs before anything is sent to the arm.

NLP hook: dispatcher.register(PutIn, make_put_in_handler(lambda: latest_state, arm)).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

from .containers import Container, ContainerState
from .planner_bridge import TargetValidationError, _get_kinematics, check_reachability, normalize_color

Vec3 = Tuple[float, float, float]

DEFAULT_CLEARANCE_MM = 20.0
_COMMS_DIR = Path(__file__).resolve().parent.parent.parent / "firmware" / "comms"


class ArmMoveError(RuntimeError):
    """The arm backend could not carry out the move."""


def hover_point_m(c: Container, clearance_mm: float = DEFAULT_CLEARANCE_MM) -> Vec3:
    """Tip position (m) straight above the container's centre, clearance_mm above its rim."""
    return (float(c.center_mm[0]) / 1000.0, float(c.center_mm[1]) / 1000.0,
            (float(c.height_mm) + clearance_mm) / 1000.0)


def select_container(state: ContainerState, color: Optional[str]) -> Container:
    """The container find() picks (in zone, seen, stable, nearest); raises TargetValidationError otherwise."""
    if not state.calibrated:
        raise TargetValidationError("container finder is not calibrated (run find_containers.py --calibrate)")
    c = state.find(color)
    if c is None:
        want = normalize_color(color)
        others = [f"#{o.id} {o.color}" + ("" if o.in_zone else " out of zone") + ("" if o.seen else " not seen")
                  for o in state.containers if want is None or o.color == want]
        raise TargetValidationError(f"no usable {want or 'any'} container"
                                    + (f" (have: {', '.join(others)})" if others else " in view"))
    vals = [*c.center_mm[:2], c.height_mm]
    if not all(math.isfinite(float(v)) for v in vals):
        raise TargetValidationError(f"container #{c.id} has non-finite coordinates {vals}")
    return c


def move_above_container(arm, state: ContainerState, color: Optional[str],
                         clearance_mm: float = DEFAULT_CLEARANCE_MM) -> Tuple[Container, Vec3]:
    """Validate, IK-check, then move the tip clearance_mm above the container. Returns (container, xyz_m)."""
    c = select_container(state, color)
    xyz = hover_point_m(c, clearance_mm)
    check_reachability(xyz)
    arm.move_to_xyz(xyz)
    return c, xyz


def make_put_in_handler(get_state: Callable[[], ContainerState], arm,
                        clearance_mm: float = DEFAULT_CLEARANCE_MM):
    """Handler for nlp's PutIn action: hover above action.container's colour (no release yet)."""
    def handler(action):
        return move_above_container(arm, get_state(), getattr(action.container, "color", None), clearance_mm)
    return handler


# ---------------------------------------------------------------------------------------------- backends
class SimArm:
    """Kinematics Planner on K2's MockArm: home, then a straight validated line to the target."""

    def __init__(self):
        _get_kinematics()                  # puts Kinematics/ on sys.path
        import simulate
        from kinematics import fk
        self._sim, self._fk = simulate, fk
        self.last_tip: Optional[Vec3] = None

    def move_to_xyz(self, xyz: Vec3) -> None:
        try:
            recorder, planner = self._sim.run("goto", [tuple(xyz)])
        except Exception as exc:
            raise ArmMoveError(f"simulation rejected {tuple(round(v, 3) for v in xyz)} m: {exc}") from exc
        self.last_tip = tuple(self._fk(recorder.get_q()))
        print(f"sim: {planner.state.value}, tip at " + ", ".join(f"{v * 1000:.1f}" for v in self.last_tip) + " mm")

    def close(self) -> None:
        pass


def _send_ik():
    if str(_COMMS_DIR) not in sys.path:
        sys.path.insert(0, str(_COMMS_DIR))
    import send_ik
    return send_ik


def _fmt(angles) -> str:
    return ", ".join(f"{a:.1f}" for a in angles)


class DryArm:
    """No hardware: the same IK, servo mapping and firmware payload limits as UsbArm, printed instead of sent."""

    def __init__(self, gripper_deg: float = 90.0):
        self._ik, self.gripper_deg = _send_ik(), gripper_deg
        self._last_angles: Optional[list] = None

    def angles_for(self, xyz: Vec3) -> list:
        """Servo angles for xyz (m) at the current gripper angle; ArmMoveError if IK or the payload limits reject it."""
        try:
            angles = self._ik.angles_from_xyz(tuple(xyz), self.gripper_deg)
            self._ik.payload_for_angles(angles)
        except ValueError as exc:
            raise ArmMoveError(str(exc)) from exc
        return angles

    def check_xyz(self, xyz: Vec3) -> None:
        self.angles_for(xyz)

    def move_to_xyz(self, xyz: Vec3, settle: bool = True) -> None:
        self._last_angles = self.angles_for(xyz)
        print(f"dry: MOVE {' '.join(f'{v * 1000:.0f}' for v in xyz)} mm -> servos {_fmt(self._last_angles)}")

    def set_gripper(self, deg: float) -> None:
        if self._last_angles is None:
            raise ArmMoveError("move the arm before using the gripper")
        self._ik.payload_for_angles([*self._last_angles[:3], deg])
        self.gripper_deg, self._last_angles = deg, [*self._last_angles[:3], deg]
        print(f"dry: gripper -> {deg:.0f} deg")

    def close(self) -> None:
        pass


class UsbArm:
    """Real arm through firmware/comms/send_ik.py: one session, ARMed once, one MOVE per command.

    After each MOVE it waits as long as send_ik.send_waypoints does (firmware steps 1 degree / 30 ms, plus 0.75 s to
    settle; 0.05 s for an intermediate waypoint), keeping the heartbeat alive. That is commanded timing, not feedback;
    the first move assumes 90 degrees.
    """

    def __init__(self, port: str, baud: int = 115200, gripper_deg: float = 90.0, verbose: bool = False):
        import serial
        self._ik, self.gripper_deg = _send_ik(), gripper_deg
        self._ik.VERBOSE = verbose             # per-packet TX/ACK lines (4 heartbeats a second) only when asked
        self._previous = [90.0] * 4            # send_ik's assumed start-up command, not a measured pose
        self._last_angles: Optional[list] = None
        self._ser = serial.Serial(port, baud, timeout=0.02, write_timeout=0.15)
        self._session = self._ik.ArmLinkSession(self._ser)
        try:
            self._session.open()           # opening USB serial may reset the ESP32: waits for its HELLO
            self._session.arm()
        except Exception as exc:
            self._ser.close()
            raise ArmMoveError(f"could not open/arm the ArmLink session on {port}: {exc}") from exc

    angles_for = DryArm.angles_for
    check_xyz = DryArm.check_xyz

    def _send(self, angles, settle: bool) -> float:
        try:
            self._session.move(self._ik.payload_for_angles(angles))
            wait_s = max(abs(a - b) for a, b in zip(angles, self._previous)) * 0.030 + (0.75 if settle else 0.05)
            self._ik.wait_with_heartbeat(self._session, wait_s)
            self._previous = self._last_angles = list(angles)
        except self._ik.SessionError as exc:
            raise ArmMoveError(f"{exc} (restart to reconnect)") from exc
        except ValueError as exc:
            raise ArmMoveError(str(exc)) from exc
        return wait_s

    def move_to_xyz(self, xyz: Vec3, settle: bool = True) -> None:
        """settle=False: an intermediate waypoint of a path, so only a short wait after the MOVE."""
        angles = self.angles_for(xyz)
        wait_s = self._send(angles, settle)
        if settle:
            print(f"arm: MOVE done, servo targets {_fmt(angles)} (waited {wait_s:.1f} s)")

    def set_gripper(self, deg: float) -> None:
        """Same arm pose, new gripper angle."""
        if self._last_angles is None:
            raise ArmMoveError("move the arm before using the gripper")
        self._send([*self._last_angles[:3], deg], settle=True)
        self.gripper_deg = deg
        print(f"arm: gripper -> {deg:.0f} deg")

    def close(self) -> None:
        try:
            if self._session.usable:
                self._session.stop()
        except Exception as exc:
            print(f"arm: STOP failed: {exc}", file=sys.stderr)
        finally:
            self._session.close()
            self._ser.close()


def make_arm(port: Optional[str] = None, baud: int = 115200, gripper_deg: float = 90.0, dry: bool = False,
             verbose: bool = False):
    """UsbArm when a serial port is given, otherwise DryArm (dry=True) or the MockArm simulation."""
    if port:
        return UsbArm(port, baud, gripper_deg, verbose)
    return DryArm(gripper_deg) if dry else SimArm()
