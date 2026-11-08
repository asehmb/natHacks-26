"""Planner state machine: turns a goal into validated steps, then drives an ArmInterface.

    planner = Planner(arm)                       # arm implements mock_adapter.ArmInterface
    planner.go_home()                            # known starting pose
    planner.go_to((0.22, 0.0, 0.15))             # straight line to a chosen point
    planner.pick_and_place(obj_xyz, dest_xyz)    # full pick and place (meters)

Safety rules built in:
  * The whole sequence is validated and every pose is IK-solved BEFORE the arm moves.
    One unreachable pose rejects everything (SequenceError / PlannerError), nothing is sent.
  * Motion is streamed as plan_move() samples (trapezoidal profile, all joints finish together)
    and limited to the arm's own speed caps, so the arm never lags behind the planned line.
  * Any failure (unreachable, arm rejection, timeout, stop) ends in State.ERROR and re-raises.
    The firmware limits and the physical kill switch remain the real safety.

Every state change is reported as an Event (also passed to on_event), which is what the
dashboard shows as "current step".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

from kinematic_model import plan_move
from kinematics import fk, ik
from mock_adapter import ArmInterface
from pick_place import (
    DEFAULT_CONFIG,
    PickPlaceConfig,
    SequenceError,
    Step,
    move_sequence,
    pick_and_place_sequence,
    validate_sequence,
)

Vec3 = Tuple[float, float, float]


class PlannerError(RuntimeError):
    """The planner could not finish the command (unreachable, timeout, stopped, ...)."""


class State(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    HOMING = "homing"
    MOVE = "move"
    HOVER = "hover"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    RETRACT = "retract"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    t: float                       # arm time (seconds)
    state: State
    index: Optional[int] = None    # step index within the sequence, if any
    total: Optional[int] = None    # number of steps in the sequence
    message: str = ""
    xyz: Optional[Vec3] = None     # the step's target pose (meters)


@dataclass(frozen=True)
class PlannerConfig:
    """Motion constants. Move into arm_config.yaml once K2's loader exists."""

    dt: float = 0.02               # seconds between streamed samples
    vmax: float = 1.0              # rad/s at speed = 1 (also limited by the arm's own caps)
    amax: float = 4.0              # rad/s^2 at speed = 1
    arrive_tol: float = 0.005      # rad: how close a joint must be to count as arrived
    arrive_timeout_s: float = 10.0  # give up waiting for the arm after this long


def _state_for(step_name: str) -> State:
    try:
        return State[step_name.upper()]
    except KeyError:
        raise PlannerError(f"unknown step name {step_name!r}") from None


class Planner:
    def __init__(self, arm: ArmInterface, cfg: PlannerConfig = PlannerConfig(),
                 pp_cfg: PickPlaceConfig = DEFAULT_CONFIG,
                 on_event: Optional[Callable[[Event], None]] = None):
        if not (cfg.dt > 0 and cfg.vmax > 0 and cfg.amax > 0 and cfg.arrive_tol > 0):
            raise ValueError("PlannerConfig dt, vmax, amax and arrive_tol must be > 0")
        self.arm = arm
        self.cfg = cfg
        self.pp_cfg = pp_cfg
        self._on_event = on_event
        self.state = State.IDLE
        self.error: Optional[str] = None
        self.events: List[Event] = []
        self._stop = False
        self._index: Optional[int] = None
        self._total: Optional[int] = None

    # ------------------------------------------------------------------ events

    def _emit(self, state: State, index=None, total=None, message="", xyz=None) -> None:
        self.state = state
        event = Event(self.arm.now(), state, index, total, message, xyz)
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)

    def _command(self, name: str, action: Callable[[], None]) -> None:
        """Run one top-level command with uniform state / error handling."""
        self._stop, self.error, self._index, self._total = False, None, None, None
        self._emit(State.PLANNING, message=name)
        try:
            action()
        except Exception as exc:  # any failure leaves the planner in ERROR, then re-raises
            self.error = str(exc)
            self._emit(State.ERROR, self._index, self._total, str(exc))
            raise
        self._emit(State.DONE, message=name)

    def stop(self) -> None:
        """Ask the running command to stop at the next sample (ends in ERROR 'stopped')."""
        self._stop = True

    def _check_stop(self) -> None:
        if self._stop:
            raise PlannerError("stopped on request")

    # ----------------------------------------------------------------- helpers

    def _current_q(self) -> Vec3:
        return tuple(float(v) for v in self.arm.get_q())  # type: ignore[return-value]

    def _speed_limits(self, speed: float) -> List[float]:
        caps = self.arm.speed_limits()
        base = [self.cfg.vmax] * 3 if caps is None else [min(self.cfg.vmax, c) for c in caps]
        return [v * speed for v in base]

    def _pose_is_known(self) -> bool:
        """True if the arm's current joints are an IK-consistent pose (so lines start correctly)."""
        q = self._current_q()
        sol = ik(*fk(q))
        return sol is not None and max(abs(a - b) for a, b in zip(sol, q)) < 1e-3

    def _near(self, xyz: Vec3, tol_m: float = 0.005) -> bool:
        return self._pose_is_known() and math.dist(fk(self._current_q()), xyz) <= tol_m

    def _move_joint(self, q_goal: Vec3, speed: float) -> None:
        """Stream a trapezoidal joint-space move to q_goal, then wait until the arm arrives."""
        q_now = self._current_q()
        traj = plan_move(q_now, q_goal, self._speed_limits(speed), self.cfg.amax * speed, self.cfg.dt)
        for q in traj[1:]:
            self._check_stop()
            self.arm.move_to(q)
            self.arm.wait(self.cfg.dt)
        self._settle(q_goal)

    def _settle(self, q_goal: Vec3) -> None:
        waited = 0.0
        while True:
            err = max(abs(a - b) for a, b in zip(self._current_q(), q_goal))
            if err <= self.cfg.arrive_tol:
                return
            if waited > self.cfg.arrive_timeout_s:
                raise PlannerError(f"arm did not reach the target within {self.cfg.arrive_timeout_s} s "
                                   f"(still {err:.3f} rad away)")
            self._check_stop()
            self.arm.wait(self.cfg.dt)
            waited += self.cfg.dt

    def _run_steps(self, steps: Sequence[Step]) -> None:
        result = validate_sequence(steps, self.pp_cfg)
        if not result:
            raise SequenceError(result)
        targets = []
        for i, s in enumerate(steps):            # solve EVERYTHING before moving anything
            q = ik(*s.xyz)
            if q is None:
                raise PlannerError(f"step {i} ({s.name}): IK has no solution for {s.xyz}")
            targets.append(q)

        total = len(steps)
        for i, (s, q) in enumerate(zip(steps, targets)):
            self._index, self._total = i, total
            self._emit(_state_for(s.name), i, total, s.name, s.xyz)
            self._move_joint(q, s.speed)
            if s.gripper is not None:
                self.arm.set_gripper(s.gripper)
                self._emit(self.state, i, total, f"gripper {s.gripper}", s.xyz)
                if hasattr(self.arm, "wait_for_gripper"):
                    try:
                        self.arm.wait_for_gripper(self.cfg.arrive_timeout_s, check_stop=self._check_stop)
                    except TypeError:
                        self.arm.wait_for_gripper(self.cfg.arrive_timeout_s)
                elif hasattr(self.arm, "gripper_travel_time"):
                    travel = self.arm.gripper_travel_time(s.gripper)
                    if travel > self.cfg.arrive_timeout_s:
                        raise PlannerError(
                            f"gripper travel time {travel:.2f} s exceeds timeout {self.cfg.arrive_timeout_s} s"
                        )
                    step = getattr(self.arm, "_tick", self.cfg.dt)
                    waited = 0.0
                    while waited < travel - 1e-12:
                        self._check_stop()
                        dt_step = min(step, travel - waited)
                        self.arm.wait(dt_step)
                        waited += dt_step
            if s.dwell_s > 0:
                self.arm.wait(s.dwell_s)

    def _go_home(self) -> None:
        q_home = ik(*self.pp_cfg.home)
        if q_home is None:
            raise PlannerError(f"home pose {self.pp_cfg.home} is unreachable")
        self._emit(State.HOMING, message="home", xyz=self.pp_cfg.home)
        self._move_joint(q_home, self.pp_cfg.travel_speed)

    # ------------------------------------------------------------ public commands

    def go_home(self) -> None:
        """Joint-space move to the home pose from wherever the arm is (use at start-up)."""
        self._command("go_home", self._go_home)

    def go_to(self, xyz: Sequence[float]) -> None:
        """Straight-line move of the tip to a chosen point (meters)."""
        def action():
            steps = move_sequence(xyz, self.pp_cfg, from_xyz=None)   # validate before moving
            if not self._pose_is_known():
                self._go_home()
            steps = move_sequence(xyz, self.pp_cfg, from_xyz=fk(self._current_q()))
            self._run_steps(steps)
        self._command("go_to", action)

    def plan_pick_and_place(self, object_xyz: Sequence[float], dest_xyz: Sequence[float]) -> List[Step]:
        """Build and validate the full sequence without moving the arm."""
        return pick_and_place_sequence(object_xyz, dest_xyz, self.pp_cfg)

    def pick_and_place(self, object_xyz: Sequence[float], dest_xyz: Sequence[float]) -> None:
        """Pick at object_xyz and place at dest_xyz (tip positions, meters)."""
        def action():
            steps = self.plan_pick_and_place(object_xyz, dest_xyz)   # rejects before any motion
            if not self._near(self.pp_cfg.home):
                self._go_home()
            self._run_steps(steps)
        self._command("pick_and_place", action)

    def execute(self, steps: Sequence[Step]) -> None:
        """Validate and run an arbitrary step list (it must start from the arm's current pose)."""
        self._command("execute", lambda: self._run_steps(steps))
