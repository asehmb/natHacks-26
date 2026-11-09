"""Hand-over errand with the base locked: take an item from the person, drop it in a container, come back.

    errand = Errand(make_arm(port, dry=not port), lambda: latest_state)
    errand.ready()               # reach out to the person (hand-off pose), gripper open
    errand.grab()                # "grab this item": close the gripper on what they hand over
    errand.put_in(action)        # "put it in the blue container": action.container.color, duck-typed (no nlp import)
    errand.come_back()           # "come back to me": back to the hand-off pose, gripper still open

The base servo is broken, so every point is in the arm's vertical plane (y = 0): the arm only reaches forward/back and
up/down. A container has to sit on that line (|y| <= y_tol_mm). Paths are straight lines in step_mm steps, and every
waypoint of a step is checked (IK + firmware limits) before the first MOVE is sent.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .containers import ContainerState
from .container_move import select_container
from .planner_bridge import TargetValidationError

Vec3 = Tuple[float, float, float]


@dataclass
class ErrandConfig:
    handoff_mm: Tuple[float, float] = (320.0, 120.0)   # (x, z) tip pose where the person hands items over
    clearance_mm: float = 40.0                          # tip height above the rim when releasing
    y_tol_mm: float = 40.0                              # how far off the arm's line a container may be
    gripper_open: float = 120.0                         # servo degrees (placeholder: measure)
    gripper_closed: float = 60.0                        # servo degrees (placeholder: measure)
    step_mm: float = 10.0
    release_wait_s: float = 0.5
    find_timeout_s: float = 3.0
    tip_offset_mm: float = 40.0                         # the real tip lands this far forward of the commanded point


def line(a: Vec3, b: Vec3, step_mm: float) -> List[Vec3]:
    """Points from a (exclusive) to b (inclusive), at most step_mm apart (meters in, meters out)."""
    n = max(1, math.ceil(math.dist(a, b) * 1000.0 / step_mm))
    return [tuple(a[k] + (b[k] - a[k]) * i / n for k in range(3)) for i in range(1, n + 1)]


class Errand:
    def __init__(self, arm, get_state: Callable[[], ContainerState], cfg: Optional[ErrandConfig] = None,
                 log: Callable[[str], None] = print):
        self.arm, self.get_state, self.cfg, self.log = arm, get_state, cfg or ErrandConfig(), log
        self.pos: Optional[Vec3] = None     # last commanded tip position (m); None until ready()

    @property
    def handoff(self) -> Vec3:
        x, z = self.cfg.handoff_mm
        return (x / 1000.0, 0.0, z / 1000.0)

    # ------------------------------------------------------------------ steps
    def ready(self) -> None:
        """Jump to the hand-off pose (from an unknown pose) and open the gripper."""
        self._follow([self.handoff])
        self.arm.set_gripper(self.cfg.gripper_open)

    def grab(self, _action=None) -> None:
        if self.pos is None:
            self.ready()
        self.arm.set_gripper(self.cfg.gripper_closed)

    def release(self, _action=None) -> None:
        self._need_pose()
        self.arm.set_gripper(self.cfg.gripper_open)

    def put_in(self, action) -> Vec3:
        """Over the container at transit height, down to clearance_mm above the rim, open, lift. Returns the drop xyz."""
        self._need_pose()
        color = getattr(getattr(action, "container", None), "color", None)
        drop = self.drop_point(color)
        transit = max(self.pos[2], drop[2])
        above = (drop[0], 0.0, transit)
        path = self._path(self.pos, (self.pos[0], 0.0, transit), above, drop)
        self.log(f"moving over it and down to z={drop[2] * 1000:.0f} mm ({len(path)} steps)")
        self._follow(path)
        self.arm.set_gripper(self.cfg.gripper_open)
        time.sleep(self.cfg.release_wait_s)
        self._follow(line(drop, above, self.cfg.step_mm))
        return drop

    def come_back(self, _action=None) -> None:
        """Back to the hand-off pose, travelling at the higher of the two heights. The gripper is left as it is."""
        self._need_pose()
        h = self.handoff
        transit = max(self.pos[2], h[2])
        self.log(f"coming back to x={h[0] * 1000:.0f} z={h[2] * 1000:.0f} mm")
        self._follow(self._path(self.pos, (self.pos[0], 0.0, transit), (h[0], 0.0, transit), h))

    # ---------------------------------------------------------------- helpers
    def drop_point(self, color: Optional[str]) -> Vec3:
        """Release point (m) above the colour's container, on the arm's line. Waits up to find_timeout_s for it."""
        deadline = time.monotonic() + self.cfg.find_timeout_s
        waited = False
        while True:
            try:
                c = select_container(self.get_state(), color)
                break
            except TargetValidationError:
                if time.monotonic() >= deadline:
                    raise
                if not waited:
                    self.log(f"looking for the {color or 'nearest'} container...")
                    waited = True
                time.sleep(0.1)
        x, y = float(c.center_mm[0]), float(c.center_mm[1])
        self.log(f"found {c.color} container #{c.id} at x={x:.0f} y={y:.0f} mm, rim {float(c.height_mm):.0f} mm")
        if abs(y) > self.cfg.y_tol_mm:
            side = "left" if y > 0 else "right"
            raise TargetValidationError(
                f"{c.color} container #{c.id} is {abs(y):.0f} mm {side} of the arm's line (max {self.cfg.y_tol_mm:.0f}, "
                f"the base can't turn): slide it in front of the arm")
        return (x / 1000.0, 0.0, (float(c.height_mm) + self.cfg.clearance_mm) / 1000.0)

    def _path(self, start: Vec3, *corners: Vec3) -> List[Vec3]:
        pts, at = [], start
        for c in corners:
            if math.dist(at, c) > 1e-6:
                pts += line(at, c, self.cfg.step_mm)
                at = c
        return pts

    def _command(self, p: Vec3) -> Vec3:
        """Where to send the arm so the real tip ends up at p (tip_offset_mm forward along the arm's line)."""
        return (p[0] - self.cfg.tip_offset_mm / 1000.0, p[1], p[2])

    def _follow(self, points: List[Vec3]) -> None:
        """Check every point first; only then move, settling on the last one. Points (and pos) are real tip
        positions; the arm is sent the offset-corrected ones."""
        for p in points:
            self.arm.check_xyz(self._command(p))
        for i, p in enumerate(points):
            self.arm.move_to_xyz(self._command(p), settle=i == len(points) - 1)
            self.pos = p

    def _need_pose(self) -> None:
        if self.pos is None:
            raise TargetValidationError("the arm isn't at a known pose yet (run ready() first)")
