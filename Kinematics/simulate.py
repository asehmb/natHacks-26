"""Run the planner against K2's MockArm and watch it in 3D.

    py simulate.py                                   # default pick and place
    py simulate.py --goto 0.22 0.0 0.15              # move the tip to a chosen point (meters)
    py simulate.py --obj 0.24 0.02 0.12 --dest 0.24 -0.05 0.12
    py simulate.py --save run.gif --no-show          # write a GIF instead of opening a window

Positions are meters in the arm frame: origin on the base axis at table height,
x forward, y left, z up. Unreachable targets are rejected before the arm moves.

Needs K2's modules next to this file:
    mock_arm.py   (the MockArm class)   and   config.py   (load_config, get_joint_limits)
Edit JOINT_MAP below if the five mock joints are not ordered base, shoulder, elbow, wrist, gripper.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from kinematics import fk, joint_positions
from mock_adapter import JointMap, MockArmAdapter
from pick_place import SequenceError
from planner import Event, Planner, PlannerError, State

Vec3 = Tuple[float, float, float]

DEFAULT_OBJECT: Vec3 = (0.24, 0.02, 0.12)   # tip position where the gripper closes (meters)
DEFAULT_DEST: Vec3 = (0.24, -0.05, 0.12)    # tip position where the gripper opens (meters)

# !! CONFIRM WITH K2: which mock joints are base / shoulder / elbow / gripper, and their signs.
# gripper_open / gripper_close default (below) to the gripper joint's limit ends: a DEMO guess.
JOINT_MAP = JointMap(arm_indices=(0, 1, 2), signs=(1.0, 1.0, 1.0), offsets=(0.0, 0.0, 0.0),
                     gripper_index=4, gripper_open=None, gripper_close=None)


# ------------------------------------------------------------------- recording


@dataclass(frozen=True)
class Sample:
    t: float
    q: Vec3
    tip: Vec3
    label: str            # planner state at this moment
    holding: bool         # True between the gripper closing and opening
    gripper: Optional[float]


class TraceRecorder:
    """Wraps an ArmInterface and records a Sample after every wait()."""

    def __init__(self, arm: MockArmAdapter):
        self._arm = arm
        self.samples: List[Sample] = []
        self._label, self._holding = "idle", False

    def on_event(self, ev: Event) -> None:           # pass to Planner(on_event=...)
        self._label = ev.state.value
        if ev.message == "gripper close":
            self._holding = True
        elif ev.message == "gripper open":
            self._holding = False

    def wait(self, seconds: float) -> None:
        self._arm.wait(seconds)
        q = self._arm.get_q()
        self.samples.append(Sample(self._arm.now(), q, fk(q), self._label, self._holding,
                                   self._arm.gripper_value()))

    def __getattr__(self, name):                      # everything else goes straight to the arm
        return getattr(self._arm, name)


# --------------------------------------------------------------------- running


def build_arm(joint_map: JointMap = JOINT_MAP) -> MockArmAdapter:
    """K2's MockArm wrapped in the adapter. Fills in demo gripper values from the config limits."""
    try:
        from config import get_joint_limits, load_config
        from mock_arm import MockArm
    except ImportError as exc:
        sys.exit(f"Could not import K2's modules ({exc}). Put mock_arm.py and config.py next to "
                 "this file, or fix the import names at the top of build_arm().")
    config = load_config()
    mock = MockArm(config)
    g = joint_map.gripper_index
    if g is not None and (joint_map.gripper_open is None or joint_map.gripper_close is None):
        lo, hi = get_joint_limits(config)[g]
        joint_map = JointMap(joint_map.arm_indices, joint_map.signs, joint_map.offsets, g,
                             hi if joint_map.gripper_open is None else joint_map.gripper_open,
                             lo if joint_map.gripper_close is None else joint_map.gripper_close)
    return MockArmAdapter(mock, joint_map)


def run(task: str, targets: Sequence[Vec3], joint_map: JointMap = JOINT_MAP):
    """Run 'goto' (one target) or 'pick_place' (object, destination). Returns (recorder, planner)."""
    arm = build_arm(joint_map)
    recorder = TraceRecorder(arm)
    planner = Planner(recorder, on_event=recorder.on_event)
    planner.go_home()
    if task == "goto":
        planner.go_to(targets[0])
    else:
        planner.pick_and_place(targets[0], targets[1])
    return recorder, planner


def summarize(recorder: TraceRecorder, planner: Planner, task: str, targets: Sequence[Vec3]) -> str:
    lines = [f"final state: {planner.state.value}   simulated time: {recorder.now():.2f} s"]
    steps = [e for e in planner.events if e.index is not None and e.message not in
             ("gripper open", "gripper close")]
    lines.append(f"steps executed: {len(steps)}")
    final_tip = fk(recorder.get_q())
    if task == "goto":
        err = math.dist(final_tip, targets[0]) * 1000
        lines.append(f"final tip {tuple(round(v, 4) for v in final_tip)} m, error {err:.2f} mm from target")
    else:
        grasp = next((s for s in recorder.samples if s.label == "grasp"), None)
        place = next((s for s in recorder.samples if s.label == "place"), None)
        if grasp:
            lines.append(f"tip error at grasp: {math.dist(grasp.tip, targets[0]) * 1000:.2f} mm")
        if place:
            lines.append(f"tip error at place: {math.dist(place.tip, targets[1]) * 1000:.2f} mm")
    return "\n".join(lines)


# ------------------------------------------------------------------- animation


def animate(samples: Sequence[Sample], obj: Optional[Vec3], dest: Optional[Vec3], home: Vec3,
            save: Optional[str] = None, show: bool = True, fps: int = 25, speedup: float = 2.0):
    """3D animation of the arm. Block (red) is carried between grasp and place."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not samples:
        raise ValueError("no samples recorded")
    dt = samples[1].t - samples[0].t if len(samples) > 1 else 0.02
    stride = max(1, round(speedup / (fps * dt)))
    frames = list(range(0, len(samples), stride)) + [len(samples) - 1]

    # block position over time: at obj, then with the tip while held, then where it was released
    block, held_before, rest = [], False, obj
    for s in samples:
        if s.holding:
            held_before, rest = True, s.tip
            block.append(s.tip)
        else:
            block.append(rest if (held_before or obj is not None) else None)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(0, 0.4), ylim=(-0.25, 0.25), zlim=(0, 0.3), xlabel="x forward (m)",
           ylabel="y left (m)", zlabel="z up (m)")
    ax.set_box_aspect((0.4, 0.5, 0.3))
    ax.plot([0, 0.4, 0.4, 0, 0], [-0.25, -0.25, 0.25, 0.25, -0.25], [0] * 5, color="0.7", lw=1)  # table
    if obj:
        ax.scatter(*obj, marker="x", color="red", s=60, label="object")
    if dest:
        ax.scatter(*dest, marker="x", color="green", s=60, label="destination")
    ax.scatter(*home, marker="+", color="gray", s=60, label="home")
    (arm_line,) = ax.plot([], [], [], "-o", lw=4, color="tab:blue")
    (trail,) = ax.plot([], [], [], "-", lw=1, color="tab:orange")
    (blk,) = ax.plot([], [], [], "s", ms=9, color="red")
    (tip_pt,) = ax.plot([], [], [], "o", ms=8, color="black")
    title = ax.set_title("")
    ax.legend(loc="upper left", fontsize=8)

    def draw(frame_no):
        i = frames[frame_no]
        pts = joint_positions(samples[i].q)
        arm_line.set_data_3d([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        tip_path = [s.tip for s in samples[: i + 1 : max(1, stride)]]
        trail.set_data_3d([p[0] for p in tip_path], [p[1] for p in tip_path], [p[2] for p in tip_path])
        if block[i] is not None:
            blk.set_data_3d([block[i][0]], [block[i][1]], [block[i][2]])
        tip_pt.set_data_3d([pts[-1][0]], [pts[-1][1]], [pts[-1][2]])
        tip_pt.set_color("tab:red" if samples[i].holding else "black")
        title.set_text(f"t = {samples[i].t:5.2f} s   state: {samples[i].label}")
        return arm_line, trail, blk, tip_pt, title

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    if save:
        anim.save(save, writer="pillow", fps=fps)
        print(f"saved {save}")
    if show:
        plt.show()
    return anim


# ------------------------------------------------------------------------ CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Simulate the arm on K2's MockArm.")
    ap.add_argument("--goto", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="move the tip to this point (meters) instead of pick and place")
    ap.add_argument("--obj", nargs=3, type=float, metavar=("X", "Y", "Z"), default=DEFAULT_OBJECT)
    ap.add_argument("--dest", nargs=3, type=float, metavar=("X", "Y", "Z"), default=DEFAULT_DEST)
    ap.add_argument("--save", metavar="FILE.gif", help="save the animation as a GIF")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--no-anim", action="store_true", help="print the log only")
    args = ap.parse_args(argv)

    if args.goto:
        task, targets = "goto", [tuple(args.goto)]
    else:
        task, targets = "pick_place", [tuple(args.obj), tuple(args.dest)]

    try:
        recorder, planner = run(task, targets)
    except (SequenceError, PlannerError, ValueError) as exc:
        print(f"REJECTED: {exc}")
        print("The arm did not execute the sequence.")
        return 1

    for ev in planner.events:
        if ev.index is None or ev.message.startswith("gripper") or ev.index == 0 or ev.state in (State.GRASP, State.PLACE):
            where = f" step {ev.index + 1}/{ev.total}" if ev.index is not None else ""
            print(f"t={ev.t:6.2f}s  {ev.state.value:<9}{where}  {ev.message}")
    print(summarize(recorder, planner, task, targets))

    if not args.no_anim and (args.save or not args.no_show):
        obj, dest = (None, None) if task == "goto" else (targets[0], targets[1])
        animate(recorder.samples, obj, dest, planner.pp_cfg.home, save=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
