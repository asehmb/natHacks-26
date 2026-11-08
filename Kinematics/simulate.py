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

from kinematics import fk
from mock_adapter import ArmError, JointMap, MockArmAdapter
from pick_place import DEFAULT_CONFIG, SequenceError
from planner import Event, Planner, PlannerError, State

try:
    from kinematics import joint_positions
except ImportError:  # kinematics.py does not provide it: draw the links from the same geometry as fk()
    def joint_positions(q: Sequence[float]):
        """Base, shoulder, elbow, wrist, tip positions (meters) for joints q (rad)."""
        from kinematic_model import EEZYbotARM_Mk2 as _M
        l1, l2, l3, l4 = (v / 1000.0 for v in (_M.L1, _M.L2, _M.L3, _M.L4))
        q1, q2, q3 = q
        c1, s1 = math.cos(q1), math.sin(q1)
        r_elbow = l2 * math.cos(q2)
        z_elbow = l1 + l2 * math.sin(q2)
        r_wrist = r_elbow + l3 * math.cos(q2 + q3)
        z_wrist = z_elbow + l3 * math.sin(q2 + q3)
        return [(0.0, 0.0, 0.0), (0.0, 0.0, l1),
                (r_elbow * c1, r_elbow * s1, z_elbow),
                (r_wrist * c1, r_wrist * s1, z_wrist),
                (r_wrist * c1 + l4 * c1, r_wrist * s1 + l4 * s1, z_wrist)]

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


_AXIS_NAMES = ("q1 base", "q2 shoulder", "q3 elbow")
_TOL = 1e-6


def task_q_ranges(task: str, targets: Sequence[Vec3]) -> List[Tuple[float, float]]:
    """(min, max) of each of OUR joint angles (rad) over every pose this task visits, home included.

    Building the sequence also rejects unreachable targets (SequenceError) before anything runs.
    """
    from kinematics import ik
    from pick_place import move_sequence, pick_and_place_sequence

    cfg = DEFAULT_CONFIG
    if task == "goto":
        steps = move_sequence(targets[0], cfg, from_xyz=cfg.home)
    else:
        steps = pick_and_place_sequence(targets[0], targets[1], cfg)
    poses = [cfg.home] + [s.xyz for s in steps]
    solutions = []
    for p in poses:
        q = ik(*p)
        if q is None:
            raise PlannerError(f"IK has no solution for {p}")
        solutions.append(q)
    return [(min(q[k] for q in solutions), max(q[k] for q in solutions)) for k in range(3)]


def fit_axis(need: Tuple[float, float], limit: Tuple[float, float]) -> Tuple[float, float, bool]:
    """Pick (sign, offset, shifted) so that sign * q + offset keeps [need] inside [limit].

    Tries the plain mapping, then a flipped sign, then (keeping sign +1) shifts the needed range to the
    middle of the limit range. Raises ValueError if the limit range is too narrow for the motion.
    """
    (a, b), (lo, hi) = need, limit
    for sign in (1.0, -1.0):
        m0, m1 = sorted((sign * a, sign * b))
        if m0 >= lo - _TOL and m1 <= hi + _TOL:
            return sign, 0.0, False
    if b - a > hi - lo + _TOL:
        raise ValueError(f"motion spans {b - a:.3f} rad but the configured limits only span {hi - lo:.3f} rad")
    return 1.0, (lo + hi) / 2.0 - (a + b) / 2.0, True


def fit_joint_map(joint_map: JointMap, limits, need: Sequence[Tuple[float, float]]) -> JointMap:
    """Return joint_map with signs / offsets chosen so the task fits inside the config's joint limits.

    The simulator only uses this mapping for the mock's limit checks, so any consistent mapping that fits
    gives the same animation. For real hardware the mapping must come from the calibration, not from here.
    """
    signs, offsets, notes, shifted = [], [], [], False
    for k, idx in enumerate(joint_map.arm_indices):
        lo, hi = float(limits[idx][0]), float(limits[idx][1])
        try:
            sign, offset, moved = fit_axis(need[k], (lo, hi))
        except ValueError as exc:
            raise ValueError(f"{_AXIS_NAMES[k]} -> mock joint {idx}: {exc}. Needed range "
                             f"[{need[k][0]:.3f}, {need[k][1]:.3f}] rad, config range [{lo:.3f}, {hi:.3f}] rad. "
                             "Check joints[] min/max in arm_config.yaml.") from exc
        signs.append(sign)
        offsets.append(offset)
        shifted = shifted or moved
        notes.append(f"  {_AXIS_NAMES[k]:<12} -> mock joint {idx}: needs [{need[k][0]:+.3f}, {need[k][1]:+.3f}] rad, "
                     f"config allows [{lo:+.3f}, {hi:+.3f}] rad -> sign {sign:+.0f}, offset {offset:+.3f}")
    print("Joint map for the simulation:" + ("  (shifted to fit the config limits)" if shifted else ""))
    print("\n".join(notes))
    if shifted:
        print("  NOTE: the config's elbow/shoulder zero differs from kinematic_model's angles. Fine for the\n"
              "  simulation; confirm the real zero and sign with K2 before driving hardware.")
    return JointMap(joint_map.arm_indices, tuple(signs), tuple(offsets), joint_map.gripper_index,
                    joint_map.gripper_open, joint_map.gripper_close)


def build_arm(joint_map: JointMap = JOINT_MAP,
              need: Optional[Sequence[Tuple[float, float]]] = None) -> MockArmAdapter:
    """K2's MockArm wrapped in the adapter. Fills in demo gripper values from the config limits.

    need: (min, max) of q1, q2, q3 the task will use. If given, the joint map is fitted to the config limits.
    """
    try:
        from config import get_joint_limits, load_config
        from mock_arm import MockArm
    except ImportError as exc:
        sys.exit(f"Could not import K2's modules ({exc}). Put mock_arm.py and config.py next to "
                 "this file, or fix the import names at the top of build_arm().")
    config = load_config()
    mock = MockArm(config)
    limits = get_joint_limits(config)
    if need is not None:
        joint_map = fit_joint_map(joint_map, limits, need)
    g = joint_map.gripper_index
    if g is not None and (joint_map.gripper_open is None or joint_map.gripper_close is None):
        lo, hi = limits[g]
        joint_map = JointMap(joint_map.arm_indices, joint_map.signs, joint_map.offsets, g,
                             hi if joint_map.gripper_open is None else joint_map.gripper_open,
                             lo if joint_map.gripper_close is None else joint_map.gripper_close)
    return MockArmAdapter(mock, joint_map)


def run(task: str, targets: Sequence[Vec3], joint_map: JointMap = JOINT_MAP):
    """Run 'goto' (one target) or 'pick_place' (object, destination). Returns (recorder, planner)."""
    arm = build_arm(joint_map, task_q_ranges(task, targets))
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
    except (SequenceError, PlannerError, ArmError, ValueError) as exc:
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