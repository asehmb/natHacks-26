import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from armvision.containers import Container, ContainerState
from armvision.errand import Errand, ErrandConfig, line
from armvision.planner_bridge import TargetValidationError

FAST = dict(release_wait_s=0.0, find_timeout_s=0.0, tip_offset_mm=0.0)


def box(color="blue", xy=(250.0, 0.0), h=36.5, in_zone=True, seen=True, id=1):
    return Container(id, color, np.array(xy, float), h, (80.0, 80.0), "round", 0.0, "config", in_zone, 5, seen)


class FakeArm:
    """Records calls; check_xyz rejects any point in `bad` (meters, rounded to the mm)."""

    def __init__(self, bad=()):
        self.calls, self.bad = [], {tuple(round(v * 1000) for v in p) for p in bad}

    def check_xyz(self, xyz):
        if tuple(round(v * 1000) for v in xyz) in self.bad:
            raise ValueError(f"unreachable {xyz}")

    def move_to_xyz(self, xyz, settle=True):
        self.calls.append(("move", tuple(xyz), settle))

    def set_gripper(self, deg):
        self.calls.append(("grip", deg))

    def moves(self):
        return [c[1] for c in self.calls if c[0] == "move"]


def put(color):
    return SimpleNamespace(container=SimpleNamespace(color=color))


def errand(arm, *containers, **cfg):
    return Errand(arm, lambda: ContainerState(list(containers)), ErrandConfig(**{**FAST, **cfg}))


def test_line_steps_and_ends_exactly():
    pts = line((0.32, 0, 0.12), (0.25, 0, 0.12), 10)
    assert len(pts) == 7 and pts[-1] == pytest.approx((0.25, 0, 0.12))


def test_full_errand_is_planar_and_returns_to_handoff():
    arm = FakeArm()
    e = errand(arm, box(h=36.5))
    e.ready()
    e.grab()
    drop = e.put_in(put("blue"))
    e.come_back()
    grips = [c[1] for c in arm.calls if c[0] == "grip"]
    assert grips == [120.0, 60.0, 120.0]                         # open at hand-off, close on the item, open over it
    moves = arm.moves()
    assert all(p[1] == 0.0 for p in moves)                       # the base can't turn
    assert drop == pytest.approx((0.25, 0.0, 0.0765))            # 40 mm above a 36.5 mm rim
    assert min(p[2] for p in moves) == pytest.approx(0.0765)
    assert moves[-1] == pytest.approx((0.32, 0.0, 0.12)) and e.pos == moves[-1]
    i_close, i_open = arm.calls.index(("grip", 60.0)), len(arm.calls) - 1 - arm.calls[::-1].index(("grip", 120.0))
    assert arm.calls[i_open - 1][1] == pytest.approx(drop)      # released at the drop point, not on the way
    assert all(c[0] == "move" for c in arm.calls[i_close + 1:i_open])
    # every step of a path is a small one; only the last point of each path settles
    assert max(np.linalg.norm(np.subtract(a, b)) for a, b in zip(moves, moves[1:])) <= 0.0101
    settles = [c for c in arm.calls if c[0] == "move" and c[2]]
    assert len(settles) == 4                                     # ready, down to the drop, lift, back to hand-off


def test_container_lower_than_transit_never_dips_before_it_is_above():
    arm = FakeArm()
    e = errand(arm, box(xy=(220, 10)))
    e.ready()
    e.put_in(put("blue"))
    descent = [p for p in arm.moves() if p[2] < 0.12 - 1e-9]
    assert descent and all(p[0] == pytest.approx(0.22) for p in descent)


@pytest.mark.parametrize("containers,match", [
    ((box(xy=(250, 80)),), "left of the arm's line"),
    ((box(xy=(250, -60)),), "right of the arm's line"),
    ((box("orange"),), "no usable blue"),
    ((box(in_zone=False),), "out of zone"),
])
def test_unusable_container_never_moves(containers, match):
    arm = FakeArm()
    e = errand(arm, *containers)
    e.ready()
    before = list(arm.calls)
    with pytest.raises(TargetValidationError, match=match):
        e.put_in(put("blue"))
    assert arm.calls == before


def test_unreachable_waypoint_mid_path_stops_before_any_move():
    arm = FakeArm(bad=[(0.28, 0.0, 0.12)])
    e = errand(arm, box())
    e.ready()
    before = list(arm.calls)
    with pytest.raises(ValueError, match="unreachable"):
        e.put_in(put("blue"))
    assert arm.calls == before


def test_waits_for_the_container_to_appear():
    arm, frames = FakeArm(), iter([ContainerState([]), ContainerState([]), ContainerState([box()])])
    e = Errand(arm, lambda: next(frames), ErrandConfig(release_wait_s=0.0, find_timeout_s=2.0))
    e.ready()
    assert e.put_in(put("blue"))[0] == pytest.approx(0.25)


def test_put_in_before_ready_is_refused():
    with pytest.raises(TargetValidationError, match="known pose"):
        errand(FakeArm(), box()).put_in(put("blue"))


def _nlp():
    nlp_dir = Path(__file__).resolve().parents[2] / "nlp"
    if str(nlp_dir) not in sys.path:
        sys.path.append(str(nlp_dir))
    return pytest.importorskip("nlp")


def test_sentence_to_arm_calls_end_to_end():
    _nlp()
    import talk_to_arm
    from nlp import Processor

    arm = FakeArm()
    e = errand(arm, box(xy=(260, 5)), box("orange", xy=(220, 0), id=2))
    e.ready()
    disp = talk_to_arm.build_dispatcher(e, lambda msg: None)
    result = Processor().process("grab this item, put it in the bluw container, then come back to me")
    assert [a.name for a in result.actions] == ["pick_up", "put_in", "return_to_user"]
    disp.dispatch(result.actions)
    grips = [c[1] for c in arm.calls if c[0] == "grip"]
    assert grips == [120.0, 60.0, 120.0]
    moves = arm.moves()
    lowest = min(moves, key=lambda p: p[2])
    assert lowest == pytest.approx((0.26, 0.0, 0.0765))          # the blue one, not the nearer orange one
    assert moves[-1] == pytest.approx((0.32, 0.0, 0.12))


def test_unsupported_command_moves_nothing():
    _nlp()
    import talk_to_arm
    from nlp import Processor

    arm = FakeArm()
    e = errand(arm, box())
    e.ready()
    before = list(arm.calls)
    with pytest.raises(KeyError):
        talk_to_arm.build_dispatcher(e, lambda m: None).dispatch(Processor().process("grab it and rotate joint 0 by 30").actions)
    assert arm.calls == before


@pytest.mark.real_ik
@pytest.mark.parametrize("x", [200.0, 250.0, 300.0])
def test_dry_arm_accepts_the_default_path_within_firmware_limits(x):
    pytest.importorskip("yaml")
    from armvision.container_move import DryArm

    try:
        arm = DryArm(120.0)
        arm.check_xyz((0.32, 0.0, 0.12))
    except ImportError as exc:
        pytest.skip(f"Kinematics does not import: {exc}")
    e = Errand(arm, lambda: ContainerState([box(xy=(x, 0))]), ErrandConfig(**FAST))
    e.ready()
    e.grab()
    e.put_in(put("blue"))
    e.come_back()
    assert e.pos == pytest.approx((0.32, 0.0, 0.12))


@pytest.mark.real_ik
def test_usb_arm_runs_the_errand_over_a_fake_session(monkeypatch):
    serial = pytest.importorskip("serial")
    pytest.importorskip("yaml")
    from armvision import container_move as cm

    send_ik = cm._send_ik()
    sent = []

    class FakeSession:
        usable = True

        def __init__(self, ser): pass
        def open(self): pass
        def arm(self): pass
        def move(self, payload): sent.append(payload)
        def stop(self): sent.append("STOP")
        def close(self): pass
        def check(self): pass

    class FakeSerial:
        def __init__(self, *a, **k): pass
        def close(self): pass

    monkeypatch.setattr(serial, "Serial", FakeSerial)
    monkeypatch.setattr(send_ik, "ArmLinkSession", FakeSession)
    monkeypatch.setattr(send_ik, "wait_with_heartbeat", lambda session, s: None)
    arm = cm.make_arm("/dev/fake", gripper_deg=120.0)
    e = Errand(arm, lambda: ContainerState([box()]), ErrandConfig(**FAST))
    e.ready()
    e.grab()
    e.put_in(put("blue"))
    e.come_back()
    arm.close()
    assert sent[-1] == "STOP" and len(sent) > 20
    import struct
    first, last = struct.unpack("<4h", sent[0]), struct.unpack("<4h", sent[-2])
    assert first == last and first[3] == 12000                # back at the hand-off pose, gripper open


def test_tip_offset_commands_points_back_but_tracks_the_real_tip():
    arm = FakeArm()
    e = errand(arm, box(xy=(250, 0)), tip_offset_mm=40.0)
    e.ready()
    drop = e.put_in(put("blue"))
    e.come_back()
    moves = arm.moves()
    assert moves[0] == pytest.approx((0.28, 0.0, 0.12))          # hand-off 320 mm, sent 40 mm short
    assert min(moves, key=lambda p: p[2]) == pytest.approx((0.21, 0.0, 0.0765))
    assert drop == pytest.approx((0.25, 0.0, 0.0765))            # reported positions are the real tip's
    assert e.pos == pytest.approx((0.32, 0.0, 0.12))


def test_tip_offset_is_also_applied_to_the_pre_check():
    arm = FakeArm(bad=[(0.24, 0.0, 0.12)])                       # commanded point for a real tip at 280 mm
    e = errand(arm, box(xy=(250, 0)), tip_offset_mm=40.0)
    e.ready()
    before = list(arm.calls)
    with pytest.raises(ValueError):
        e.put_in(put("blue"))
    assert arm.calls == before


@pytest.mark.real_ik
@pytest.mark.parametrize("x", [200.0, 250.0, 300.0])
def test_default_tip_offset_path_is_reachable(x):
    pytest.importorskip("yaml")
    from armvision.container_move import DryArm

    e = Errand(DryArm(120.0), lambda: ContainerState([box(xy=(x, 0))]), ErrandConfig(release_wait_s=0.0))
    e.ready()
    e.grab()
    e.put_in(put("blue"))
    e.come_back()
