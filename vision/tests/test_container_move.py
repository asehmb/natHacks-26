import math
from types import SimpleNamespace

import numpy as np
import pytest

from armvision import container_move as cm
from armvision.containers import Container, ContainerState
from armvision.planner_bridge import TargetValidationError

REACHABLE = (250.0, 0.0)        # mm; with a 60 mm tall container + 20 mm the tip is at z = 80 mm (inside the IK envelope)


def box(color="red", xy=REACHABLE, h=60.0, in_zone=True, seen=True, id=1):
    return Container(id, color, np.array(xy, float), h, (80.0, 80.0), "round", 0.0, "config", in_zone, 5, seen)


class FakeArm:
    def __init__(self):
        self.moves = []

    def move_to_xyz(self, xyz):
        self.moves.append(xyz)


def test_hover_point_is_clearance_above_rim_in_meters():
    assert cm.hover_point_m(box(xy=(250, -30), h=36.5)) == pytest.approx((0.25, -0.03, 0.0565))
    assert cm.hover_point_m(box(h=36.5), clearance_mm=40)[2] == pytest.approx(0.0765)


def test_moves_above_matching_container():
    arm = FakeArm()
    state = ContainerState([box("blue", id=1), box("red", id=2)])
    c, xyz = cm.move_above_container(arm, state, "red")
    assert c.id == 2
    assert arm.moves == [xyz] and xyz == pytest.approx((0.25, 0.0, 0.08))


def test_color_alias_is_normalized():
    arm = FakeArm()
    c, _ = cm.move_above_container(arm, ContainerState([box("purple")]), "Violet")
    assert c.color == "purple" and len(arm.moves) == 1


@pytest.mark.parametrize("state", [
    ContainerState([]),
    ContainerState([box("blue")]),
    ContainerState([box("red", in_zone=False)]),
    ContainerState([box("red", seen=False)]),
    ContainerState([box("red")], calibrated=False),
])
def test_no_usable_container_never_moves(state):
    arm = FakeArm()
    with pytest.raises(TargetValidationError):
        cm.move_above_container(arm, state, "red")
    assert arm.moves == []


def test_unreachable_container_never_moves():
    arm = FakeArm()
    with pytest.raises(TargetValidationError, match="unreachable"):
        cm.move_above_container(arm, ContainerState([box(xy=(600, 0))]), "red")
    assert arm.moves == []


def test_put_in_handler_uses_container_color():
    arm, state = FakeArm(), ContainerState([box("blue", id=7), box("red", id=8)])
    handler = cm.make_put_in_handler(lambda: state, arm)
    c, _ = handler(SimpleNamespace(container=SimpleNamespace(color="blue", noun="bin")))
    assert c.id == 7 and len(arm.moves) == 1


def test_sim_arm_reaches_hover_point():
    try:
        arm = cm.SimArm()
    except ImportError as exc:
        pytest.skip(f"Kinematics not importable: {exc}")
    _, xyz = cm.move_above_container(arm, ContainerState([box()]), "red")
    assert math.dist(arm.last_tip, xyz) < 0.005
