import dataclasses

import pytest

from nlp import (
    CloseHand, Give, Home, MoveRelative, ObjectRef, OpenHand, PickUp, PutDown, PutIn, Reply,
    ReturnToZero, RotateJoint, Shake,
)
from nlp.actions import Action, action_from_dict, actions_from_dicts

CUP = ObjectRef("cup", "red", "big", "left")

ALL = [
    PickUp(CUP), PutDown(), PutDown(CUP), PutIn(CUP), PutIn(CUP, CUP, "on"), Give(), Give(CUP),
    MoveRelative("up"), MoveRelative("left", "large", 12.5), Home(), ReturnToZero(), RotateJoint(0),
    RotateJoint(3, -45.0, False), OpenHand(), CloseHand(), Shake("maybe"), Reply("hi"),
    Reply("hi", Shake("no")),
]


@pytest.mark.parametrize("action", ALL, ids=lambda a: type(a).__name__)
def test_round_trip_through_json(action):
    import json

    d = json.loads(json.dumps(action.to_dict()))
    assert action_from_dict(d) == action


def test_every_action_class_is_covered_and_named_uniquely():
    from nlp import actions

    classes = [c for c in vars(actions).values() if isinstance(c, type) and issubclass(c, Action) and c is not Action]
    assert {type(a) for a in ALL} == set(classes)
    assert len({c.name for c in classes}) == len(classes)


def test_actions_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Home().__setattr__("x", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        CUP.noun = "pen"


@pytest.mark.parametrize("bad", [
    {"action": "move_relative", "direction": "sideways"},
    {"action": "move_relative", "direction": "up", "amount": "huge"},
    {"action": "rotate_joint", "joint": "3"},
    {"action": "rotate_joint", "joint": -2},
    {"action": "rotate_joint"},
    {"action": "shake", "kind": "YES"},
    {"action": "pick_up"},
    {"action": "pick_up", "target": {"noun": "cup", "weight": 3}},
    {"action": "home", "extra": 1},
    {"action": "reply", "text": "x", "gesture": {"action": "home"}},
    {"action": ""},
    {},
    "home",
    None,
])
def test_invalid_action_data_is_rejected(bad):
    with pytest.raises(ValueError):
        action_from_dict(bad)


def test_actions_from_dicts_all_or_nothing():
    with pytest.raises(ValueError):
        actions_from_dicts([{"action": "home"}, {"action": "nope"}])


def test_describe():
    assert CUP.describe() == "big red cup (left)"
    assert ObjectRef().describe() == "object"
    assert ObjectRef("pen").describe() == "pen"
