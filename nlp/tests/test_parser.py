import pytest

from nlp import (
    CloseHand, Give, Home, MoveRelative, ObjectRef, OpenHand, PickUp, Processor, PutDown, PutIn,
    Reply, ReturnToZero, RotateJoint, Shake,
)

SD = ObjectRef("screwdriver", "red")
BOX = ObjectRef("box")

CASES = [
    ("grab that red screwdriver", [PickUp(SD)]),
    ("pick up the red screw driver", [PickUp(SD)]),
    ("Please pick the red screwdriver up", [PickUp(SD)]),
    ("could you grab the red screwdriver for me", [PickUp(SD)]),
    ("pick up the big blue cup on the left", [PickUp(ObjectRef("cup", "blue", "big", "left"))]),
    ("put down", [PutDown(None)]),
    ("put it down", [PutDown(None)]),
    ("put the red screwdriver in the box", [PutIn(BOX, SD)]),
    ("put in the box", [PutIn(BOX, None)]),
    ("place it on the table", [PutIn(ObjectRef("table"), None, "on")]),
    ("pick up the red screwdriver and give it to me", [PickUp(SD), Give(None)]),
    ("pick up the red screwdriver and put in the box", [PickUp(SD), PutIn(BOX, None)]),
    ("pick up the red screwdriver then put it in the box", [PickUp(SD), PutIn(BOX, None)]),
    ("hand me the red screwdriver", [Give(SD)]),
    ("get me the red screwdriver", [PickUp(SD), Give(None)]),
    ("come a bit closer", [MoveRelative("forward", "small")]),
    ("go back a bit", [MoveRelative("backward", "small")]),
    ("back up a lot", [MoveRelative("backward", "large")]),
    ("move left 5 cm", [MoveRelative("left", "medium", 50.0)]),
    ("go home", [Home()]),
    ("home", [Home()]),
    ("return to 0", [ReturnToZero()]),
    ("return to zero", [ReturnToZero()]),
    ("rotate joint 3", [RotateJoint(3)]),
    ("rotate joint three by 45 degrees", [RotateJoint(3, 45.0, True)]),
    ("rotate joint 2 to 90", [RotateJoint(2, 90.0, False)]),
    ("open your hand", [OpenHand()]),
    ("close the hand", [CloseHand()]),
    ("shake yes", [Shake("yes")]),
    ("shake no", [Shake("no")]),
    ("shake maybe", [Shake("maybe")]),
    ("shrug", [Shake("maybe")]),
    ("nod", [Shake("yes")]),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_parse(text, expected):
    r = Processor().process(text)
    assert r.actions == expected, r
    assert r.source == "rules"


def test_typos():
    r = Processor().process("pik up the red screwdriver and go home")
    assert r.actions == [PickUp(SD), Home()]


def test_pronoun_resolves_across_utterances():
    p = Processor()
    p.process("pick up the red screwdriver")
    p.process("put it down")
    assert p.process("pick it up").actions == [PickUp(SD)]


def test_pick_up_without_object_asks():
    r = Processor().process("pick up")
    assert isinstance(r.actions[0], Reply)


@pytest.mark.parametrize("text,gesture", [
    ("did you have a good day", "yes"),
    ("did you have a good day?", "yes"),
    ("are you a robot", "maybe"),
])
def test_dialogue(text, gesture):
    r = Processor().process(text)
    assert r.source == "dialogue"
    assert r.actions[0].gesture == Shake(gesture)


def test_partial_command_is_not_executed():
    r = Processor().process("pick up the red screwdriver and flibber the wozzle")
    assert r.source == "none"
    assert all(isinstance(a, Reply) for a in r.actions)


def test_gibberish():
    r = Processor().process("asdf qwer")
    assert r.source == "none"
