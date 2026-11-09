import random
import string

import pytest

from nlp import (
    CloseHand, Give, Home, MoveRelative, ObjectRef, OpenHand, PickUp, Processor, PutDown, PutIn,
    Reply, ReturnToUser, ReturnToZero, RotateJoint, Shake,
)
from nlp.parser import Parser, normalize

RED_SD = ObjectRef("screwdriver", "red")


def parse(text):
    return Processor().process(text)


@pytest.mark.parametrize("text", [
    "GRAB THE RED SCREWDRIVER", "  grab   the   red   screwdriver  ", "Grab the red screwdriver!",
    "grab the red screwdriver.", "grab the red screwdriver, please", "please, grab the red screwdriver",
    "hey robot, could you please grab the red screwdriver", "I want you to grab the red screwdriver",
    "can you pick up that red screw driver for me", "go ahead and grab the red screwdriver",
])
def test_phrasing_variants_give_same_action(text):
    assert parse(text).actions == [PickUp(RED_SD)]


@pytest.mark.parametrize("verb", ["grab", "pick up", "take", "fetch", "grasp", "lift", "get"])
def test_pick_verbs(verb):
    assert parse(f"{verb} the cup").actions == [PickUp(ObjectRef("cup"))]


@pytest.mark.parametrize("text,expected", [
    ("drop it", [PutDown(None)]),
    ("set it down", [PutDown(None)]),
    ("put the cup down", [PutDown(ObjectRef("cup"))]),
    ("put the cup into the bin", [PutIn(ObjectRef("bin"), ObjectRef("cup"))]),
    ("place the cup inside the red box", [PutIn(ObjectRef("box", "red"), ObjectRef("cup"))]),
    ("put the cup on the left tray", [PutIn(ObjectRef("tray", position="left"), ObjectRef("cup"), "on")]),
])
def test_put_variants(text, expected):
    assert parse(text).actions == expected


@pytest.mark.parametrize("text,expected", [
    ("pass me the red screwdriver", [Give(RED_SD)]),
    ("give it to me", [Give(None)]),
    ("bring me the red screwdriver", [Give(RED_SD)]),
])
def test_give_variants(text, expected):
    assert parse(text).actions == expected


def test_object_attributes():
    r = parse("grab the small grey wrench on the right")
    assert r.actions == [PickUp(ObjectRef("wrench", "gray", "small", "right"))]


def test_placeholder_noun_keeps_attributes():
    assert parse("grab the red one").actions == [PickUp(ObjectRef(None, "red"))]


@pytest.mark.parametrize("text,expected", [
    ("move forward", MoveRelative("forward")),
    ("come closer", MoveRelative("forward")),
    ("step back", MoveRelative("backward")),
    ("back up", MoveRelative("backward")),
    ("move left a little", MoveRelative("left", "small")),
    ("move right slightly", MoveRelative("right", "small")),
    ("go up a lot", MoveRelative("up", "large")),
    ("move down", MoveRelative("down")),
    ("move left 2 inches", MoveRelative("left", "medium", 50.8)),
    ("come closer 30 mm", MoveRelative("forward", "medium", 30.0)),
    ("move up 1.5 cm", MoveRelative("up", "medium", 15.0)),
])
def test_movement(text, expected):
    assert parse(text).actions == [expected]


@pytest.mark.parametrize("text,expected", [
    ("go to home", Home()), ("return home", Home()), ("go back home", Home()),
    ("go home position", Home()),
    ("reset", ReturnToZero()), ("go to zero", ReturnToZero()), ("return to 0", ReturnToZero()),
])
def test_home_and_zero(text, expected):
    assert parse(text).actions == [expected]


@pytest.mark.parametrize("text,expected", [
    ("turn joint 1", RotateJoint(1)),
    ("rotate joint 4 by 10 degrees", RotateJoint(4, 10.0, True)),
    ("rotate joint 4 to 10 degrees", RotateJoint(4, 10.0, False)),
    ("rotate joint 4 by 10 degrees counterclockwise", RotateJoint(4, -10.0, True)),
    ("rotate joint six by ten degrees", RotateJoint(6, 10.0, True)),
])
def test_rotate(text, expected):
    assert parse(text).actions == [expected]


def test_rotate_without_joint_number_is_not_executed():
    r = parse("rotate joint")
    assert not any(isinstance(a, RotateJoint) for a in r.actions)
    assert r.source == "none"


@pytest.mark.parametrize("text,expected", [
    ("open the gripper", OpenHand()), ("open your claws", OpenHand()), ("let go", OpenHand()),
    ("close the gripper", CloseHand()), ("close your fingers", CloseHand()), ("make a fist", CloseHand()),
])
def test_hand(text, expected):
    assert parse(text).actions == [expected]


def test_open_something_else_is_not_the_hand():
    assert parse("open the door").source == "none"


@pytest.mark.parametrize("text,kind", [
    ("shake your head no", "no"), ("shake your head", "no"), ("nod yes", "yes"),
    ("shrug maybe", "maybe"), ("shake yes", "yes"),
])
def test_gestures(text, kind):
    assert parse(text).actions == [Shake(kind)]


def test_long_chains_keep_order():
    r = parse("go home, open your hand, pick up the red screwdriver, put it in the box and then close your hand then return to 0")
    assert r.actions == [Home(), OpenHand(), PickUp(RED_SD), PutIn(ObjectRef("box")), CloseHand(), ReturnToZero()]


def test_pronoun_from_earlier_command_in_same_chain():
    assert parse("grab the red screwdriver, put it down and pick it up").actions == [
        PickUp(RED_SD), PutDown(None), PickUp(RED_SD)]


def test_pronoun_without_context_asks_instead_of_guessing():
    r = parse("pick it up")
    assert len(r.actions) == 1 and isinstance(r.actions[0], Reply)


def test_put_in_without_container_asks():
    r = parse("put it in")
    assert all(isinstance(a, Reply) for a in r.actions)


def test_parsers_do_not_share_context():
    a, b = Parser(), Parser()
    a.parse("pick up the cup")
    assert b.parse("pick it up").actions[0].__class__ is Reply


@pytest.mark.parametrize("text", ["", "   ", "...", "!!!", "?", ",,,", "and", "then", "and then and"])
def test_empty_ish_input_is_safe(text):
    r = parse(text)
    assert r.source == "none"
    assert all(isinstance(a, Reply) for a in r.actions)


@pytest.mark.parametrize("text", [
    "grab the red screwdriver and flibber", "flibber and grab the red screwdriver",
    "go home and tell me a joke", "open your hand and fly away",
])
def test_unknown_clause_blocks_the_whole_chain(text):
    r = parse(text)
    assert r.source == "none"
    assert all(isinstance(a, Reply) for a in r.actions)
    assert r.unparsed


def test_nouns_that_resemble_verbs_are_not_mangled():
    assert parse("pick up the plate").actions == [PickUp(ObjectRef("plate"))]
    assert parse("grab the glove").actions == [PickUp(ObjectRef("glove"))]


def test_normalize_is_idempotent():
    for t in ["Please, GRAB the Red Screw Driver!", "go to 0.", "Hey robot, could you move left 1.5 cm please"]:
        assert normalize(normalize(t)) == normalize(t)


def test_fuzz_never_raises_and_never_returns_non_actions():
    rng = random.Random(1234)
    words = ["grab", "pick", "up", "put", "in", "the", "red", "box", "and", "then", "joint", "3", "to",
             "home", "open", "hand", "shake", "yes", "closer", "a", "bit", "it", "me", "give", "zero"]
    proc = Processor()
    for _ in range(500):
        if rng.random() < 0.5:
            text = " ".join(rng.choice(words) for _ in range(rng.randint(0, 12)))
        else:
            text = "".join(rng.choice(string.printable + "éß漢字🙂") for _ in range(rng.randint(0, 60)))
        r = proc.process(text)
        assert r.actions, text
        assert r.source in {"rules", "dialogue", "fallback", "none"}
        if r.source == "none":
            assert all(isinstance(a, Reply) for a in r.actions), text


def test_very_long_input_is_fast_and_safe():
    import time

    t = time.time()
    r = parse("grab the red screwdriver and " * 2000 + "go home")
    assert time.time() - t < 5
    assert r.actions


BOTTLE = ObjectRef("water bottle", "blue")


@pytest.mark.parametrize("text", [
    "grab the water bottle thats blue", "grab the water bottle that's blue", "grab the water bottle that is blue",
    "grab the water bottle which is blue", "grab the water bottle that looks blue",
    "grab the blue colored water bottle", "grab the water bottle that is colored blue",
    "pick up the water bottle that's blue please", "Grab the water bottle THATS blue!",
])
def test_relative_clauses_do_not_pollute_the_noun(text):
    assert parse(text).actions == [PickUp(BOTTLE)]


@pytest.mark.parametrize("text,expected", [
    ("grab the bottle that is on the left", [PickUp(ObjectRef("bottle", position="left"))]),
    ("grab the bottle thats on the right", [PickUp(ObjectRef("bottle", position="right"))]),
    ("grab the big cup thats red", [PickUp(ObjectRef("cup", "red", "big"))]),
    ("put the cup which is red in the box", [PutIn(ObjectRef("box"), ObjectRef("cup", "red"))]),
    ("put the cup thats red on the tray thats green", [PutIn(ObjectRef("tray", "green"), ObjectRef("cup", "red"), "on")]),
    ("give me the bottle that looks green", [Give(ObjectRef("bottle", "green"))]),
    ("hand me the screwdriver thats red", [Give(RED_SD)]),
    ("grab the red screwdriver thats on the left and give it to me",
     [PickUp(ObjectRef("screwdriver", "red", position="left")), Give(None)]),
])
def test_relative_clauses_in_other_commands(text, expected):
    assert parse(text).actions == expected


def test_real_nouns_survive_relative_word_filtering():
    # relative words are dropped, but ordinary nouns are untouched
    assert parse("grab the water bottle").actions == [PickUp(ObjectRef("water bottle"))]
    assert parse("grab the screwdriver").actions == [PickUp(ObjectRef("screwdriver"))]


def test_with_clause_does_not_leak_attributes_or_nouns():
    # the description after "with" belongs to a different object (the cap), so it must not recolour the bottle
    assert parse("grab the bottle with the red cap").actions == [PickUp(ObjectRef("bottle"))]
    assert parse("grab the blue bottle with a handle").actions == [PickUp(ObjectRef("bottle", "blue"))]


@pytest.mark.parametrize("text", ["come back to me", "come to me", "return to me", "come back here",
                                  "go back to me", "please come back to me"])
def test_come_back_to_me(text):
    assert parse(text).actions == [ReturnToUser()]


def test_plain_come_back_is_still_a_relative_move():
    assert parse("come back").actions == [MoveRelative("backward")]
    assert parse("come here").actions == [MoveRelative("forward")]


def test_typo_colour_before_the_noun_is_snapped():
    assert parse("put it in the bluw container").actions == [PutIn(ObjectRef("container", "blue"))]
    assert parse("grab the greem cup").actions == [PickUp(ObjectRef("cup", "green"))]
    # the head noun is never turned into a colour
    assert parse("grab the block").actions == [PickUp(ObjectRef("block"))]
    assert parse("grab the wooden block").actions == [PickUp(ObjectRef("wooden block"))]


def test_grab_put_in_come_back_chain():
    text = "grab this item, put it in the bluw container, then come back to me"
    assert parse(text).actions == [PickUp(ObjectRef("item")), PutIn(ObjectRef("container", "blue")), ReturnToUser()]
