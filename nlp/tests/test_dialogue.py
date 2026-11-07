import pytest

from nlp import Processor, Shake
from nlp import dialogue


@pytest.mark.parametrize("text", [
    "hi", "Hello!", "good morning", "hey", "hey there", "Hey there!", "hello robot", "yo", "  hey   robot ",
])
def test_greetings(text):
    assert dialogue.respond(text).text == "Hello!"


@pytest.mark.parametrize("text,gesture", [
    ("did you have a good day", "yes"), ("Did you have a great day?", "yes"),
    ("are you okay", "yes"), ("are you tired", "no"),
])
def test_gesture_replies(text, gesture):
    r = Processor().process(text)
    assert r.source == "dialogue" and r.actions[0].gesture == Shake(gesture)


@pytest.mark.parametrize("text", ["thank you", "thanks", "good job", "how are you"])
def test_reply_without_gesture(text):
    r = Processor().process(text)
    assert r.source == "dialogue" and r.actions[0].gesture is None


@pytest.mark.parametrize("text", ["are you a robot", "is it going to rain", "do you like pizza?", "can the sky fall"])
def test_unknown_yes_no_questions_shrug(text):
    r = Processor().process(text)
    assert r.actions == [dialogue.UNSURE] and r.actions[0].gesture == Shake("maybe")


def test_commands_phrased_as_questions_are_still_commands():
    assert Processor().process("can you pick up the cup?").source == "rules"
    assert Processor().process("could you go home").source == "rules"


def test_non_question_gibberish_is_not_a_shrug():
    assert Processor().process("purple monkey dishwasher").source == "none"


def test_specific_rule_beats_greeting():
    assert dialogue.respond("hey robot, how are you").text == "I'm doing well, thanks for asking!"
    r = Processor().process("hey, did you have a good day")
    assert r.actions[0].gesture == Shake("yes")


def test_greeting_does_not_swallow_commands():
    from nlp import Home, PickUp, ObjectRef

    assert Processor().process("hey robot go home").actions == [Home()]
    for t in ["hello, pick up the cup", "hello pick up the cup", "hi robot, pick up the cup", "Hey! Pick up the cup"]:
        assert Processor().process(t).actions == [PickUp(ObjectRef("cup"))], t


def test_words_starting_like_greetings_are_not_greetings():
    assert dialogue.respond("history lesson") is None and dialogue.respond("hello") is not None
    assert dialogue.respond("yolo") is None
