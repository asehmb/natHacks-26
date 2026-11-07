import pytest

from nlp import Processor, Shake
from nlp import dialogue


@pytest.mark.parametrize("text", [
    "hi", "Hello!", "good morning",
    pytest.param("hey there", marks=pytest.mark.xfail(
        strict=True, reason="known bug: normalize() strips leading 'hey' as filler, leaving 'there'")),
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
