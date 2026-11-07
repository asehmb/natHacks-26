import json

import pytest

from nlp import Dispatcher, Home, PickUp, ObjectRef, Processor, Reply, Shake
from nlp.actions import action_from_dict
from nlp.fallback import validate_json_actions


class Stub:
    def __init__(self, actions):
        self.actions = actions

    def parse(self, text):
        return self.actions


def test_fallback_used_when_rules_fail():
    r = Processor(Stub([Home()])).process("tell me a joke")
    assert r.source == "fallback" and r.actions == [Home()]


def test_rules_win_over_fallback():
    r = Processor(Stub([Home()])).process("pick up the cup")
    assert r.source == "rules"


def test_validate_good_json():
    raw = json.dumps({"actions": [{"action": "pick_up", "target": {"noun": "cup", "color": "red"}}]})
    assert validate_json_actions(raw) == [PickUp(ObjectRef("cup", "red"))]


@pytest.mark.parametrize("raw", [
    "not json",
    json.dumps({"actions": [{"action": "self_destruct"}]}),
    json.dumps({"actions": [{"action": "shake", "kind": "banana"}]}),
    json.dumps({"actions": [{"action": "home", "speed": 9}]}),
    json.dumps({"actions": [{"action": "rotate_joint", "joint": -1}]}),
    json.dumps({"nope": []}),
])
def test_validate_rejects_bad_json(raw):
    assert validate_json_actions(raw) is None


def test_round_trip():
    a = Reply("hi", Shake("yes"))
    assert action_from_dict(a.to_dict()) == a


def test_dispatcher_runs_in_order_and_gestures():
    log = []
    d = Dispatcher()
    d.register(Home, lambda a: log.append("home"))
    d.register(Reply, lambda a: log.append(a.text))
    d.register(Shake, lambda a: log.append(a.kind))
    d.dispatch([Home(), Reply("hi", Shake("yes"))])
    assert log == ["home", "hi", "yes"]


def test_dispatcher_missing_handler_runs_nothing():
    log = []
    d = Dispatcher()
    d.register(Home, lambda a: log.append("home"))
    with pytest.raises(KeyError):
        d.dispatch([Home(), PickUp(ObjectRef("cup"))])
    assert log == []
