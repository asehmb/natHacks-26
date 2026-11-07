"""OllamaFallback with the network mocked out: no Ollama server needed."""
import io
import json
import urllib.error

import pytest

from nlp import Home, ObjectRef, OllamaFallback, PickUp, Processor


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def reply_with(monkeypatch, response_text, capture=None):
    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture["url"], capture["body"], capture["timeout"] = req.full_url, json.loads(req.data), timeout
        return FakeResponse(json.dumps({"response": response_text}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_valid_model_output_becomes_actions(monkeypatch):
    cap = {}
    reply_with(monkeypatch, json.dumps({"actions": [{"action": "home"}]}), cap)
    assert OllamaFallback("m", url="http://host:1/", timeout=3).parse("take me home") == [Home()]
    assert cap["url"] == "http://host:1/api/generate"
    assert cap["body"]["model"] == "m" and cap["body"]["format"] == "json" and cap["body"]["stream"] is False
    assert cap["body"]["options"]["temperature"] == 0 and cap["timeout"] == 3
    assert cap["body"]["prompt"] == "take me home"


@pytest.mark.parametrize("bad", [
    "not json", "{}", '{"actions": []}', '{"actions": [{"action": "launch_missiles"}]}',
    '{"actions": [{"action": "home"}, {"action": "shake", "kind": "banana"}]}',
    '{"actions": "home"}', "null", "[]",
])
def test_bad_model_output_is_rejected_entirely(monkeypatch, bad):
    reply_with(monkeypatch, bad)
    assert OllamaFallback().parse("x") is None


@pytest.mark.parametrize("exc", [
    urllib.error.URLError("refused"), ConnectionRefusedError(), TimeoutError(), OSError("boom"),
])
def test_network_errors_return_none(monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert OllamaFallback().parse("x") is None


def test_non_json_http_body_returns_none(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse(b"<html>502</html>"))
    assert OllamaFallback().parse("x") is None


def test_processor_uses_ollama_only_when_rules_fail(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        calls.append(1)
        return FakeResponse(json.dumps({"response": json.dumps({"actions": [{"action": "home"}]})}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    p = Processor(OllamaFallback())
    assert p.process("pick up the cup").actions == [PickUp(ObjectRef("cup"))] and not calls
    r = p.process("tell me a joke")
    assert r.source == "fallback" and r.actions == [Home()] and len(calls) == 1


def test_processor_survives_dead_ollama(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    r = Processor(OllamaFallback()).process("tell me a joke")
    assert r.source == "none"
