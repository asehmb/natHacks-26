"""Pluggable fallback for utterances the rule parser can't handle."""
from __future__ import annotations

import json
import urllib.request
from typing import Optional, Protocol

from .actions import Action, actions_from_dicts

SYSTEM_PROMPT = """Convert the user's request for a robot arm into JSON: {"actions": [...]}.
Allowed actions (field names exact):
{"action":"pick_up","target":{"noun":str,"color":str|null,"size":str|null,"position":str|null}}
{"action":"put_down","target":ObjectRef|null}
{"action":"put_in","container":ObjectRef,"target":ObjectRef|null,"relation":"in"|"on"}
{"action":"give","target":ObjectRef|null}
{"action":"move_relative","direction":"forward|backward|left|right|up|down","amount":"small|medium|large","distance_mm":number|null}
{"action":"home"} {"action":"return_to_zero"} {"action":"rotate_joint","joint":int,"angle":number|null,"relative":bool}
{"action":"open_hand"} {"action":"close_hand"} {"action":"shake","kind":"yes|no|maybe"}
{"action":"reply","text":str,"gesture":{"action":"shake","kind":"yes|no|maybe"}|null}
If the request is not a robot command, answer with a single reply action. Output JSON only."""


class Fallback(Protocol):
    def parse(self, text: str) -> Optional[list[Action]]:
        """Return validated actions, or None if it can't help."""


class NullFallback:
    def parse(self, text: str) -> Optional[list[Action]]:
        return None


def validate_json_actions(raw: str) -> Optional[list[Action]]:
    """Parse and validate model output; anything malformed yields None."""
    try:
        data = json.loads(raw)
        actions = actions_from_dicts(data["actions"])
    except (ValueError, KeyError, TypeError):
        return None
    return actions or None


class OllamaFallback:
    """Local small LLM via Ollama's HTTP API (stdlib only). Output is schema-validated."""

    def __init__(self, model: str = "qwen2.5:1.5b", url: str = "http://localhost:11434", timeout: float = 20.0):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout

    def parse(self, text: str) -> Optional[list[Action]]:
        body = json.dumps({
            "model": self.model, "system": SYSTEM_PROMPT, "prompt": text,
            "format": "json", "stream": False, "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(f"{self.url}/api/generate", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return validate_json_actions(json.load(resp)["response"])
        except (OSError, ValueError, KeyError):
            return None
