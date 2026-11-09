"""Typed actions emitted by the NLP layer and consumed by the robot action layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Optional

DIRECTIONS = ("forward", "backward", "left", "right", "up", "down")
AMOUNTS = ("small", "medium", "large")
SHAKES = ("yes", "no", "maybe")


@dataclass(frozen=True)
class ObjectRef:
    """A description of an object to find, e.g. red screwdriver on the left."""

    noun: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    position: Optional[str] = None

    def describe(self) -> str:
        parts = [self.size, self.color, self.noun or "object"]
        text = " ".join(p for p in parts if p)
        return f"{text} ({self.position})" if self.position else text


class Action:
    """Base class. Subclasses are frozen dataclasses with a unique `name`."""

    name: ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.name}
        for f in fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if isinstance(v, ObjectRef):
                v = asdict(v)
            elif isinstance(v, Action):
                v = v.to_dict()
            out[f.name] = v
        return out


_REGISTRY: dict[str, type[Action]] = {}


def _action(cls):
    cls = dataclass(frozen=True)(cls)
    _REGISTRY[cls.name] = cls
    return cls


@_action
class PickUp(Action):
    name = "pick_up"
    target: ObjectRef


@_action
class PutDown(Action):
    """Put down the held object (target None) or a specific one."""

    name = "put_down"
    target: Optional[ObjectRef] = None


@_action
class PutIn(Action):
    """Put the held object (target None) in/on a container."""

    name = "put_in"
    container: ObjectRef
    target: Optional[ObjectRef] = None
    relation: str = "in"


@_action
class Give(Action):
    """Hand an object (None = the held one) to the user."""

    name = "give"
    target: Optional[ObjectRef] = None


@_action
class MoveRelative(Action):
    name = "move_relative"
    direction: str
    amount: str = "medium"
    distance_mm: Optional[float] = None

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"bad direction {self.direction!r}")
        if self.amount not in AMOUNTS:
            raise ValueError(f"bad amount {self.amount!r}")


@_action
class Home(Action):
    name = "home"


@_action
class ReturnToUser(Action):
    """Come back to the person (the hand-off pose), e.g. "come back to me"."""

    name = "return_to_user"


@_action
class ReturnToZero(Action):
    name = "return_to_zero"


@_action
class RotateJoint(Action):
    name = "rotate_joint"
    joint: int
    angle: Optional[float] = None
    relative: bool = True

    def __post_init__(self):
        if not isinstance(self.joint, int) or self.joint < 0:
            raise ValueError(f"bad joint {self.joint!r}")


@_action
class OpenHand(Action):
    name = "open_hand"


@_action
class CloseHand(Action):
    name = "close_hand"


@_action
class Shake(Action):
    """Gesture: kind is 'yes', 'no' or 'maybe' (shrug)."""

    name = "shake"
    kind: str

    def __post_init__(self):
        if self.kind not in SHAKES:
            raise ValueError(f"bad shake {self.kind!r}")


@_action
class Reply(Action):
    """Say something, optionally with a gesture."""

    name = "reply"
    text: str
    gesture: Optional[Shake] = None


def _convert(key: str, value: Any) -> Any:
    if key in ("target", "container") and isinstance(value, dict):
        return ObjectRef(**value)
    if key == "gesture" and isinstance(value, dict):
        a = action_from_dict(value)
        if not isinstance(a, Shake):
            raise ValueError("gesture must be a shake action")
        return a
    return value


def action_from_dict(d: dict[str, Any]) -> Action:
    """Build and validate an Action from untrusted data (e.g. LLM JSON)."""
    if not isinstance(d, dict) or d.get("action") not in _REGISTRY:
        raise ValueError(f"unknown action: {d!r}")
    cls = _REGISTRY[d["action"]]
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    extra = set(d) - allowed - {"action"}
    if extra:
        raise ValueError(f"unexpected fields {sorted(extra)} for {d['action']}")
    try:
        return cls(**{k: _convert(k, v) for k, v in d.items() if k != "action"})
    except TypeError as e:
        raise ValueError(str(e)) from e


def actions_from_dicts(items: list[dict[str, Any]]) -> list[Action]:
    return [action_from_dict(d) for d in items]
