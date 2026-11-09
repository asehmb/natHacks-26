"""Rule-based command parser: text -> list[Action]. Pure stdlib."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from . import lexicon as L
from .actions import (
    Action, CloseHand, Give, Home, MoveRelative, ObjectRef, OpenHand, PickUp, PutDown,
    PutIn, Reply, ReturnToUser, ReturnToZero, RotateJoint, Shake,
)


@dataclass
class ParseResult:
    actions: list[Action] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    source: str = "rules"  # rules | dialogue | fallback | none


def correct_tokens(tokens: list[str]) -> list[str]:
    """Snap near-miss words (typos / ASR errors) onto known vocabulary.

    The clause-initial word is matched loosely against command verbs only; later words are
    matched strictly against closed-class vocab so real nouns ("plate") aren't mangled.
    """
    out = []
    for i, t in enumerate(tokens):
        if t.isdigit() or t in L.VOCAB:
            pass
        elif i == 0 and len(t) >= 3:
            m = difflib.get_close_matches(t, L.START_VOCAB, n=1, cutoff=0.8)
            t = m[0] if m else t
        elif i > 0 and len(t) >= 4:
            m = difflib.get_close_matches(t, L.REST_VOCAB, n=1, cutoff=0.84)
            t = m[0] if m else t
        out.append(t)
    return out


def clean(text: str) -> str:
    """Lowercase and strip punctuation/extra spaces, keeping every word (greetings included)."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9\s,.'-]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # drop sentence periods, keep decimals
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """clean() plus removal of polite filler ("please", "could you") for command parsing."""
    text = clean(text)
    prefix = re.compile(rf"^{L.POLITE_PREFIX}[ ,]+")
    while prefix.match(text):
        text = prefix.sub("", text, count=1)
    suffix = re.compile(rf"[ ,]+{L.POLITE_SUFFIX}$")
    while suffix.search(text):
        text = suffix.sub("", text, count=1)
    for k, v in L.COMPOUNDS.items():
        text = text.replace(k, v)
    return text


def tokenize(segment: str) -> list[str]:
    toks = correct_tokens(segment.replace(",", " ").split())
    out = []
    for i, t in enumerate(toks):
        if t in L.NUMBER_WORDS and i > 0 and toks[i - 1] in L.NUMBER_CONTEXT:
            t = str(L.NUMBER_WORDS[t])
        out.append(t)
    return out


def parse_object(toks: list[str]) -> Optional[ObjectRef]:
    """Tokens -> ObjectRef, or None for a bare pronoun / nothing ("it", "that")."""
    cut = next((i for i, t in enumerate(toks) if t in L.OBJECT_PREPS and i > 0), len(toks))
    main, rest = toks[:cut], toks[cut:]
    color = size = position = None
    noun = []
    for t in main:
        t = L.COLOR_ALIASES.get(t, t)
        if t in L.COLORS:
            color = color or t
        elif t in L.SIZES:
            size = size or t
        elif t in L.POSITIONS:
            position = position or t
        elif t in L.DETERMINERS or t in L.PRONOUNS or t in L.PLACEHOLDER_NOUNS or t in L.RELATIVE_WORDS:
            continue
        else:
            noun.append(t)
    if color is None and len(noun) > 1:
        # A typo'd colour in front of the head noun ("bluw container"); the head noun itself is never snapped.
        for i, t in enumerate(noun[:-1]):
            m = difflib.get_close_matches(t, L.COLORS, n=1, cutoff=0.75) if len(t) >= 4 and t not in L.VOCAB else []
            if m:
                color = m[0]
                del noun[i]
                break
    for t in rest:
        if t in L.POSITIONS:
            position = position or t
    ref = ObjectRef(" ".join(noun) or None, color, size, position)
    return None if ref == ObjectRef() else ref


def _is_hand_cmd(toks: list[str]) -> Optional[Action]:
    if toks[0] in ("open", "close") and any(t in L.HAND_WORDS for t in toks[1:]):
        return OpenHand() if toks[0] == "open" else CloseHand()
    if toks[:2] == ["let", "go"]:
        return OpenHand()
    if toks[:3] == ["make", "a", "fist"] or toks[:2] == ["make", "fist"]:
        return CloseHand()
    return None


class Parser:
    def __init__(self):
        self.last_object: Optional[ObjectRef] = None  # for resolving "it"

    # ---- public API ----
    def parse(self, text: str) -> ParseResult:
        result = ParseResult()
        for seg in self._split(normalize(text)):
            actions = self._parse_clause(tokenize(seg))
            if actions is None:
                result.unparsed.append(seg)
            else:
                result.actions.extend(actions)
        return result

    # ---- internals ----
    def _split(self, text: str) -> list[str]:
        return [p for p in re.split(r"\s*(?:,|\band then\b|\bthen\b|\band\b)\s*", text) if p]

    def _remember(self, ref: Optional[ObjectRef]):
        if ref is not None:
            self.last_object = ref

    def _parse_clause(self, toks: list[str]) -> Optional[list[Action]]:
        if not toks:
            return None
        head, rest = toks[0], toks[1:]

        # rotate joint N [by|to ANGLE]
        if head in L.ROTATE_VERBS and "joint" in toks:
            nums = [t for t in toks if re.fullmatch(r"\d+(?:\.\d+)?", t)]
            if not nums:
                return None
            joint = int(float(nums[0]))
            angle = float(nums[1]) if len(nums) > 1 else None
            if angle is not None and "counterclockwise" in toks:
                angle = -angle
            relative = "to" not in toks[toks.index("joint") + 2:]
            return [RotateJoint(joint, angle, relative)]

        hand = _is_hand_cmd(toks)
        if hand:
            return [hand]

        if head in L.SHAKE_VERBS:
            kind = next((t for t in rest if t in ("yes", "no", "maybe")), None)
            kind = kind or {"nod": "yes", "shrug": "maybe"}.get(head, "no")
            return [Shake(kind)]

        joined = " ".join(toks)
        if re.fullmatch(r"(?:(?:return|go|move|come|reset)(?: back)?(?: to)? )?(?:0|zero|origin)(?: position)?", joined) \
                or joined in ("reset", "zero position"):
            return [ReturnToZero()]
        if re.fullmatch(r"(?:come|go|move|head|get)(?: back)?(?: over)? to me|come back here|return to me|back to me",
                        joined):
            return [ReturnToUser()]
        if re.fullmatch(r"(?:(?:go|return|come|move|head)(?: back)?(?: to)? )?home(?: position)?", joined):
            return [Home()]

        if head in L.GIVE_VERBS:
            ref = parse_object(rest)
            self._remember(ref)
            return [Give(ref)]

        if head in L.PUT_VERBS:
            return self._parse_put(rest)

        if head in L.PICK_VERBS and not (head == "get" and not rest):
            body = [t for t in rest if t != "up"]
            wants_me = "me" in body and parse_object([t for t in body if t != "me"]) is not None
            ref = parse_object(body)
            if ref is None:
                ref = self.last_object
            if ref is None:
                return [Reply("What should I pick up?")]
            self._remember(ref)
            return [PickUp(ref), Give(None)] if wants_me else [PickUp(ref)]

        move = self._parse_move(toks)
        if move:
            return [move]
        return None

    def _parse_put(self, rest: list[str]) -> Optional[list[Action]]:
        if "down" in rest:
            ref = parse_object([t for t in rest if t != "down"])
            self._remember(ref)
            return [PutDown(ref)]
        i = next((i for i, t in enumerate(rest) if t in L.CONTAINER_PREPS), None)
        if i is None:
            return [PutDown(parse_object(rest))] if not rest or rest == ["it"] else None
        relation = L.CONTAINER_PREPS[rest[i]]
        target = parse_object(rest[:i])
        container = parse_object(rest[i + 1:])
        if container is None:
            return [Reply(f"Put it {relation} what?")]
        self._remember(target)
        return [PutIn(container, target, relation)]

    def _parse_move(self, toks: list[str]) -> Optional[MoveRelative]:
        if toks[0] not in L.MOVE_VERBS and toks[0] not in L.DIRECTION_WORDS:
            return None
        direction = next((L.DIRECTION_WORDS[t] for t in toks if t in L.DIRECTION_WORDS), None)
        if direction is None:
            return None
        amount, distance = "medium", None
        if any(t in L.SMALL_WORDS for t in toks):
            amount = "small"
        elif any(t in L.LARGE_WORDS for t in toks):
            amount = "large"
        for i, t in enumerate(toks[:-1]):
            if re.fullmatch(r"\d+(?:\.\d+)?", t) and toks[i + 1] in L.UNIT_MM:
                distance = float(t) * L.UNIT_MM[toks[i + 1]]
        return MoveRelative(direction, amount, distance)
