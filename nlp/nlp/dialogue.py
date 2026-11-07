"""Small talk and quick yes/no questions."""
from __future__ import annotations

import re
from typing import Optional

from .actions import Reply, Shake
from .parser import clean, normalize

_RULES: list[tuple[str, Reply]] = [
    (r"\bhow are you\b|\bhow'?s it going\b", Reply("I'm doing well, thanks for asking!")),
    (r"\b(?:did you have|have you had) a (?:good|great|nice) day\b", Reply("Yes, I did!", Shake("yes"))),
    (r"\b(?:are you|you) (?:ok|okay|alright|good)\b", Reply("Yes, all good.", Shake("yes"))),
    (r"\b(?:thank you|thanks)\b", Reply("You're welcome!")),
    (r"\b(?:good job|well done|nice work)\b", Reply("Thank you!")),
    (r"\b(?:are you|do you) (?:hurt|tired|sad|angry)\b", Reply("No, I'm fine.", Shake("no"))),
    # Last, so a more specific rule wins ("hey robot, how are you" -> how are you).
    (r"^(?:hi|hello|hey|yo|howdy|good (?:morning|afternoon|evening))\b", Reply("Hello!")),
]

_QUESTION_START = re.compile(
    r"^(?:did|do|does|are|is|was|were|have|has|can|could|would|will|should)\b"
)


def respond(text: str) -> Optional[Reply]:
    t = clean(text)  # not normalize(): that strips a leading 'hey' as filler
    for pattern, reply in _RULES:
        if re.search(pattern, t):
            return reply
    return None


def looks_like_yes_no_question(text: str) -> bool:
    return bool(_QUESTION_START.match(normalize(text)))


UNSURE = Reply("I'm not sure.", Shake("maybe"))
