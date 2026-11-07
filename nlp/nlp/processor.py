"""Top-level entry point: rules -> dialogue -> fallback."""
from __future__ import annotations

from typing import Optional

from . import dialogue
from .actions import Reply
from .fallback import Fallback, NullFallback
from .parser import Parser, ParseResult


class Processor:
    def __init__(self, fallback: Optional[Fallback] = None):
        self.parser = Parser()
        self.fallback = fallback or NullFallback()

    def process(self, text: str) -> ParseResult:
        result = self.parser.parse(text)
        if result.actions and not result.unparsed:
            return result

        # Never execute a partially understood command chain.
        if not result.actions:
            reply = dialogue.respond(text)
            if reply:
                return ParseResult([reply], [], "dialogue")

        actions = self.fallback.parse(text)
        if actions:
            return ParseResult(actions, [], "fallback")

        if not result.actions and dialogue.looks_like_yes_no_question(text):
            return ParseResult([dialogue.UNSURE], [], "dialogue")

        bad = "; ".join(result.unparsed) or text
        return ParseResult([Reply(f"Sorry, I didn't understand: {bad}")], result.unparsed, "none")
