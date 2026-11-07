"""Route parsed actions to handlers registered by the robot code."""
from __future__ import annotations

from typing import Callable

from .actions import Action, Reply


class Dispatcher:
    def __init__(self):
        self._handlers: dict[type[Action], Callable[[Action], None]] = {}

    def register(self, action_cls: type[Action], handler: Callable[[Action], None]) -> None:
        self._handlers[action_cls] = handler

    def dispatch(self, actions: list[Action]) -> None:
        """Run actions in order. Raises KeyError if any has no handler (checked up front)."""
        missing = [type(a).__name__ for a in actions if type(a) not in self._handlers]
        if missing:
            raise KeyError(f"no handler registered for: {', '.join(missing)}")
        for a in actions:
            self._handlers[type(a)](a)
            if isinstance(a, Reply) and a.gesture is not None:
                self.dispatch([a.gesture])
