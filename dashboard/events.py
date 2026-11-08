"""Shared emit(event, payload) event bus interface for all teams."""

from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from dashboard.logger import get_logger

# Standard documented event names
DOCUMENTED_EVENTS = {
    "nlp:command_received": "Natural language command submitted by operator",
    "nlp:parsed": "NLP parser completed parsing with action list and validation status",
    "nlp:clarification_needed": "NLP requires clarifying user input (ambiguous object/action)",
    "safety:command_evaluated": "Safety layer evaluated command against workspace & kinematic bounds",
    "safety:rejected": "Command or motion rejected by safety layer with explicit reason",
    "safety:estop": "Emergency stop commanded or triggered",
    "safety:reset": "Emergency stop cleared / system reset to safe idle",
    "planner:step_changed": "Arm trajectory planner transitioned to next step in sequence",
    "arm:status_update": "Arm telemetry snapshot (joint angles, task step, packet counter, link)",
    "arm:packet_sent": "Binary/serial packet constructed and sent to ESP32 / armlink",
    "arm:packet_ack": "Arm firmware acknowledged packet receipt",
    "arm:stop": "Stop command sent to arm interface",
    "vision:detection_frame": "Vision detection pipeline produced detected objects with bounding boxes",
    "vision:target_selected": "Target block/part selected for manipulation",
    "sensors:ina219_read": "INA219 current/power sensor telemetry reading",
    "stats:production_update": "Production metrics updated (sorted, rejects, cycle time, success rate)",
    "system:watchdog_tick": "Periodic watchdog timer heartbeat",
    "system:watchdog_timeout": "Watchdog timeout alert (loss of comms heartbeat)",
}


@dataclass
class Event:
    event: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    time_ms: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
    seq: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "time_ms": self.time_ms,
            "seq": self.seq,
        }


class EventBus:
    def __init__(self):
        self._lock = Lock()
        self._seq = 0
        self._subscribers: List[tuple[str, Callable[[Event], None]]] = []
        self._recent_events: List[Event] = []
        self._max_recent = 500
        self._broadcast_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def subscribe(self, pattern: str, handler: Callable[[Event], None]) -> None:
        """Register a handler for events matching pattern (e.g. 'arm:*' or 'safety:rejected')."""
        with self._lock:
            self._subscribers.append((pattern, handler))

    def unsubscribe(self, handler: Callable[[Event], None]) -> None:
        """Remove a subscriber."""
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s[1] != handler]

    def set_broadcast_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback to broadcast event to live WebSocket connections."""
        self._broadcast_callback = cb

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> Event:
        """Emit an event onto the shared event bus, log it to JSONL, and notify subscribers.

        Args:
            event: The dot/colon-separated event name (e.g. 'safety:rejected')
            payload: Arbitrary JSON-serializable dictionary with event details
        """
        if payload is None:
            payload = {}
        elif not isinstance(payload, dict):
            payload = {"data": payload}

        with self._lock:
            self._seq += 1
            ev = Event(event=event, payload=payload, seq=self._seq)
            self._recent_events.append(ev)
            if len(self._recent_events) > self._max_recent:
                self._recent_events.pop(0)
            subs = list(self._subscribers)
            broadcast = self._broadcast_callback

        # Log event to JSONL file
        try:
            get_logger().append(ev.to_dict())
        except Exception:
            pass

        # Push to live WebSockets if registered
        if broadcast:
            try:
                broadcast(ev.to_dict())
            except Exception:
                pass

        # Notify matching subscribers
        for pattern, handler in subs:
            if pattern == "*" or fnmatch.fnmatch(event, pattern):
                try:
                    handler(ev)
                except Exception as exc:
                    print(f"Error in event handler for {event}: {exc}")

        return ev

    def get_recent(self, limit: int = 100, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the most recent events in reverse-chronological order or filtered by prefix."""
        with self._lock:
            events = list(self._recent_events)
        if category:
            events = [e for e in events if e.event.startswith(category)]
        return [e.to_dict() for e in events[-limit:]]

    def clear(self) -> None:
        """Clear recent event cache in memory."""
        with self._lock:
            self._recent_events.clear()


# Global event bus singleton
_bus = EventBus()


def emit(event: str, payload: Optional[Dict[str, Any]] = None) -> Event:
    """Public helper: Emit an event to the shared event bus."""
    return _bus.emit(event, payload)


def subscribe(pattern: str, handler: Callable[[Event], None]) -> None:
    """Public helper: Subscribe to events by pattern."""
    _bus.subscribe(pattern, handler)


def unsubscribe(handler: Callable[[Event], None]) -> None:
    """Public helper: Unsubscribe from events."""
    _bus.unsubscribe(handler)


def get_recent_events(limit: int = 100, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Public helper: Retrieve recent events."""
    return _bus.get_recent(limit=limit, category=category)


def set_broadcast_callback(cb: Callable[[Dict[str, Any]], None]) -> None:
    """Public helper: Connect live WebSocket broadcaster."""
    _bus.set_broadcast_callback(cb)
