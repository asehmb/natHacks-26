"""JSONL Run Logger for metrics, events, and debugging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class JSONLLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        
        # Create a session-specific log file and a latest symlink/file
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.current_log_path = self.log_dir / f"run_{timestamp_str}.jsonl"
        self.latest_log_path = self.log_dir / "latest_run.jsonl"
        
        # Ensure files exist
        self.current_log_path.touch(exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        """Thread-safely write one JSON record to the run log."""
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            with open(self.current_log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            # Also keep latest_run.jsonl updated
            with open(self.latest_log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    def get_lines(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Read the most recent records from the current log file."""
        if not self.current_log_path.exists():
            return []
        with self._lock:
            try:
                with open(self.current_log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError:
                return []
        
        records = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def export_content(self) -> str:
        """Return the complete content of the current run log."""
        with self._lock:
            if not self.current_log_path.exists():
                return ""
            try:
                with open(self.current_log_path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError:
                return ""


# Global singleton instance
_logger_instance: Optional[JSONLLogger] = None


def get_logger() -> JSONLLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = JSONLLogger()
    return _logger_instance
