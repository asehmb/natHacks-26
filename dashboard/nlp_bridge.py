"""NLP Bridge: Parses natural language commands, evaluates safety, and dispatches actions."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from dashboard.arm_bridge import arm
from dashboard.events import emit
from dashboard.metrics import metrics
from dashboard.safety import safety
from dashboard.vision_bridge import vision

# Try loading NLP Processor
try:
    from nlp import Processor
    _processor = Processor()
except Exception as exc:
    print(f"Warning: NLP Processor fallback active: {exc}")
    _processor = None


class CommandResult:
    def __init__(self, raw_text: str, status: str, parsed_json: Dict[str, Any], reason: Optional[str] = None, prompt: Optional[str] = None):
        self.raw_text = raw_text
        self.status = status  # "accepted" | "rejected" | "needs_clarification"
        self.parsed_json = parsed_json
        self.reason = reason
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.raw_text,
            "status": self.status,
            "parsed_json": self.parsed_json,
            "reason": self.reason,
            "prompt": self.prompt,
        }


def process_command(text: str) -> CommandResult:
    """Process a plain English command, validate safety, and execute if accepted."""
    text = text.strip()
    if not text:
        return CommandResult(
            raw_text="",
            status="needs_clarification",
            parsed_json={},
            reason="Empty command received.",
            prompt="Please type a command, e.g. 'Pick up the red cube' or 'Home the arm'.",
        )

    emit("nlp:command_received", {"text": text})

    # Direct emergency stop keywords
    if any(k in text.lower() for k in ("stop", "halt", "kill", "abort", "freeze", "emergency stop")):
        arm.stop(reason=f"Natural language emergency command: '{text}'")
        res = CommandResult(
            raw_text=text,
            status="accepted",
            parsed_json={"action": "emergency_stop", "keyword": text},
            reason="Software E-STOP commanded. Physical kill switch remains the primary hardware safety.",
        )
        emit("nlp:parsed", res.to_dict())
        return res

    # Direct reset keywords
    if any(k in text.lower() for k in ("reset", "resume", "clear error", "re-enable")):
        arm.reset()
        res = CommandResult(
            raw_text=text,
            status="accepted",
            parsed_json={"action": "reset_safety"},
            reason="Safety cleared and arm reset to ready state.",
        )
        emit("nlp:parsed", res.to_dict())
        return res

    # Check if E-STOP is active before accepting normal motion commands
    if safety.is_stopped:
        reason = "System in E-STOP state. Safety layer rejected motion. Please reset or click 'Reset Safety'."
        safety.reject(reason)
        metrics.record_reject(reason=reason)
        res = CommandResult(
            raw_text=text,
            status="rejected",
            parsed_json={"action": "blocked_by_estop", "raw": text},
            reason=reason,
        )
        emit("nlp:parsed", res.to_dict())
        return res

    # Parse using NLP Processor
    actions = []
    unparsed = []
    source = "rules"

    if _processor:
        try:
            parse_res = _processor.process(text)
            actions = [a.to_dict() for a in parse_res.actions]
            unparsed = parse_res.unparsed
            source = parse_res.source
        except Exception as exc:
            print(f"Processor exception: {exc}")

    # Fallback heuristic parser if NLP processor had unparsed segments
    t_low = text.lower()
    import re

    # Direct cartesian coordinate command (e.g. "move to 0.45 0.20 0.10" or "go to 0.22 0.0 0.15")
    coord_match = re.search(r"[-+]?(?:\d*\.\d+|\d+)\s+[-+]?(?:\d*\.\d+|\d+)\s+[-+]?(?:\d*\.\d+|\d+)", text)
    if coord_match:
        try:
            parts = [float(v) for v in coord_match.group(0).split()]
            if len(parts) == 3:
                actions = [{"action": "move_cartesian", "xyz": parts}]
                unparsed = []
                source = "coordinates"
        except ValueError:
            pass

    if not actions or unparsed:
        if "home" in t_low or "zero" in t_low:
            actions = [{"action": "home"}]
            unparsed = []
        elif "open" in t_low and ("hand" in t_low or "gripper" in t_low or "claw" in t_low):
            actions = [{"action": "open_hand"}]
            unparsed = []
        elif "close" in t_low and ("hand" in t_low or "gripper" in t_low or "claw" in t_low):
            actions = [{"action": "close_hand"}]
            unparsed = []
        elif "pick" in t_low or "grab" in t_low or "lift" in t_low:
            # Detect target descriptor
            color = next((c for c in ("red", "blue", "green", "yellow") if c in t_low), None)
            noun = next((n for n in ("cube", "block", "bin", "cylinder", "bolt", "part", "object") if n in t_low), None)
            if not color and not noun:
                actions = [{"action": "pick_up", "target": {}}]
            else:
                actions = [{"action": "pick_up", "target": {"color": color, "noun": noun}}]
            unparsed = []

    # Filter out processor "Sorry, I didn't understand" reply
    if actions and actions[0].get("action") == "reply" and "didn't understand" in actions[0].get("text", "").lower():
        prompt = actions[0].get("text", f"Could not parse '{text}'.")
        res = CommandResult(
            raw_text=text,
            status="needs_clarification",
            parsed_json={"raw": text, "source": source},
            prompt=prompt,
        )
        emit("nlp:clarification_needed", {"text": text, "prompt": prompt})
        emit("nlp:parsed", res.to_dict())
        return res

    # If completely unparsed or dialogue asking for clarification
    if not actions:
        prompt = f"Could not parse '{text}'. Try 'pick up the red cube', 'home the arm', or 'move forward 50mm'."
        res = CommandResult(
            raw_text=text,
            status="needs_clarification",
            parsed_json={"raw": text, "source": source},
            prompt=prompt,
        )
        emit("nlp:clarification_needed", {"text": text, "prompt": prompt})
        emit("nlp:parsed", res.to_dict())
        return res

    primary_action = actions[0]
    act_name = primary_action.get("action")

    # Evaluate safety of the parsed action
    current_detections = vision.get_detections()
    safety_check = safety.evaluate_action(primary_action, current_detections)

    if not safety_check.allowed:
        if safety_check.status == "needs_clarification":
            res = CommandResult(
                raw_text=text,
                status="needs_clarification",
                parsed_json={"actions": actions, "source": source},
                prompt=safety_check.reason or "Target ambiguous or not located.",
            )
            emit("nlp:clarification_needed", {"text": text, "prompt": safety_check.reason})
        else:
            metrics.record_reject(reason=safety_check.reason or "Safety limit violation")
            res = CommandResult(
                raw_text=text,
                status="rejected",
                parsed_json={"actions": actions, "source": source},
                reason=safety_check.reason,
            )
        emit("nlp:parsed", res.to_dict())
        return res

    # Command is ACCEPTED! Now dispatch motion
    if act_name == "home" or act_name == "return_to_zero":
        arm.execute_home_async()
    elif act_name == "open_hand":
        arm.set_joints(arm.joints.base_deg, arm.joints.shoulder_deg, arm.joints.elbow_deg, arm.joints.wrist_deg, 110.0)
    elif act_name == "close_hand":
        arm.set_joints(arm.joints.base_deg, arm.joints.shoulder_deg, arm.joints.elbow_deg, arm.joints.wrist_deg, 75.0)
    elif act_name in ("pick_up", "put_in", "put_down"):
        target_ref = primary_action.get("target") or {}
        matched_obj = vision.find_best_match(
            noun=target_ref.get("noun"),
            color=target_ref.get("color"),
            position=target_ref.get("position"),
        )
        if matched_obj:
            vision.select_target(matched_obj.id)
            target_xyz = tuple(matched_obj.table_xyz)
        else:
            target_xyz = (0.22, -0.06, 0.025)  # safe default

        dest_xyz = (0.24, 0.08, 0.04)  # blue bin location
        arm.execute_pick_and_place_async(target_xyz=target_xyz, dest_xyz=dest_xyz)

    res = CommandResult(
        raw_text=text,
        status="accepted",
        parsed_json={"actions": actions, "source": source, "safety": "verified_safe"},
    )
    emit("nlp:parsed", res.to_dict())
    return res
