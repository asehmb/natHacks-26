"""Arm Bridge: Drives kinematics planner / mock arm, tracks joint telemetry, watchdog, and packet counters."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add repo search paths for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([
    str(REPO_ROOT),
    str(REPO_ROOT / "Kinematics"),
    str(REPO_ROOT / "firmware" / "comms"),
    str(REPO_ROOT / "nlp"),
])

from dashboard.events import emit
from dashboard.metrics import metrics
from dashboard.safety import safety


@dataclass
class JointState:
    base_deg: float = 90.0
    shoulder_deg: float = 90.0
    elbow_deg: float = 90.0
    wrist_deg: float = 90.0
    gripper_deg: float = 90.0

    base_rad: float = 0.0
    shoulder_rad: float = 0.0
    elbow_rad: float = 0.0
    wrist_rad: float = 0.0
    gripper_rad: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deg": {
                "base": round(self.base_deg, 1),
                "shoulder": round(self.shoulder_deg, 1),
                "elbow": round(self.elbow_deg, 1),
                "wrist": round(self.wrist_deg, 1),
                "gripper": round(self.gripper_deg, 1),
            },
            "rad": {
                "base": round(self.base_rad, 3),
                "shoulder": round(self.shoulder_rad, 3),
                "elbow": round(self.elbow_rad, 3),
                "wrist": round(self.wrist_rad, 3),
                "gripper": round(self.gripper_rad, 3),
            },
        }


class ArmBridge:
    def __init__(self):
        self._lock = threading.Lock()
        self.link_mode: str = "SIMULATED"  # "SIMULATED" | "CONNECTED" | "DISCONNECTED"
        self.link_state: str = "ONLINE"
        self.packet_counter: int = 1042     # Starts at realistic counter for demo
        self.task_step: str = "IDLE"        # "IDLE" | "PLANNING" | "HOMING" | "HOVER" | "DESCEND" | "GRASP" | "LIFT" | "PLACE" | "RETRACT" | "DONE" | "ERROR" | "ESTOP"
        self.current_step_index: int = 0
        self.total_step_count: int = 0
        self.step_message: str = "System ready for commands"
        self.joints = JointState()
        self.is_moving: bool = False
        self.is_gripping: bool = False
        
        # Watchdog settings
        self.watchdog_timeout_ms: int = 2000
        self.last_heartbeat_time: float = time.time()
        self.watchdog_state: str = "HEALTHY" # "HEALTHY" | "WARNING" | "TRIPPED"
        
        # Background execution thread
        self._stop_requested: bool = False
        self._execution_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._running: bool = True

        # Initialize mock arm if available
        self._init_arm()

        # Start background watchdog thread
        self._start_watchdog()

    def _init_arm(self):
        try:
            from mock_arm import MockArm
            self._mock_arm = MockArm()
        except Exception:
            self._mock_arm = None

    def _start_watchdog(self):
        def loop():
            while self._running:
                time.sleep(0.2)
                self._check_watchdog()
        self._watchdog_thread = threading.Thread(target=loop, daemon=True)
        self._watchdog_thread.start()

    def _check_watchdog(self):
        with self._lock:
            elapsed_ms = int((time.time() - self.last_heartbeat_time) * 1000)
            if self.task_step == "ESTOP":
                self.watchdog_state = "ESTOP_LATCHED"
            elif elapsed_ms > self.watchdog_timeout_ms:
                if self.watchdog_state != "TRIPPED":
                    self.watchdog_state = "TRIPPED"
                    emit("system:watchdog_timeout", {
                        "message": f"Comms watchdog timeout: {elapsed_ms}ms without heartbeat",
                    })
            elif elapsed_ms > self.watchdog_timeout_ms * 0.7:
                self.watchdog_state = "WARNING"
            else:
                self.watchdog_state = "HEALTHY"

            state = self.watchdog_state

        emit("system:watchdog_tick", {
            "elapsed_ms": elapsed_ms,
            "timeout_ms": self.watchdog_timeout_ms,
            "status": state,
        })
        # Periodically update INA219 current sensor reading based on active motion
        metrics.update_ina219(
            arm_state=self.task_step.lower(),
            moving=self.is_moving,
            gripping=self.is_gripping,
        )

    def heartbeat(self) -> None:
        """Reset the comms watchdog timer."""
        with self._lock:
            self.last_heartbeat_time = time.time()
            if self.watchdog_state != "ESTOP_LATCHED":
                self.watchdog_state = "HEALTHY"

    def stop(self, reason: str = "Operator clicked STOP button") -> None:
        """Emergency stop: halts motion immediately."""
        with self._lock:
            self._stop_requested = True
            self.task_step = "ESTOP"
            self.step_message = "EMERGENCY STOPPED"
            self.is_moving = False
            self.is_gripping = False

        safety.trigger_estop(reason)
        emit("arm:stop", {"reason": reason})
        self._emit_status()

    def reset(self) -> None:
        """Reset arm from stop / error to idle."""
        with self._lock:
            self._stop_requested = False
            self.task_step = "IDLE"
            self.step_message = "Ready"
            self.is_moving = False
            self.is_gripping = False
            self.last_heartbeat_time = time.time()
            self.watchdog_state = "HEALTHY"

        safety.reset_estop()
        self._emit_status()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            elapsed_ms = int((time.time() - self.last_heartbeat_time) * 1000)
            return {
                "link_mode": self.link_mode,
                "link_state": self.link_state,
                "task_step": self.task_step,
                "step_index": self.current_step_index,
                "total_steps": self.total_step_count,
                "step_message": self.step_message,
                "packet_counter": self.packet_counter,
                "watchdog_state": self.watchdog_state,
                "watchdog_ms_remaining": max(0, self.watchdog_timeout_ms - elapsed_ms),
                "is_moving": self.is_moving,
                "is_gripping": self.is_gripping,
                "joints": self.joints.to_dict(),
            }

    def _emit_status(self):
        status = self.get_status()
        emit("arm:status_update", status)

    def set_joints(self, base_deg: float, shoulder_deg: float, elbow_deg: float, wrist_deg: float, gripper_deg: float) -> None:
        with self._lock:
            self.packet_counter += 1
            self.joints.base_deg = base_deg
            self.joints.shoulder_deg = shoulder_deg
            self.joints.elbow_deg = elbow_deg
            self.joints.wrist_deg = wrist_deg
            self.joints.gripper_deg = gripper_deg

            self.joints.base_rad = math.radians(base_deg - 90.0)
            self.joints.shoulder_rad = math.radians(shoulder_deg - 90.0)
            self.joints.elbow_rad = math.radians(elbow_deg - 90.0)
            self.joints.wrist_rad = math.radians(wrist_deg - 90.0)
            self.joints.gripper_rad = (gripper_deg - 70.0) / 40.0

        self._emit_status()

    def execute_pick_and_place_async(self, target_xyz: Tuple[float, float, float], dest_xyz: Tuple[float, float, float] = (0.24, -0.08, 0.04)) -> None:
        """Run full pick and place sequence in a background thread."""
        if self._stop_requested or safety.is_stopped:
            emit("safety:rejected", {"reason": "Cannot start motion: E-STOP is active."})
            return

        def task():
            metrics.start_cycle()
            steps = [
                ("PLANNING", "Computing IK trajectory and obstacle clearance", (90, 90, 90, 90, 90), False, 0.4),
                ("HOMING", "Moving to safe transit home pose", (90, 95, 85, 90, 90), False, 0.5),
                ("HOVER", f"Hovering above object at ({target_xyz[0]:.2f}, {target_xyz[1]:.2f})", (82, 105, 100, 90, 110), False, 0.6),
                ("DESCEND", "Descending toward object grasp altitude", (82, 115, 112, 90, 110), False, 0.6),
                ("GRASP", "Closing gripper claws on object", (82, 115, 112, 90, 75), True, 0.7),
                ("LIFT", "Lifting object clear of workspace", (82, 95, 90, 90, 75), True, 0.6),
                ("PLACE", f"Translating to destination bin at ({dest_xyz[0]:.2f}, {dest_xyz[1]:.2f})", (108, 112, 105, 90, 75), True, 0.7),
                ("RETRACT", "Opening gripper and returning to safe home", (90, 90, 90, 90, 110), False, 0.6),
                ("DONE", "Pick and place sequence successfully completed", (90, 90, 90, 90, 90), False, 0.2),
            ]

            total = len(steps)
            for idx, (step_name, msg, angles, gripping, duration) in enumerate(steps):
                if self._stop_requested or safety.is_stopped:
                    with self._lock:
                        self.task_step = "ESTOP"
                        self.step_message = "Motion aborted by STOP command"
                        self.is_moving = False
                        self.is_gripping = False
                    self._emit_status()
                    metrics.record_reject(reason="Stopped by user")
                    return

                with self._lock:
                    self.task_step = step_name
                    self.current_step_index = idx + 1
                    self.total_step_count = total
                    self.step_message = msg
                    self.is_moving = (step_name not in ("IDLE", "DONE", "ERROR", "ESTOP"))
                    self.is_gripping = gripping
                    self.packet_counter += 1

                self.set_joints(*angles)
                emit("planner:step_changed", {
                    "step": step_name,
                    "index": idx + 1,
                    "total": total,
                    "message": msg,
                    "xyz": list(target_xyz) if step_name in ("HOVER", "DESCEND", "GRASP") else list(dest_xyz),
                })
                emit("arm:packet_sent", {
                    "counter": self.packet_counter,
                    "cmd": "MOVE",
                    "targets": list(angles),
                })
                emit("arm:packet_ack", {
                    "counter": self.packet_counter,
                    "accepted": True,
                })
                self.heartbeat()
                time.sleep(duration)

            with self._lock:
                self.task_step = "IDLE"
                self.step_message = "Ready"
                self.is_moving = False
                self.is_gripping = False
            self._emit_status()
            metrics.record_success()

        self._execution_thread = threading.Thread(target=task, daemon=True)
        self._execution_thread.start()

    def execute_home_async(self) -> None:
        """Home the arm."""
        def task():
            with self._lock:
                self.task_step = "HOMING"
                self.step_message = "Moving arm to home calibration pose"
                self.is_moving = True
            self._emit_status()
            time.sleep(0.8)
            self.set_joints(90.0, 90.0, 90.0, 90.0, 90.0)
            with self._lock:
                self.task_step = "IDLE"
                self.step_message = "Homed and ready"
                self.is_moving = False
            self._emit_status()
        threading.Thread(target=task, daemon=True).start()


# Global singleton arm bridge
arm = ArmBridge()
