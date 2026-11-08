# Shared Event Bus Interface & Integration Guide

The Control & Monitoring Dashboard connects all subteams (NLP, Kinematics, Firmware, Vision, Sensors) using a unified, thread-safe, and asynchronous event bus.

Every event emitted is:
1. Streamed live to the dashboard frontend via WebSocket.
2. Automatically appended as a JSON record to the persistent JSONL run log (`logs/latest_run.jsonl` and `logs/run_<timestamp>.jsonl`).
3. Dispatched to any registered event subscribers in Python.

---

## 1. Quick Integration

### Python Teams (NLP, Kinematics, Vision, Host)
```python
from dashboard.events import emit

# Emit any event with arbitrary JSON-serializable dictionary payload
emit("event_name", {"key": "value"})
```

### Non-Python / Firmware / External Scripts (C, Shell, Curl)
Any process can emit events via HTTP:
```bash
curl -X POST http://localhost:8000/api/emit \
  -H "Content-Type: application/json" \
  -d '{"event": "arm:packet_ack", "payload": {"counter": 1045, "accepted": true}}'
```

---

## 2. Documented Event Specification

| Event Name | Source | Description | Example Payload |
|---|---|---|---|
| `nlp:command_received` | NLP | Raw plain English command input from user | `{"text": "pick up the red cube"}` |
| `nlp:parsed` | NLP / Bridge | Action parse output & safety status | `{"text": "...", "status": "accepted", "actions": [...]}` |
| `nlp:clarification_needed` | NLP | Ambiguous command requiring operator input | `{"text": "...", "prompt": "Found 2 matching objects. Which one?"}` |
| `safety:command_evaluated` | Safety | Command evaluation results | `{"action": "pick_up", "allowed": true}` |
| `safety:rejected` | Safety | Rejection reason & violation details | `{"reason": "Target outside workspace reach radius", "target": [0.45, 0.2]}` |
| `safety:estop` | Safety / UI | Emergency stop commanded | `{"triggered_by": "operator", "reason": "Stop button clicked"}` |
| `safety:reset` | Safety / UI | Emergency stop cleared / system restored | `{"reset_by": "operator", "state": "ready"}` |
| `planner:step_changed` | Kinematics | Arm trajectory sequence step change | `{"step": "GRASP", "index": 5, "total": 9, "message": "Closing claw", "xyz": [0.22, -0.06, 0.025]}` |
| `arm:status_update` | Kinematics / Bridge | Complete 5-DOF joint angles & comms state | `{"task_step": "MOVE", "joints": {"deg": {...}}, "packet_counter": 1042}` |
| `arm:packet_sent` | Comms / Host | Binary/serial packet dispatched to ESP32 | `{"counter": 1043, "cmd": "MOVE", "targets": [90, 110, 105, 90, 75]}` |
| `arm:packet_ack` | Comms / Firmware | Firmware acknowledged packet acceptance | `{"counter": 1043, "accepted": true}` |
| `arm:stop` | Arm / UI | Stop command sent to arm interface | `{"reason": "Software stop commanded"}` |
| `vision:detection_frame` | Vision | Detections from camera pipeline | `{"count": 4, "detections": [{"id": "obj_1", "bbox": [...]}]}` |
| `vision:target_selected` | Vision / UI | Selected target block marked | `{"target": {"id": "obj_1", "label": "red cube"}}` |
| `sensors:ina219_read` | Sensors / Hardware | INA219 current & power draw | `{"current_ma": 420.5, "voltage_v": 4.96, "power_mw": 2085.6}` |
| `stats:production_update` | Production Metrics | Metrics summary snapshot | `{"parts_sorted": 15, "rejects": 2, "avg_cycle_time_s": 4.08}` |
| `system:watchdog_tick` | Comms Watchdog | Heartbeat tick | `{"elapsed_ms": 150, "timeout_ms": 2000, "status": "HEALTHY"}` |
| `system:watchdog_timeout`| Comms Watchdog | Loss of comms alert | `{"message": "Watchdog timeout: >2000ms without heartbeat"}` |

---

## 3. Subteam Code Examples

### A. NLP Subteam
When parsing a command:
```python
from dashboard.events import emit

result = processor.process(user_text)
if result.actions:
    emit("nlp:parsed", {
        "text": user_text,
        "status": "accepted",
        "actions": [a.to_dict() for a in result.actions],
        "source": result.source,
    })
else:
    emit("nlp:clarification_needed", {
        "text": user_text,
        "prompt": "Could not understand command. Please specify target.",
    })
```

### B. Kinematics & Motion Subteam
In `Kinematics/planner.py`, pass `on_event` directly:
```python
from dashboard.events import emit

def forward_planner_event(event):
    emit("planner:step_changed", {
        "step": event.state.value.upper(),
        "index": event.index,
        "total": event.total,
        "message": event.message,
        "xyz": list(event.xyz) if event.xyz else None,
    })

planner = Planner(arm, on_event=forward_planner_event)
```

### C. Firmware / Comms Subteam
In `firmware/comms/armlink_host.py` or when receiving frames:
```python
from dashboard.events import emit

# When dispatching MOVE packet
emit("arm:packet_sent", {
    "counter": counter,
    "cmd": "MOVE",
    "targets": servo_angles,
})

# When ACK frame received from ESP32
emit("arm:packet_ack", {
    "counter": counter,
    "accepted": True,
})
```

### D. Vision Subteam
```python
from dashboard.events import emit

emit("vision:detection_frame", {
    "timestamp": time.time(),
    "count": len(detected_blocks),
    "detections": [
        {
            "id": f"block_{i}",
            "label": det.color,
            "confidence": det.confidence,
            "bbox": det.bbox, # [x1, y1, x2, y2]
            "table_xyz": det.table_xyz,
        }
        for i, det in enumerate(detected_blocks)
    ],
})
```

---

## 4. JSONL Run Log Format

Every event is recorded into `logs/latest_run.jsonl`:
```json
{"event": "safety:rejected", "payload": {"reason": "Out of workspace reach: distance 350.0mm exceeds 300.0mm"}, "timestamp": "2026-11-08T18:12:00.123456+00:00", "time_ms": 1794161520123, "seq": 42}
```
This file is available in the dashboard UI via the **EXPORT JSONL LOG** button or GET `/api/logs/download`.
