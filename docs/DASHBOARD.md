# natHacks '26 Robotic Sorting Arm - Control & Monitoring Dashboard

The Dashboard is the project's central control and monitoring station. Designed specifically for the judges' pitch and live demonstrations, it provides complete visibility into natural language interpretation, safety boundaries, real-time arm kinematics, vision targeting, and production statistics.

---

## Deliverables & Features

### 1. Plain English Command Box
- Type any plain English command (e.g. *"Pick up the red cube and put it in the blue bin"*, *"Home the arm"*, *"Open hand"*, *"Close hand"*).
- Instant JSON display of the parsed robotic action structure.
- Status indicator pill with 3 distinct states:
  - **`ACCEPTED`** (green): Command passed kinematic & safety validation and is dispatched for execution.
  - **`REJECTED`** (red): Safety violation detected with explicit human-readable reason (e.g. *"Target (0.45, 0.20, 0.10) exceeds maximum arm reach (0.30m)"* or *"E-STOP is latched active"*).
  - **`NEEDS CLARIFICATION`** (amber): Ambiguity detected (e.g. target unspecified or multiple candidate objects), prompting the operator for clarification.

### 2. Live Vision & Webcam Feed
- **Live Local Webcam**: Captures the operator's real camera feed directly via `navigator.mediaDevices.getUserMedia` in modern browsers.
- **Simulated Inspection Bench**: Synthetic vision bench fallback generating detected objects (red cube, blue bin, green cylinder, yellow bolt).
- **Bounding Boxes & Target Marker**: Dynamic HTML5 canvas overlay draws real-time bounding boxes, class labels, confidence scores, and a pulsing cyan targeting reticle on the active target object.
- **Interactive Click-to-Target**: Click any detected object on the camera feed to immediately mark it as the target.

### 3. Arm Status & Kinematics Panel
- **Current Task Step Phase Trail**: Visual stepped pipeline displaying:
  `PLANNING` ➔ `HOMING` ➔ `HOVER` ➔ `DESCEND` ➔ `GRASP` ➔ `LIFT` ➔ `PLACE` ➔ `RETRACT` ➔ `IDLE`
- **Commanded Joint Positions**:
  - Base (J0), Shoulder (J1), Elbow (J2), Wrist (J3), and Gripper (J4).
  - Displayed in degrees, radians, and animated visual progress gauges.
- **2D Animated Kinematic Arm Preview**: Real-time canvas diagram rendering the geometric arm linkage angles and gripper open/closed state.
- **Last Accepted Packet Counter**: Live incrementing packet sequence number matching the `armlink` protocol counter (`PKT #1042`).
- **Link & Watchdog State**:
  - Link status indicator (`SIMULATED (ONLINE)` / `CONNECTED` / `DISCONNECTED`).
  - Active watchdog countdown timer (2000ms timeout window with heartbeat ticks and timeout detection).

### 4. Timestamped Event Log with Rejections Highlighted
- Microsecond/millisecond resolution ISO timestamps (`[HH:MM:SS.mmm]`).
- Prominently styled log entries with **REJECTIONS** and **EMERGENCY STOP** highlighted with high-contrast red warning borders and backgrounds.
- Category filters: `ALL`, `REJECTIONS`, `SAFETY`, `ARM`, `NLP`, `VISION`.

### 5. Production Statistics
- **Parts Sorted**: Counter of successfully sorted blocks.
- **Rejects**: Count of safety rejections and pick failures.
- **Success Rate**: Pick accuracy percentage `sorted / (sorted + rejects) * 100%`.
- **Cycle Time**: Average cycle time and duration of the latest pick & place cycle.

### 6. Emergency Stop Button
- Large high-visibility red **STOP** button.
- Software halt command that immediately terminates motion, aborts the planner, latches the E-STOP state, and displays the safety banner:
  *(“The physical kill switch remains the real safety”)*.
- **RESET** button to restore safe ready operation.

### 7. Shared Event Bus Interface (`emit(event, payload)`)
- Global Python function `emit(event, payload)` and HTTP `POST /api/emit` endpoint.
- Complete documentation in `docs/EVENT_BUS.md` and accessible via the **EVENT BUS API** button in the dashboard.

### 8. JSONL Run Log
- All events, commands, kinematic telemetry, and sensor samples are persistently logged to `logs/latest_run.jsonl`.
- One-click **EXPORT JSONL LOG** button to download the dataset for metrics analysis and post-run grading.

### 9. Stretch Deliverables
- **Stretch 1: Open-Loop vs Camera-Guided Pick Success Table**:
  - Live side-by-side comparison table tracking Attempts, Successes, Rejects/Misses, Accuracy %, and Average Cycle Time.
  - Interactive Mode Toggle button (`CAMERA-GUIDED` vs `OPEN-LOOP`) allowing judges to see the direct accuracy improvement of closed-loop visual servoing over blind open-loop execution.
- **Stretch 2: INA219 Current & Power Telemetry**:
  - Real-time current draw monitor (`mA`), bus voltage (`V`), shunt voltage (`mV`), and power consumption (`mW`).
  - Dynamically responds to arm motion and motor stall during gripper clamping, with overcurrent warning thresholds (> 1400mA).

---

## Running the Dashboard

### Prerequisites
Python 3.10+ with `fastapi`, `uvicorn`, `numpy`, `PyYAML`, `Pillow`.

### Launch Command
```bash
python3 run_dashboard.py
```
Or with specific port:
```bash
python3 run_dashboard.py --port 8080
```

Open your browser to:
**http://localhost:8000**
