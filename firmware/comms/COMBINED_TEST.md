# Checkpoint 3A: combined command path, outputs disabled

Checkpoint 2 passed physically. This next test sends ARM and MOVE with servo
power disconnected and firmware outputs disabled. It does not yet attach servos
or verify Servo.write calls. F1's original sketch remains separate and untouched.

## Recorded physical result

User-reported physical ESP32 result on Windows **COM3**: **checkpoint 3A passed**.
Authenticated session establishment, MOVE validation and centidegree-to-degree
conversion, software STOP handling, and communication watchdog timeout checks
passed. `outputs_enabled=0` throughout.

This is physical protocol/firmware verification, distinct from the laptop
simulation below. It verifies commanded state and software command gating.
It does **not** verify servo attachment, PWM, Servo.write calls, actual position,
motion, STOP detachment, or attached-output holding after timeout. No physical
servo movement was tested. F1 subsequently confirmed the STOP/timeout policies
recorded below; their physical servo behavior remains unverified.

## Prepare and upload

The current package is /tmp/armlink-checkpoint3/combined_only. To regenerate in
a NEW destination from WSL at the repository root:

```bash
python3 firmware/comms/prepare_combined.py --output /tmp/armlink-checkpoint3-new/combined_only
```

Copy the entire generated combined_only folder using Windows Explorer's WSL
view to C:\ArmLink-checkpoint3\combined_only. Copy test_combined.py,
test_handshake.py, and send_ik.py from firmware/comms to C:\ArmLink-checkpoint3.

With servo power disconnected, open
C:\ArmLink-checkpoint3\combined_only\combined_only.ino in Windows Arduino IDE.
Use the same board and COM port as checkpoint 2. Upload, then close Serial
Monitor and Plotter. Before each test run, reset EN to clear commanded positions
and counters. If opening the port does not reset your board, press EN just after
starting the Python test. Do not run this test against the handshake-only sketch.

In Windows PowerShell (replace COM5):

```powershell
py -m pip install pyserial
py C:\ArmLink-checkpoint3\test_combined.py --port COM5
```

## Hardware pass

The script must exit successfully and print every PASS line, ending with:

```text
PASS: watchdog disarmed and rotated session; targets held in software
PASS: checkpoint 3A; servo outputs disabled, no physical movement verified
```

State must show outputs_enabled=0 throughout. Commanded degrees start at
(90,90,90,90). An invalid last-joint target leaves all four unchanged and moves=0.
The valid centidegree tuple (9000,8500,10000,9000) produces commanded degrees
(90,85,100,90), moves=1. STOP disarms and the next MOVE is rejected. After
0.8 seconds without traffic, a new session opens disarmed, preserving targets
and moves=1, with timeouts=1. No physical movement is expected.

This proves physical ESP32 execution of authenticated dispatch, atomic target
validation, units/order, commanded state, STOP gating, and watchdog session
invalidation. The test leaves USB plugged in while stopping traffic; unplugging
USB can also remove ESP32 power, which is a different test.

## Protocol and adapter

F1 order: base GPIO18, shoulder GPIO19, elbow GPIO21, gripper GPIO22. Limits in
degrees: 30..150, 60..120, 45..135, 70..110. Validate all four values in
centidegrees against those limits before conversion or updates. Positive values
round to nearest integer degree for F1's integer Servo.write interface. Limits
are copied from F1's source, not claimed mechanically calibrated.

OPEN_SESSION uses checkpoint 2's ACK unchanged. Other commands return signed
ACK payload [command, active, armed, result], result=0 accepted or 1 rejected.
Malformed, stale-counter, or unauthenticated packets are silent. Authenticated
rejections do not advance counters or refresh the watchdog. Reply header counter
echoes the request. GET_STATE returns kind=0x81 with payload `<BBB4hII`:
active, armed, outputs_enabled, four commanded integer degrees, accepted move
count, watchdog timeout count. Session and counters are authenticated.

Software STOP preserves the session and targets and rejects MOVE until rearmed.
Timeout disarms, invalidates the session, and generates a fresh HELLO session.
All physical servo outputs remain disabled. The F1-confirmed policy is:

- STOP: disarm and detach servo outputs.
- Communication timeout: invalidate the session and disarm, but keep outputs
  attached at the last valid targets if already attached. Outputs detached by
  STOP remain detached; timeout must never reattach them.

Checkpoint 3B implements this policy through a mock backend only. Do not enable
physical servo outputs yet. Rate limiting, the physical servo backend, and
reattachment behavior after STOP remain later work.

## Checkpoint 3B: mock servo interface result

**Laptop checks passed; no new physical ESP32 or servo result is claimed.**
The combined sketch now calls `ServoBridge<MockServoBackend>` instead of
updating commanded positions directly. The backend records degree writes and
detach requests without using a servo library, attaching outputs, or touching
GPIO/PWM. STATE remains compatible with 3A and reports outputs_enabled=0.
Mock attachment is deliberately not reported as physical output enablement.

Verified:

- Valid MOVE invokes four writes in base, shoulder, elbow, gripper order, with
  nearest-integer degree conversion.
- Every joint's lower/upper out-of-range value causes zero writes and no target
  changes; inclusive limit values are accepted.
- Authenticated STOP reaches the backend's detach operation and disarms.
- Watchdog invalidation disarms and preserves mock attachment and last targets.
- Timeout after STOP leaves mock outputs detached. All 16 combinations of
  initial attachment flags are covered by the bridge test.
- ARM changes protocol state only; it does not attach or reattach outputs.

The full sketch simulation uses test-only injection of already-attached mock
channels to exercise hold behavior. No physical attachment code is introduced.
The confirmed policy does not yet define startup/reattachment; those decisions
must be settled with F1 before a powered test.

Run in WSL from the repository root:

```bash
g++ -std=c++11 -Wall -Wextra -Werror firmware/comms/test_servo_bridge.cpp -o /tmp/test_servo_bridge
/tmp/test_servo_bridge
python3 firmware/comms/test_combined_local.py
```

Expected final checks:

```text
PASS: 3B bridge order, rounding, limits, zero partial writes, STOP detach, timeout hold
PASS: 3B sketch dispatch reaches mock writes/detach; timeout holds attachment and targets
```

The existing 20 C checks and C/Python handshake simulation also passed after
the change. F1's servo sketch is unchanged. The previously generated 3A package
in /tmp is a snapshot and does not contain this update; use prepare_combined.py
with a new destination to package the current sources. No hardware upload is
required for this mock-only checkpoint.

## Testing-branch scope and handoff

This work on full-comms is protocol testing and mock integration, not F1's
physical servo implementation. Commit/handoff summary: checkpoint 2 passed on
hardware; checkpoint 3A passed on hardware on COM3; checkpoint 3B passed with
mocks. Physical outputs remain disabled, and F1's sketch is unchanged. The
untracked root comms/ folder is unrelated and must remain outside this commit.

F1 owns servo wiring/power, PWM, bench movement, startup and reattachment,
mechanical limits, and rate limiting. The arm is not assembled; individual
unloaded-servo testing can proceed under F1 without waiting for final arm
kinematic limits. The limits copied into this test are not final calibration.

F2's remaining protocol work includes ACK-aware host sending, heartbeat/failure
handling, and additional physical rejection/reconnect tests. These are follow-up
tasks, not completed claims or prerequisites to committing these test artifacts.
The physical servo backend has not been implemented or enabled. Joint testing
with F1 follows when their backend is ready, including physical STOP-detach and
timeout-hold verification. No further servo implementation is required to
complete the scope of this testing handoff.

## Laptop validation

```bash
python3 firmware/comms/test_combined_local.py
```

This compiles the actual sketch with mocked Arduino serial, time and random
APIs, links the real ArmLink C helpers, and runs the same Python test. It is a
laptop simulation, not an ESP32 build or hardware pass. Arduino CLI is not
available in the current WSL environment. Existing C checks and the handshake
simulation also passed after these additions.
