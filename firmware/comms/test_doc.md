
# EasyBot MK2 — ESP32 ArmLink Integration Test Handoff

**Historical handoff:** host ACK verification has since been implemented; see
[ACK_SENDER.md](ACK_SENDER.md) for current sender behavior and laptop validation.
See COMBINED_TEST.md for recorded hardware results and the current mock backend.

**Scope:** Verify that Python kinematics commands reach the ESP32, are authenticated and decoded, and are handed to the servo-control layer. **Test communication with all servo power disconnected first.** This is a teammate handoff/test plan, not a claim that the ESP32 receiver is already integrated.

## 0. What is already working

On the Python machine, this command ran successfully:

```bash
python3 firmware/comms/send_ik.py --dry-run --xyz 0.20 0.0 0.15
```

Observed:

```text
Servo targets: {'base': 91.73, 'shoulder': 84.65, 'elbow': 103.68, 'gripper': 90}
MOVE payload: d523112180282823
```

This confirms the XYZ → IK → servo-angle mapping → MOVE payload path. It does **not** yet prove the packet was received or accepted by the ESP32, and does not prove the physical servos reach those angles. The MOVE values are hundredths of degrees, ordered **base, shoulder, elbow, gripper**.

## 1. Files to use and important gap

- `firmware/comms/send_ik.py`: Python sender, with `--dry-run`, `--angles`, `--xyz`, and `--send-motion`.
- `firmware/comms/armlink.c` and `armlink.h`: C protocol helpers for HMAC, MOVE decoding, sessions, timeouts, COBS.
- `firmware/servoWork/servo_actuation.ino`: existing Arduino sketch that receives **plain-text** commands such as `base 90`. **It does not currently receive binary ArmLink packets.**

**Receiver work required:** Your teammate must integrate a firmware serial receiver that assembles a COBS frame ending in `0x00`, calls the COBS decoder, validates/authenticates the packet with `armlink.c`, dispatches session/ARM/MOVE/STOP/HEARTBEAT messages, and forwards accepted MOVE targets to the servo-control logic. If using `armlink.c` in an Arduino sketch, arrange for correct compilation/linkage of C functions with C++ (`extern "C"` where appropriate), and confirm the required ESP32 mbedTLS headers are available. Do not simply upload the existing text-command `.ino` and expect this Python sender to work.

The reference sender uses the **development test key** `0x0B` repeated 32 times. Both sides must use the same key for this bench test. It is not a production security key.

## 2. Stage A — ESP32 communication only (NO servo wiring)

**Physical setup:**

- Laptop USB → ESP32 USB connector.
- **No external 5 V servo adapter connected.**
- **No servo power, servo signals or servo grounds connected for this first stage.**
- ESP32 on a dry, nonconductive surface. Do not use the previously melted breadboard or damaged wires.

**What the receiver should do:**

1. Initialize serial at **115200 baud** and initialize `ReceiverState`.
2. Generate a fresh 16-byte session ID and call `prepare_receiver_session()`.
3. Use `build_hello_packet()`, COBS-encode it, append `0x00`, and transmit it to the Python host when the host opens/resets the serial connection.
4. Read incoming bytes into a bounded buffer until the `0x00` delimiter; reject oversized or malformed frames.
5. Decode COBS; verify packet layout, HMAC, session ID and monotonic counter.
6. Dispatch messages using `receive_open_session()`, `receive_arm_or_stop()`, `receive_move()` and `receive_heartbeat()`.
7. For an accepted MOVE, **log or expose** the four targets without calling `Servo.write()` / `writeMicroseconds()`.
8. Call `check_receiver_timeout()` regularly; a timeout disarms and closes the session.

**Important:** The Python sender transmits binary data. Don't print arbitrary debug text onto the **same serial channel before HELLO**, because the host searches for a valid binary HELLO. Prefer a separate debug interface, or use a guarded log method / receiver-side diagnostic counters. The regular Arduino Serial Monitor cannot be open at the same time as the Python script on the same port.

**From the repository root on the laptop:**

```bash
python3 -m pip install pyserial
python3 firmware/comms/send_ik.py --dry-run --angles 90 85 100 90
```

Confirm the offline test runs. Locate the board's actual port (`/dev/cu.usbserial-10` was the previous Mac port, but may have changed) and run, **with servo actuation disabled in firmware**:

```bash
python3 firmware/comms/send_ik.py \
  --port /dev/cu.usbserial-10 \
  --angles 90 85 100 90 \
  --send-motion
```

Even with servos unplugged, `--send-motion` sends an actual **ARM and MOVE** protocol sequence. It is safe for this test only if the receiver has **all physical actuator output disabled**. Do not connect motor power during this phase.

### Expected sequence and observations

| Order | Sender / receiver action | What to verify |
|---|---|---|
| 1 | ESP32 sends authenticated `HELLO` | Host prints `RX valid HELLO. Session: ...` |
| 2 | Host sends `OPEN_SESSION`, counter 1 | ESP32 accepts; `session_active=true` |
| 3 | Host sends `ARM`, counter 2 | ESP32 accepts; `armed=true` (physical outputs still disabled) |
| 4 | Host sends `MOVE`, counter 3 | ESP32 accepts and decodes `[9000, 8500, 10000, 9000]` |
| 5 | Host sends HEARTBEAT every ~100 ms for ~1 s | Valid counters; receiver remains alive |
| 6 | Host sends `STOP` | ESP32 disarms; no servo movement |

An accepted MOVE should correspond to these *commanded* angles: **base=90°, shoulder=85°, elbow=100°, gripper=90°**. The payload should be `2823342110272823` (8 bytes, little-endian signed `int16` values).

**Pass criteria:** Valid HELLO is received, OPEN_SESSION and ARM accepted, exactly the correct MOVE targets decoded, heartbeat accepted, STOP disarms, and **no actuators move**.

**Important limitation:** Current `send_ik.py` prints transmitted packet lines but does **not** wait for or verify ESP32 ACK replies. `TX` output alone means **bytes were written**, not that the receiver accepted them. Use receiver-side instrumentation to confirm acceptance. If there is no valid HELLO, the sender times out without sending motion.

## 3. Stage B — Actual XYZ from kinematics (still NO servo power)

Once the direct-angle test passes, run:

```bash
python3 firmware/comms/send_ik.py \
  --port /dev/cu.usbserial-10 \
  --xyz 0.20 0.0 0.15 \
  --send-motion
```

Expected **decoded MOVE targets** in hundredths of a degree:

```text
base:      9173
shoulder:  8465
elbow:    10368
gripper:   9000
```

Expected MOVE payload hex: `d523112180282823`.

**Pass criteria:** The receiver reports the exact values above in the same joint order. Report **accepted by receiver**, not **servo reached target**.

## 4. Stage C — Rejection and safety tests (still NO servo power)

Run these tests at the receiver/unit-test layer as applicable. Start a fresh session whenever a failed packet or timeout has invalidated the current one.

- Send a MOVE before ARM → reject.
- Alter one byte of the packet without recomputing HMAC → reject.
- Repeat a previously accepted packet with the same counter → reject.
- Use the wrong session ID → reject.
- Send a malformed or oversized COBS frame → reject, don't overflow the buffer.
- Stop heartbeats for **at least 500 ms** → receiver disarms via timeout.
- Send STOP → receiver becomes disarmed and MOVE no longer accepted until re-armed through a valid session.
- Out-of-range angles → **reject before any actuation**. Don't rely only on Python range checks; check again in ESP32 firmware.

**Pass criteria:** No rejected command reaches the servo-writing function, and communication loss causes a safe, explicitly defined stop behavior. Note: software disarming a hobby servo is not a substitute for a physical emergency power disconnect.

## 5. Stage D — ONE physical servo, only AFTER electrical inspection

**Do not wire the entire servo chain for the first packet test.** Your earlier DS3218MG test caused the breadboard ground rail to smoke. Before any motor is powered, have a venue mentor inspect the adapter, affected servo, replacement wiring and connectors; use a multimeter to verify the 5 V output polarity, voltage and lack of shorts. Do not reuse melted breadboard contacts or heat-damaged leads. Obtain suitable overcurrent protection and a DC-rated power disconnect. If these checks or protections cannot be completed, **stop at the packet-only demonstration**.

After the inspection passes, test **one unloaded MG90S or MG996R**, not the DS3218MG shoulder first:

| Connection | For J3 gripper MG90S test |
|---|---|
| Servo signal | ESP32 GPIO 22 |
| Servo + | External regulated 5 V through properly rated positive distribution, fuse and disconnect |
| Servo − | Directly to properly rated negative distribution |
| ESP32 ground | Low-current reference to negative distribution; may pass through an intact breadboard rail **only if motor return current never does** |
| ESP32 power | Laptop USB |

Keep all other servos physically disconnected and disable their channels in firmware. Use rated, correctly terminated power and ground leads; do not clamp servo pins or undersized wire into connectors. Avoid wiring the servo power return through a breadboard. Keep the servo horn unloaded and clear.

First send an unchanged/centered position. After observing stable supply and no overheating, try **small changes only** (for example, gripper 90 → 85 → 95 → 90) **within verified mechanical limits**. Your current Python sender sends all four joint targets together, so only the physically tested channel should be enabled in the firmware. Do not assume that its placeholder angular limits are the arm's calibrated safe limits.

**Physical test pass criteria:** Only the enabled servo responds, direction is as expected, no violent twitching, no reset, no buzzing/stalling, no warm or loose connectors, and emergency power disconnect works.

**Stop immediately** if wiring heats, smoke/smell occurs, the servo stalls, the ESP32 repeatedly resets, or the supply sags substantially. Disconnect external servo power first. Do not work on energized connectors.

## 6. Known integration issues to fix before full-arm movement

1. **Receiver not yet integrated:** Original `servo_actuation.ino` parses text lines; ArmLink uses COBS-framed binary data. They need an explicit adapter between packet acceptance and servo commands.
2. **Joint order:** Index 0 base (GPIO 18), 1 shoulder (GPIO 19), 2 elbow (GPIO 21), 3 gripper (GPIO 22). Both Python and ESP32 must agree.
3. **Units:** MOVE contains **centidegrees**. Convert `int16` → degrees (divide by 100.0) before commanding servos. Apply per-joint calibration, directions, and limits **after** decoding and **before** actuation.
4. **Bug in existing Arduino sketch:** `positions[id] = angle` occurs before validating the limits. Remove that premature update so rejected commands cannot change the tracked target.
5. **Channel enable flags:** Existing `.ino` enables all four joints. Keep actuation disabled for Stages A–C, then enable **one** tested channel only in Stage D.
6. **ACK verification:** Host script does not wait for ACK. Receiver-side validation/logging is required until ACK parsing is implemented.
7. **Timeout:** `armlink.c` uses a 500 ms timeout. The sender sends heartbeat for ~1 second and then STOP; it is **not** yet a continuous trajectory-streaming controller.
8. **No physical feedback:** An accepted target isn't a measured servo angle. Standard hobby servos here do not provide actual joint-position feedback to the code.
9. **Power budget:** The 5 V / 10 A supply must be verified under load. Multiple servos, especially the shoulder, may demand more current than the supply or harness can safely deliver.

## 7. Teammate report template

```text
Board/port:
Firmware version or commit:
Receiver integrated? yes/no
Stage A: HELLO / OPEN_SESSION / ARM / MOVE / HEARTBEAT / STOP:
Decoded direct-angle targets:
Stage B decoded XYZ targets:
Stage C rejection tests:
Servo power disconnected for A-C? yes/no
Electrical inspection and protection verified? yes/no
Stage D (if performed): which single servo / GPIO:
Any unexpected reset, heating, or rejected commands:
Next blocker:
```

**Final success definition for this handoff:** The ESP32 correctly accepts and interprets kinematics-derived ArmLink MOVE packets while all motors are disabled. Only after an electrical check and safe single-servo test should the teammate connect packet acceptance to physical actuation.
