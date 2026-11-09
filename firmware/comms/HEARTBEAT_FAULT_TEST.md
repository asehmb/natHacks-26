# Physical USB host heartbeat ACK-failure checkpoint

Use the same outputs-disabled combined_only firmware. No upload is needed.
Keep servo power disconnected and close Serial Monitor/Plotter.

Copy these current files from firmware/comms to C:\ArmLink-checkpoint3:

- test_heartbeat_faults_usb.py
- test_rejections_usb.py
- test_combined.py
- test_handshake.py
- send_ik.py

Run in Windows PowerShell:

```powershell
py C:\ArmLink-checkpoint3\test_heartbeat_faults_usb.py --port COM3
```

This test communicates with the physical ESP32 through the real host session.
A test-only transport wrapper first verifies the ESP32's real HEARTBEAT ACK,
then drops or changes that reply before the production host validates it.
Cases: missing ACK, malformed COBS, bad HMAC, synthetic rejection, wrong session,
counter, command, active flag, and armed flag. Semantic mutations are re-signed
with the development key to exercise field validation independently of HMAC.
Synthetic rejected/mismatched replies are injected by the harness; they are not
claims that the ESP32 generated those replies or that the USB link corrupted them.

Preflight requires an authenticated STATE reporting outputs_enabled=0. The
wrapper guards against forwarding any MOVE or STOP to USB. Each case sends
only OPEN_SESSION, ARM, and one HEARTBEAT through the production host; the host
must then invalidate its session and refuse later move/arm/stop/open attempts.
Attempted MOVE after failure must be blocked by the host before reaching even
the transport guard. No automatic STOP is allowed because it would request
detachment rather than the agreed timeout-hold behavior.

After failure the unmodified ESP32 must timeout and issue a fresh session.
Independent authenticated state probes verify disarmed recovery without target
or move-count changes. Probe sessions are deliberately allowed to expire between
cases. Expect multiple session-open messages and FAIL lines from the production
host: these are expected injected faults when followed by the matching PASS.
Run time is roughly 15–25 seconds, depending on serial timing.

Required final line:

```text
PASS: USB host heartbeat fault checkpoint; outputs disabled; no MOVE or STOP sent
```

This tests host failure handling with a physical receiver and controlled reply
injection. It does not test servo motion, actual attachment/holding, arbitrary
electrical faults, or long-duration reliability. Position state is commanded,
not measured; the servos provide no position feedback.

The same harness passed through test_combined_local.py against the real sketch
with mocked Arduino APIs; zero additional mock servo writes were verified.
That laptop pass is separate from the physical result below.

## Recorded physical USB result

The user supplied the complete Windows COM3 console output: **all nine fault
cases passed** with the physical ESP32 running outputs-disabled firmware.
Missing, malformed, bad-HMAC, synthetic rejection, wrong-session, wrong-counter,
wrong-command, wrong-active and wrong-armed heartbeat ACKs each invalidated the
production host session. Each expected FAIL diagnostic was followed by the
matching PASS confirming no later commands or automatic STOP. Fresh session
probes succeeded after receiver watchdog expiry. The final line was:

```text
PASS: USB host heartbeat fault checkpoint; outputs disabled; no MOVE or STOP sent
```

This is a physical USB exchange with controlled host-side reply injection,
not a claim that the firmware emitted corrupted or rejected heartbeat ACKs.
No servo output, motion, physical detachment or holding was tested.
