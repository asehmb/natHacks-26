# USB rejection and reconnect checkpoint

Use the already-uploaded outputs-disabled combined_only firmware. Leave servo
power disconnected and close Arduino Serial Monitor/Plotter. No firmware change
or physical output enablement is needed. The test sends ARM, STOP, GET_STATE,
and invalid packets, but **no MOVE**. It checks outputs_enabled=0 before ARM.

Copy these current files from firmware/comms to C:\ArmLink-checkpoint3:

- test_rejections_usb.py
- test_combined.py
- test_handshake.py
- send_ik.py

Run in Windows PowerShell (COM3 was the last observed port):

```powershell
py C:\ArmLink-checkpoint3\test_rejections_usb.py --port COM3
```

All PASS lines must appear, ending with:

```text
PASS: USB rejection/reconnect checkpoint; outputs disabled; no MOVE sent
```

Checks include invalid HMAC, signed wrong session, malformed/truncated COBS,
oversized frames and recovery at delimiter, and replayed STOP after a fresh ARM.
Silence alone is insufficient: authenticated state afterward must still respond,
with unchanged targets and move count and expected armed state. Reusing the
invalid packet's candidate counter for a valid state request also checks that
rejection did not consume that counter.

During an 0.8-second stream of invalid heartbeat packets, no valid traffic is
sent. The receiver must timeout once, issue a fresh session, remain disarmed
after OPEN_SESSION, retain targets/move count, and reject an ARM carrying the
prior session. Explicit ARM in the fresh session must work, followed by STOP.

Laptop validation: test_combined_local.py compiles and runs the actual sketch
with mocked Arduino APIs, then executes this same test. It passed, including
an assertion of zero additional mock servo writes. This is not a physical USB
pass.

## Recorded physical result

The user supplied the complete passing console output from the physical ESP32
USB run (COM3 in the provided command). All checks above passed: authenticated
session, outputs-disabled preflight, invalid HMAC without counter consumption,
signed wrong session, malformed COBS, oversized-frame recovery, replayed STOP,
watchdog expiry under invalid traffic, old-session ARM rejection, and explicit
ARM/STOP in the fresh session. The final line was:

```text
PASS: USB rejection/reconnect checkpoint; outputs disabled; no MOVE sent
```

This is physical receiver protocol verification. Outputs remained disabled and
no MOVE was sent. The servos provide no position feedback; targets and state are
software commands, not measurements. No servo movement, detachment, or physical
holding behavior is verified by this result.

This checkpoint tests receiver rejection and reconnect behavior. It does not
inject corrupt ACKs into the production host or physically verify host heartbeat
ACK-failure handling; those host failure cases currently have laptop coverage.
