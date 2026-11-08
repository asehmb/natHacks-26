# Checkpoint 2: handshake only

## Recorded physical result

User reported a physical ESP32 pass: authenticated HELLO, OPEN_SESSION counter
1, and authenticated ACK with the matching session and counter, active=1 and
armed=0. No ARM or MOVE was sent. This is a hardware protocol result, not a
servo movement result.

Keep servo power disconnected. F1's sketch stays separate. No ARM or MOVE is
sent or dispatched, and the sketch contains no servo output code.

## Prepare in WSL

From the repository root:

```bash
python3 firmware/comms/prepare_handshake.py --output /tmp/armlink-checkpoint2/handshake_only
```

The destination must not already exist; choose another parent directory to
regenerate. The generated folder contains handshake_only.ino, armlink.c and
armlink.h. This copies the current helpers, avoiding stale duplicate sources in
the repo. Do not upload the source .ino alone.

Using Windows Explorer's WSL filesystem view, copy the generated handshake_only
folder into C:\ArmLink-checkpoint2. Also copy test_handshake.py and send_ik.py from
firmware/comms into C:\ArmLink-checkpoint2. send_ik.py is imported for packet
helpers; its motion-sending main function is not executed.

## Upload on Windows

Open C:\ArmLink-checkpoint2\handshake_only\handshake_only.ino in Arduino IDE.
Select the same ESP32 board and COM port used for checkpoint 1, then upload.
Uses the ESP32 board package's mbedTLS and esp_system headers; no ESP32Servo
library is needed. Close Serial Monitor and Serial Plotter before running Python.
The firmware uses binary serial, so readable Serial Monitor output is not expected.

## Run in Windows PowerShell

```powershell
py -m pip install pyserial
py C:\ArmLink-checkpoint2\test_handshake.py --port COM5
```

Replace COM5 with the observed port. Windows Python opens Windows COM directly;
do not run this command using WSL Python.

Expected physical pass (both session values must match):

```text
RX authenticated HELLO session=<32 hex digits>
TX OPEN_SESSION counter=1
RX authenticated ACK session=<same 32 hex digits> counter=1 command=OPEN_SESSION session_active=1 armed=0
PASS: USB handshake verified; no ARM or MOVE sent
```

This proves physical USB exchange, C acceptance of OPEN_SESSION, and Python
verification of the ACK's HMAC, session, echoed counter and disarmed state.
It does not prove motion, state reporting, or physical watchdog behavior.

HELLO repeats every 250 ms while waiting. A fresh random session is generated
at boot and after the existing 500 ms communication watchdog expires. The test
exits after ACK and sends no heartbeats; session expiry afterward is expected.
Oversized input is discarded through the next delimiter. Only accepted
OPEN_SESSION generates an ACK; duplicate or invalid packets are silent.

## ACK format

Existing 27-byte header and 32-byte HMAC-SHA256, COBS framed with zero delimiter.
Direction=1, kind=0x80, payload length=3, current session ID, header counter=the
accepted OPEN_SESSION counter (1). Payload: `[0x02, 0x01, 0x00]`, meaning
OPEN_SESSION accepted, session active, disarmed. This ACK counter is a request
correlation value, not an independent device transmit sequence. The bench key
remains 0x0B repeated 32 times.

## Laptop validation (no physical board)

```bash
gcc -std=c11 -Wall -Wextra -Werror firmware/comms/armlink.c firmware/comms/packet_test.c -lmbedcrypto -o /tmp/armlink_packet_test
/tmp/armlink_packet_test
python3 firmware/comms/send_ik.py --dry-run --angles 90 85 100 90
python3 firmware/comms/send_ik.py --dry-run --xyz 0.20 0.0 0.15
python3 firmware/comms/test_handshake_local.py
```

The final test uses simulated serial with the real C helpers compiled as a shared
library. Its printed USB PASS line is simulated, not a hardware observation.
It checks the actual C/Python handshake, exactly one OPEN_SESSION transmission,
ACK semantic mismatches with valid signatures, tampering, truncation and replay.
