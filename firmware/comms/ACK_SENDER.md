# ACK-aware host sender

Implemented on f2/ack-heartbeat after the testing handoff. PR #12's final checks
passed for Python 3.10, 3.12, 3.14 and vision; that PR is merged.

send_ik.py now waits for authenticated acceptance of OPEN_SESSION before ARM,
ARM before MOVE, and each MOVE/HEARTBEAT/STOP before proceeding. Each ACK checks
HMAC-SHA256, magic/version/direction/type, exact payload length, current session,
echoed request counter, acknowledged command, active/armed flags and success
result. OPEN_SESSION retains its 3-byte payload; subsequent ACKs have 4 bytes
as documented in COMBINED_TEST.md. A queued signed HELLO for the same session
may be skipped while awaiting ACK; a changed session aborts the sequence.

Missing ACKs time out after 300 ms (below the firmware's 500 ms watchdog).
Invalid/mismatched/rejected ACKs and incomplete writes abort immediately. No
retries or subsequent commands, including STOP, are sent after a failure;
the receiver watchdog applies timeout policy. This deliberately avoids implying
that an unacknowledged command did not execute. ACK acceptance does not prove
physical motion. Serial read timeouts can add some latency to the deadline.

The existing one-second heartbeat phase remains a one-shot test, not a
continuous background heartbeat service. Its heartbeats and final STOP now
require ACKs. Hardware mode still requires --send-motion. Dry-run stays offline.

Laptop checks:

```bash
python3 -m unittest discover -s firmware/comms -p test_send_ik_ack.py
python3 firmware/comms/test_combined_local.py
python3 firmware/comms/test_handshake_local.py
python3 firmware/comms/send_ik.py --dry-run --angles 90 85 100 90
```

The ACK unit tests cover valid exchanges, signed session/counter/command/header
and state mismatches, rejection, tampering, truncation, missing ACKs, fragmented
reads, queued HELLO and short writes. The combined simulation additionally runs
the production sender against the actual receiver sketch with mocked Arduino
APIs and the real C authentication helpers. These are laptop tests only; no
serial port is opened, physical command sent, or servo output enabled.

Remaining: continuous heartbeat lifecycle, expanded physical rejection/reconnect
tests, and joint integration with F1's real servo backend. The earlier
test_doc.md handoff describes the pre-ACK sender; this document supersedes its
statements about missing host ACK verification.
