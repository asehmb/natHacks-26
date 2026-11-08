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

Persistent-session ACKs time out after 150 ms (below the 500 ms watchdog).
Invalid/mismatched/rejected ACKs and incomplete writes abort immediately. No
retries or subsequent commands, including STOP, are sent after a failure;
the receiver watchdog applies timeout policy. This deliberately avoids implying
that an unacknowledged command did not execute. ACK acceptance does not prove
physical motion. Serial read timeouts can add some latency to the deadline.

## Continuous session lifecycle

`ArmLinkSession` owns a background heartbeat worker and exclusive serial access.
`open()` reads an authenticated HELLO and verifies OPEN_SESSION ACK. It leaves
the host disarmed. `arm()` explicitly requests ARM and starts armed keepalive
behavior. While armed, HEARTBEAT is sent about every 250 ms since the last
command transmission, until explicit STOP, close, or any failed exchange. Other
accepted commands also refresh the receiver watchdog, so they reset the next
heartbeat deadline. The worker and application commands share a lock and one
monotonic counter; only one ACK is outstanding at a time. Heartbeats continue
while the application is idle, not just during a fixed one-second loop.

Each heartbeat requires the same authenticated, correlated acceptance as MOVE:
current session, counter, HEARTBEAT command, result=0, active=1 and armed=1.
150 ms ACK timeout plus the 250 ms interval leaves nominal margin under 500 ms.
Real serial reads are capped at 20 ms and writes at 150 ms. Scheduling and OS
serial buffering are not hard real-time guarantees; the receiver watchdog
remains authoritative. A locally observed 500 ms gap invalidates the session
before another command is sent.

Any rejected command aborts the sequence and invalidates the host session.
Any rejected, malformed, unauthenticated, mismatched or missing heartbeat ACK
does the same. `failure` records the command, counter and reason; later API calls
raise `SessionError` without writing. Rejections report the numeric result code;
the current wire format does not encode a more specific firmware rejection
reason. Field mismatches, HMAC errors, framing errors and ACK timeout are
reported distinctly. There are no automatic retries, reconnect, or ARM requests.

Recovery requires a new session object, fresh authenticated HELLO/OPEN_SESSION,
and explicit `arm()` before MOVE. A failed object cannot be reopened. No STOP
is sent on failure or `close()`: communication loss must use timeout-hold,
whereas STOP requests servo detachment. Closing stops and joins the worker
without closing the caller-owned serial port. Explicit successful `stop()`
ends armed heartbeats; a disarmed session will expire if no other traffic is
sent. Re-arm is explicit and must precede the local watchdog deadline or use a
fresh session.

The command-line tool still performs a finite one-second demonstration, now
using this persistent session lifecycle, and sends STOP only on successful
completion. Other applications can use `ArmLinkSession.open()/arm()/move()`
for arbitrary session duration, check failures with `check()`, and call
`stop()` only when detachment is intended. Always call `close()` in a finally
block. No physical backend or output enablement was added. Hardware mode still
requires --send-motion; dry-run stays offline.

Laptop checks:

```bash
python3 -m unittest discover -s firmware/comms -p test_send_ik_ack.py
python3 -m unittest discover -s firmware/comms -p test_heartbeat.py
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

The heartbeat tests cover repeated 250 ms keepalive, STOP/close cleanup, all
required failure cases, no later MOVE or automatic STOP after failure, and fresh
session plus explicit ARM recovery. The combined sketch simulation runs the
production sender long enough for repeated background heartbeats.

Remaining: physical verification of this host lifecycle, expanded physical
rejection/reconnect tests, and joint integration with F1's real servo backend. The earlier
test_doc.md handoff describes the pre-ACK sender; this document supersedes its
statements about missing host ACK verification.
