# Five-minute heartbeat reliability checkpoint

Use the existing outputs-disabled combined_only firmware. Keep servo power
disconnected and Serial Monitor/Plotter closed. No firmware upload is needed.
Copy these current files from firmware/comms to C:\ArmLink-checkpoint3:

- test_heartbeat_reliability.py
- test_rejections_usb.py
- test_combined.py
- test_handshake.py
- send_ik.py

Run in Windows PowerShell:

```powershell
py C:\ArmLink-checkpoint3\test_heartbeat_reliability.py --port COM3 --seconds 300 --log C:\ArmLink-checkpoint3\heartbeat-reliability.log
```

Authenticated preflight requires outputs_enabled=0 and disarmed state. After
preflight session expiry, the production host opens a fresh session, explicitly
arms, and maintains background heartbeats for five minutes. A transport guard
prohibits MOVE. No targets change. Successful completion explicitly sends STOP
and verifies its ACK plus authenticated disarmed STATE, unchanged targets/move
count, and unchanged receiver watchdog timeout count during the measured run.

The console reports progress every 30 seconds. Full packet TX/ACK messages go
to the specified log (overwritten on each run). The RESULT line reports accepted
heartbeats (roughly 1,200), elapsed time, median/max heartbeat ACK exchange time,
and maximum host transmit gap. Exchange time includes writing, flushing,
reading, authentication and console/log printing; it is not pure wire latency.
Host timestamps do not measure exact ESP32 receive times. The final unchanged
receiver timeout count and successful same-session replies establish that the
watchdog did not expire during this run.

Pass requires all expected ACKs, sufficient repeated heartbeats, host transmit
gaps below 500 ms, stable receiver state, and accepted STOP. Required final line:

```text
PASS: heartbeat reliability run; STOP acknowledged; outputs disabled; no MOVE sent
```

Failures or Ctrl+C stop the worker without automatically sending STOP; the
receiver's communication watchdog applies the agreed timeout policy. Preserve
the detailed log and report the error. Do not rerun and discard an unexplained
failure. There are no automatic retries or reconnects during measurement.

test_combined_local.py passed a two-second smoke test of this harness against
the real receiver sketch with mocked Arduino APIs, including zero additional
mock servo writes. That short laptop test does
not establish five-minute reliability or physical USB results. The physical
five-minute result is recorded below.

## Recorded physical USB result

The user supplied the completed Windows COM3 run's console output: **PASS**.

| Metric | Observed result |
| --- | --- |
| Measured duration | 300.0 seconds |
| Authenticated heartbeat acceptances | 1,198 |
| Median heartbeat ACK exchange time | 15.0 ms |
| Maximum heartbeat ACK exchange time | 47.0 ms |
| Maximum host transmit gap | 266.0 ms |
| Receiver watchdog timeout count | Unchanged during measurement |
| Session rotations during measurement | 0 |
| Final STOP | Authenticated ACK accepted |
| Physical outputs enabled | 0 |
| MOVE commands sent | 0 |

The earlier run was interrupted when wires were disconnected and does not count
as a reliability pass. A subsequent connection attempt reported no authenticated
HELLO; the later complete run above established a fresh session and passed.
These interruptions remain part of the test history, not unexplained failures
erased by the successful run. The supplied console result is recorded here;
the Windows detailed log has not been independently inspected in WSL.

This establishes one successful five-minute outputs-disabled USB keepalive run,
not servo movement/holding, assembled-arm operation, or an indefinite reliability
guarantee. No servo position feedback is available.

No servo motion, physical attachment/detachment or holding is verified. Servos
have no position feedback; STATE reports commanded targets. Repeated successful
runs can provide more evidence but do not guarantee operation in every condition.
