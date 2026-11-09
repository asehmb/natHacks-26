"""Timed outputs-disabled USB keepalive check. No MOVE commands."""
import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import statistics
import sys
import time

import send_ik as host
from test_combined import Client
from test_rejections_usb import snapshot


class GuardSerial:
    def __init__(self, serial):
        self.serial = serial
        self.sent = []

    def read(self, count):
        return self.serial.read(count)

    def read_until(self, delimiter, size):
        data = bytearray()
        deadline = time.monotonic()+0.1
        while len(data) < size and time.monotonic() < deadline:
            byte = self.read(1)
            data.extend(byte)
            if byte == delimiter:
                break
        return bytes(data)

    def write(self, frame):
        raw = host.cobs_decode(frame)
        if raw[4] == host.MOVE:
            raise OSError('Reliability test guard: MOVE prohibited')
        self.sent.append((raw[4], time.monotonic()))
        return self.serial.write(frame)

    def flush(self):
        self.serial.flush()


class MeasuredSession(host.ArmLinkSession):
    def __init__(self, serial):
        super().__init__(serial)
        self.latencies = []

    def _exchange(self, kind, payload=b''):
        start = time.monotonic()
        super()._exchange(kind, payload)
        if kind == host.HEARTBEAT:
            # Host exchange time includes serial write/flush, ACK receipt and validation.
            self.latencies.append(time.monotonic()-start)


def run(serial, seconds=300, report=print):
    if not math.isfinite(seconds) or seconds < 1:
        raise ValueError('Duration must be finite and at least one second')
    preflight = Client(serial)
    preflight.open()
    before = snapshot(preflight)
    if before[1] != 0:
        raise ValueError('Preflight must be disarmed')
    report('PASS: preflight outputs_enabled=0; no MOVE will be sent')
    time.sleep(0.6)  # Allow preflight session to expire before production host opens.
    transport = GuardSerial(serial)
    client = MeasuredSession(transport)
    try:
        client.open()
        # Snapshot immediately before ARM, using the same session/counter. The
        # host worker is disarmed and the lock gives this probe exclusive I/O.
        with client._lock:
            probe = Client(serial)
            probe.session, probe.counter = client.session, client.counter
            baseline = snapshot(probe)
            client.counter = probe.counter
        client.arm()
        start = time.monotonic()
        next_report = 30
        report(f'RUNNING: {seconds:g} seconds of armed keepalive; outputs disabled')
        while time.monotonic()-start < seconds:
            client.check()
            elapsed = time.monotonic()-start
            if elapsed >= next_report:
                with client._lock:
                    count = len(client.latencies)
                report(f'PROGRESS: elapsed={elapsed:.0f}s accepted_heartbeats={count}')
                next_report += 30
            time.sleep(min(0.05, max(0, seconds-elapsed)))
        client.check()
        elapsed = time.monotonic()-start
        client.stop()  # Success only; never issued by failure cleanup.
        client.close()
        probe = Client(serial)
        probe.session, probe.counter = client.session, client.counter
        final = snapshot(probe)
        if final[1] != 0 or final[3:] != baseline[3:]:
            raise ValueError('Final state changed targets/move count/timeouts or stayed armed')
        latency = client.latencies
        stamps = [stamp for kind, stamp in transport.sent if kind in (3, 6, 5)]
        gaps = [b-a for a, b in zip(stamps, stamps[1:])]
        if len(latency) < max(1, int(seconds/0.4)) or max(gaps, default=1) >= 0.5:
            raise ValueError('Insufficient heartbeats or host transmit gap reached 500 ms')
        result = {'requested_seconds': seconds, 'elapsed_seconds': round(elapsed, 3),
                  'accepted_heartbeats': len(latency),
                  'ack_exchange_median_ms': round(statistics.median(latency)*1000, 3),
                  'ack_exchange_max_ms': round(max(latency)*1000, 3),
                  'max_host_transmit_gap_ms': round(max(gaps)*1000, 3),
                  'session_rotations_during_run': 0, 'outputs_enabled': 0,
                  'move_commands_sent': 0, 'stop_acknowledged': True,
                  'receiver_timeout_count_unchanged': True}
        report('RESULT: '+json.dumps(result, sort_keys=True))
        report('PASS: heartbeat reliability run; STOP acknowledged; outputs disabled; no MOVE sent')
        return result
    finally:
        client.close()  # Failure/interrupt: no automatic STOP.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--log', type=Path, default=Path('heartbeat-reliability.log'))
    args = parser.parse_args()
    import serial
    console = sys.stdout
    report = lambda message: print(message, file=console, flush=True)
    try:
        with args.log.open('w', encoding='utf-8', buffering=1) as log:
            report('Detailed exchange log: '+str(args.log.resolve()))
            with redirect_stdout(log):
                with serial.Serial(args.port, 115200, timeout=0.01, write_timeout=0.15) as port:
                    run(port, args.seconds, report)
    except (OSError, ValueError, TimeoutError, host.SessionError, serial.SerialException) as error:
        raise SystemExit('FAIL: '+str(error)+'; no automatic STOP sent on failure')
    except KeyboardInterrupt:
        raise SystemExit('ABORTED: no automatic STOP sent; receiver watchdog handles timeout')


if __name__ == '__main__':
    main()
