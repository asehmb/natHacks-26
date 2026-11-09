"""Fault-inject physical USB heartbeat ACKs into the real host. No MOVE/STOP sent."""
import argparse
import hmac
import time

import send_ik as host
from test_combined import Client
from test_rejections_usb import snapshot


class FaultSerial:
    """Wrap a serial port, altering only a verified real HEARTBEAT acceptance.

    Signed mutations use the bench key to test semantic validation independently
    of HMAC. They are synthetic replies, not genuine ESP32 rejections.
    """
    def __init__(self, serial, mode):
        self.serial = serial
        self.mode = mode
        self.pending = bytearray()
        self.output = bytearray()
        self.commands = []
        self.injected = False
        self.expected = None
        self.blocked_writes = 0

    def write(self, data):
        raw = host.cobs_decode(data)
        kind = raw[4]
        if kind in (host.MOVE, host.STOP):
            self.blocked_writes += 1
            raise OSError('Test guard: MOVE/STOP must never reach USB')
        self.commands.append(kind)
        if kind == host.HEARTBEAT:
            self.expected = (raw[7:23], int.from_bytes(raw[23:27], 'little'))
        return self.serial.write(data)

    def flush(self):
        self.serial.flush()

    def read_until(self, delimiter, size):
        result = bytearray()
        deadline = time.monotonic()+0.1
        while len(result) < size and time.monotonic() < deadline:
            byte = self.read(1)
            result.extend(byte)
            if byte == delimiter:
                break
        return bytes(result)

    def read(self, count):
        if self.output:
            data = bytes(self.output[:count])
            del self.output[:count]
            return data
        byte = self.serial.read(1)
        if not byte:
            return b''
        if byte != b'\0':
            self.pending.extend(byte)
            if len(self.pending) > 256:
                raise OSError('Unexpected oversized physical reply')
            return b''
        if not self.pending:
            return byte
        frame = bytes(self.pending)+byte
        self.pending.clear()
        try:
            raw = host.cobs_decode(frame)
        except ValueError:
            self.output.extend(frame)  # Boot noise is handled by HELLO reader.
            return b''
        if (self.expected and len(raw) >= 31 and raw[4] == 0x80 and
                raw[27] == host.HEARTBEAT and not self.injected):
            host.verify_ack(raw, *self.expected, host.HEARTBEAT)
            self.injected = True
            if self.mode == 'missing':
                return b''
            if self.mode == 'malformed':
                self.output.extend(b'\x05\x01\0')
                return b''
            changed = bytearray(raw)
            offset = {'rejected': 30, 'bad-hmac': -1, 'wrong-session': 7,
                      'wrong-counter': 23, 'wrong-command': 27,
                      'wrong-active': 28, 'wrong-armed': 29}[self.mode]
            changed[offset] ^= 1
            if self.mode != 'bad-hmac':
                changed[-32:] = hmac.digest(host.KEY, changed[:-32], 'sha256')
            frame = host.cobs_encode(bytes(changed))
        self.output.extend(frame)
        return b''


def run(serial):
    probe = Client(serial)
    probe.open()
    baseline = snapshot(probe)
    if baseline[1] != 0:
        raise ValueError('Preflight must be disarmed')
    print('PASS: preflight outputs_enabled=0; USB guard blocks all MOVE/STOP writes')
    # Allow the preflight session to expire before a real host opens a new one.
    time.sleep(0.6)
    cases = {'missing': 'timeout', 'malformed': 'Invalid COBS',
             'bad-hmac': 'HMAC', 'rejected': 'rejected',
             'wrong-session': 'session mismatch', 'wrong-counter': 'counter mismatch',
             'wrong-command': 'command mismatch', 'wrong-active': 'flags mismatch',
             'wrong-armed': 'flags mismatch'}
    for mode, reason in cases.items():
        transport = FaultSerial(serial, mode)
        client = host.ArmLinkSession(transport)
        try:
            client.open()
            session = client.session
            client.arm()
            deadline = time.monotonic()+2
            while client.usable and time.monotonic() < deadline:
                time.sleep(0.01)
            if not transport.injected or client.usable or reason not in (client.failure or ''):
                raise ValueError(f'{mode}: expected injected failure; got {client.failure}')
            client.close()  # Join the worker; never writes STOP.
            sent = list(transport.commands)
            for action in (lambda: client.move(b'\0'*8), client.arm, client.stop, client.open):
                try:
                    action()
                except host.SessionError:
                    pass
                else:
                    raise ValueError('Failed session unexpectedly accepted a command')
            if transport.commands != sent or transport.blocked_writes:
                raise ValueError('Host attempted traffic after heartbeat failure')
            if sent != [host.OPEN_SESSION, host.ARM, host.HEARTBEAT]:
                raise ValueError('Unexpected host command sequence: '+str(sent))
            print(f'PASS: {mode}: host invalidated session; no later commands or automatic STOP')
        finally:
            client.close()
        # Probe after watchdog expiry: verify actual ESP32 session rotation.
        probe.open()
        recovered = snapshot(probe)
        if (probe.session == session or recovered[1] != 0 or
                recovered[3:8] != baseline[3:8] or recovered[8] <= baseline[8]):
            raise ValueError('Receiver failed to timeout/recover without target changes')
        baseline = recovered
        time.sleep(0.6)  # Expire this disarmed probe session before next case.
    print('PASS: USB host heartbeat fault checkpoint; outputs disabled; no MOVE or STOP sent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    args = parser.parse_args()
    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=0.01, write_timeout=0.15) as port:
            run(port)
    except (OSError, ValueError, TimeoutError, host.SessionError, serial.SerialException) as error:
        raise SystemExit('FAIL: '+str(error))


if __name__ == '__main__':
    main()
