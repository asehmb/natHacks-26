"""Checkpoint 3A: signed commands with firmware servo output disabled."""
import argparse
import hmac
import struct
import time
from send_ik import HEADER, KEY, packet, cobs_encode, cobs_decode, unpack_authenticated_hello
from test_handshake import verify_ack


def frames(ser, seconds):
    deadline = time.monotonic()+seconds
    pending = bytearray()
    dropping = False
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b'\0':
            if pending and not dropping:
                try:
                    raw = cobs_decode(bytes(pending)+b)
                except ValueError:
                    raw = None
                if raw is not None:
                    yield raw
            pending.clear()
            dropping = False
        elif not dropping:
            pending.extend(b)
            if len(pending) > 256:
                pending.clear()
                dropping = True


class Client:
    def __init__(self, ser):
        self.ser = ser
        self.counter = 0
        self.session = None

    def open(self):
        for raw in frames(self.ser, 10):
            try:
                session = unpack_authenticated_hello(raw)
            except ValueError:
                continue
            self.session = session
            self.counter = 0
            break
        else:
            raise TimeoutError('No authenticated HELLO')
        raw = self.request(2)
        verify_ack(raw, self.session, 1)
        print('PASS: authenticated session opened, disarmed')

    def request(self, kind, payload=b''):
        self.counter += 1
        outgoing = cobs_encode(packet(kind, self.session, self.counter, payload))
        if self.ser.write(outgoing) != len(outgoing):
            raise OSError('Incomplete write')
        self.ser.flush()
        for raw in frames(self.ser, 2):
            if len(raw) < 59:
                continue
            magic, version, direction, reply_kind, size, session, counter = HEADER.unpack(raw[:27])
            if (magic, version, direction, session, counter) != (
                    b'AR', 1, 1, self.session, self.counter):
                continue
            if len(raw) != 59+size or not hmac.compare_digest(
                    raw[-32:], hmac.digest(KEY, raw[:-32], 'sha256')):
                raise ValueError('Invalid authenticated reply')
            if reply_kind != (0x81 if kind == 7 else 0x80):
                raise ValueError('Unexpected reply kind')
            return raw
        raise TimeoutError('No matching authenticated reply')

    def ack(self, kind, payload=b'', rejected=False, armed=False):
        raw = self.request(kind, payload)
        if raw[27:-32] != bytes([kind, 1, int(armed), int(rejected)]):
            raise ValueError('Unexpected ACK flags/result')

    def state(self, armed, positions, moves, timeouts=None):
        raw = self.request(7)
        values = struct.unpack('<BBB4hII', raw[27:-32])
        if values[:3] != (1, int(armed), 0):
            raise ValueError('State flags mismatch; outputs must be disabled')
        if values[3:7] != tuple(positions) or values[7] != moves:
            raise ValueError('Commanded positions or move count mismatch: '+str(values))
        if timeouts is not None and values[8] != timeouts:
            raise ValueError('Timeout count mismatch')
        print('STATE commanded_degrees='+str(values[3:7])+
              f' armed={int(armed)} outputs_enabled=0 moves={moves} timeouts={values[8]}')
        return values[8]


def run(ser):
    client = Client(ser)
    client.open()
    initial = (90, 90, 90, 90)
    old_timeouts = client.state(False, initial, 0)
    move = struct.pack('<4h', 9000, 8500, 10000, 9000)
    client.ack(4, move, rejected=True)
    client.state(False, initial, 0)
    print('PASS: disarmed MOVE rejected')
    client.ack(3, armed=True)
    # Invalid LAST joint with changed first three: prove no partial application.
    client.ack(4, struct.pack('<4h', 9100, 8600, 10100, 11001), rejected=True, armed=True)
    client.state(True, initial, 0)
    print('PASS: invalid gripper rejected; all four targets unchanged')
    client.ack(4, move, armed=True)
    target = (90, 85, 100, 90)
    client.state(True, target, 1)
    print('PASS: MOVE order and centidegree conversion')
    client.ack(5)
    client.ack(4, struct.pack('<4h', 9200, 8700, 10200, 9100), rejected=True)
    client.state(False, target, 1)
    print('PASS: STOP disarmed; subsequent MOVE rejected')
    client.ack(3, armed=True)
    print('Waiting 0.8 s without traffic to exercise the 500 ms watchdog...')
    time.sleep(0.8)
    old_session = client.session
    client.open()
    if old_session == client.session:
        raise ValueError('Watchdog did not rotate session')
    client.state(False, target, 1, old_timeouts+1)
    print('PASS: watchdog disarmed and rotated session; targets held in software')
    print('PASS: checkpoint 3A; servo outputs disabled, no physical movement verified')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', required=True)
    a = p.parse_args()
    import serial
    try:
        with serial.Serial(a.port, 115200, timeout=0.05, write_timeout=1) as ser:
            run(ser)
    except (OSError, ValueError, TimeoutError, serial.SerialException) as error:
        raise SystemExit('FAIL: '+str(error))


if __name__ == '__main__':
    main()
