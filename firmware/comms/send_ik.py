

"""EasyBot MK2 host sender for firmware/comms/armlink.c.

Packet-only test: python3 firmware/comms/send_ik.py --dry-run --angles 90 85 100 90
Hardware: python3 firmware/comms/send_ik.py --port /dev/cu.usbserial-10 --angles 90 85 100 90 --send-motion
Requires pyserial only for hardware mode. The ESP32 must implement HELLO and receive
OPEN_SESSION, ARM, MOVE, HEARTBEAT and return authenticated ACKs.
"""
import argparse
import hmac
import math
import struct
import sys
import time
import threading
from pathlib import Path

HEADER = struct.Struct('<2sBBBH16sI')
KEY = bytes([0x0B]) * 32  # Existing firmware TEST KEY, not secure for deployment.
HELLO, OPEN_SESSION, ARM, MOVE, STOP, HEARTBEAT = 1, 2, 3, 4, 5, 6
JOINT_NAMES = ('base', 'shoulder', 'elbow', 'gripper')
# Temporary conservative *command* limits. Replace after mechanical calibration.
waypoints = [
    (0.20,  0.00, 0.15),  # Center
    (0.183, 0.10, 0.15),  # Left 8 cm
    (0.20,  0.00, 0.15),  # Center
    (0.183, -0.10, 0.15), # Right 8 cm
    (0.20,  0.00, 0.15),  # Center
]
LIMITS = (
    (0, 180),    # Base
    (20, 160),   # Shoulder
    (15, 130),   # Elbow
    (0, 180),    # Gripper
)
COMMAND_NAMES = {2: 'OPEN_SESSION', 3: 'ARM', 4: 'MOVE', 5: 'STOP', 6: 'HEARTBEAT'}


class SessionError(RuntimeError):
    """The host cannot use this session again."""


def cobs_encode(data):
    out = bytearray(b'\x00')
    code_location, code = 0, 1
    for value in data:
        if value == 0:
            out[code_location] = code
            code_location = len(out)
            out.append(0)
            code = 1
        else:
            out.append(value)
            code += 1
            if code == 255:
                out[code_location] = code
                code_location = len(out)
                out.append(0)
                code = 1
    out[code_location] = code
    return bytes(out) + b'\x00'


def cobs_decode(frame):
    if not frame or frame[-1] != 0 or 0 in frame[:-1]:
        raise ValueError('Expected exactly one zero-terminated COBS frame')
    encoded = frame[:-1]
    result = bytearray()
    cursor = 0
    while cursor < len(encoded):
        block = encoded[cursor]
        end = cursor + block
        if block == 0 or end > len(encoded):
            raise ValueError('Invalid COBS block')
        result.extend(encoded[cursor + 1:end])
        if block != 255 and end < len(encoded):
            result.append(0)
        cursor = end
    return bytes(result)


def packet(kind, session, counter, payload=b''):
    if len(session) != 16 or not 0 <= counter <= 0xFFFFFFFF:
        raise ValueError('Invalid session or counter')
    head = HEADER.pack(b'AR', 1, 0, kind, len(payload), session, counter)
    body = head + payload
    return body + hmac.digest(KEY, body, 'sha256')


def unpack_authenticated_hello(raw):
    if len(raw) != 59:
        raise ValueError('HELLO must contain 59 bytes')
    magic, version, direction, kind, length, session, counter = HEADER.unpack(raw[:27])
    if (magic, version, direction, kind, length, counter) != (b'AR', 1, 1, HELLO, 0, 0):
        raise ValueError('Unexpected HELLO header')
    expected = hmac.digest(KEY, raw[:-32], 'sha256')
    if not hmac.compare_digest(raw[-32:], expected):
        raise ValueError('HELLO HMAC invalid')
    return session


def payload_for_angles(angles):
    if len(angles) != 4:
        raise ValueError('Expected four joint targets')
    values = []
    for name, angle, (lower, upper) in zip(JOINT_NAMES, angles, LIMITS):
        if not math.isfinite(angle) or not lower <= angle <= upper:
            raise ValueError(f'{name}: angle {angle} outside temporary {lower}..{upper} degree limits')
        values.append(round(angle * 100))
    return struct.pack('<4h', *values)


def angles_from_xyz(xyz, gripper):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'Kinematics'))
    from kinematics import ik, joint_to_servo, JointLimitError
    q = ik(*xyz)
    if q is None:
        raise ValueError(f'Unreachable XYZ target: {xyz}')
    try:
        arm_servo_angles = joint_to_servo(q)
    except JointLimitError as err:
        raise ValueError(f'IK servo mapping rejected: {err}') from err
    return [*arm_servo_angles, gripper]


def wait_hello(ser, deadline_seconds=5.0):
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        frame = ser.read_until(b'\x00', 256)
        if not frame or frame[-1:] != b'\x00':
            continue
        try:
            return unpack_authenticated_hello(cobs_decode(frame))
        except ValueError:
            continue
    raise TimeoutError('No valid ArmLink HELLO received. Receiver must emit HELLO after reset.')


def send(ser, kind, session, counter, payload=b''):
    frame = cobs_encode(packet(kind, session, counter, payload))
    if ser.write(frame) != len(frame):
        raise OSError('Incomplete serial write')
    ser.flush()
    print(f'TX kind=0x{kind:02x} counter={counter} bytes={len(frame)}')


def verify_ack(raw, session, counter, command):
    """Verify the checkpoint-2 OPEN ACK or checkpoint-3 command ACK."""
    size = 3 if command == OPEN_SESSION else 4
    if len(raw) != HEADER.size + size + 32:
        raise ValueError('ACK length mismatch')
    if not hmac.compare_digest(raw[-32:], hmac.digest(KEY, raw[:-32], 'sha256')):
        raise ValueError('ACK HMAC invalid')
    expected = (b'AR', 1, 1, 0x80, size, session, counter)
    actual = HEADER.unpack(raw[:HEADER.size])
    for label, got, wanted in zip(('magic', 'version', 'direction', 'type',
                                   'payload length', 'session', 'counter'), actual, expected):
        if got != wanted:
            raise ValueError(f'ACK {label} mismatch: expected {wanted!r}, received {got!r}')
    payload = raw[HEADER.size:-32]
    if payload[0] != command:
        raise ValueError('ACK command mismatch')
    if size == 4 and payload[3] != 0:
        raise ValueError(f'Receiver rejected command 0x{command:02x}, result={payload[3]}')
    expected_armed = 0 if command in (OPEN_SESSION, STOP) else 1
    if payload[1:3] != bytes([1, expected_armed]):
        raise ValueError('ACK active/armed flags mismatch')


def send_and_wait_ack(ser, kind, session, counter, payload=b'', ack_timeout=0.3):
    send(ser, kind, session, counter, payload)
    deadline = time.monotonic() + ack_timeout
    pending = bytearray()
    while time.monotonic() < deadline:
        byte = ser.read(1)
        if not byte:
            continue
        if byte != b'\x00':
            pending.extend(byte)
            if len(pending) > 256:
                raise ValueError('Oversized ACK frame')
            continue
        if not pending:  # Receiver's leading delimiters are permitted.
            continue
        raw = cobs_decode(bytes(pending) + byte)
        pending.clear()
        # A queued, authenticated HELLO from the same session may precede ACK.
        if len(raw) >= 5 and raw[4] == HELLO:
            if unpack_authenticated_hello(raw) != session:
                raise ValueError('Receiver session changed while awaiting ACK')
            continue
        verify_ack(raw, session, counter, kind)
        print(f'RX authenticated ACK command=0x{kind:02x} counter={counter}')
        return
    raise TimeoutError(f'ACK timeout command=0x{kind:02x} counter={counter}; sequence aborted')


class ArmLinkSession:
    """Own serial I/O and counters; heartbeat runs until STOP, close or failure.

    No other reader/writer may use this serial object during the session.
    A failed instance is terminal; construct a new one and open/arm explicitly.
    """
    def __init__(self, ser, heartbeat_interval=0.25, ack_timeout=0.15):
        if not 0 < heartbeat_interval <= 0.25 or not 0 < ack_timeout <= 0.15:
            raise ValueError('Heartbeat interval must be <=250 ms; ACK timeout <=150 ms')
        self.ser = ser
        # Bound real pyserial reads so an ACK deadline cannot be hidden by an
        # indefinitely blocking read. Mock transports need not expose timeout.
        if hasattr(ser, 'timeout') and (ser.timeout is None or ser.timeout > 0.02):
            ser.timeout = 0.02
        if hasattr(ser, 'write_timeout') and (ser.write_timeout is None or ser.write_timeout > 0.15):
            ser.write_timeout = 0.15
        self.interval = heartbeat_interval
        self.ack_timeout = ack_timeout
        self.session = None
        self.counter = 0
        self.usable = False
        self.armed = False
        self.failure = None
        self.last_sent = None
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._worker = None
        self._opened = False

    def _fail(self, reason):
        self.failure = reason
        self.usable = False
        self.armed = False
        self._closed.set()

    def _check(self):
        if not self.usable:
            raise SessionError((self.failure or 'Session is not open') +
                               '; fresh HELLO/OPEN_SESSION and explicit ARM required')

    def _exchange(self, kind, payload=b''):
        # Caller holds the lock: commands and heartbeat cannot steal each other's ACK.
        self._check()
        name = COMMAND_NAMES[kind]
        if self.last_sent is not None and time.monotonic()-self.last_sent >= 0.5:
            self._fail('Local watchdog deadline exceeded before '+name)
            self._check()
        self.counter += 1
        self.last_sent = time.monotonic()
        try:
            send_and_wait_ack(self.ser, kind, self.session, self.counter,
                              payload, self.ack_timeout)
        except (OSError, ValueError, TimeoutError) as error:
            reason = f'{name} counter={self.counter}: {error}'
            self._fail(reason)
            raise SessionError(reason + '; session unusable; fresh HELLO/OPEN_SESSION and explicit ARM required') from error

    def open(self):
        with self._lock:
            if self._opened:
                raise SessionError('Create a new session object for a fresh HELLO/OPEN_SESSION')
            self._opened = True
            try:
                self.session = wait_hello(self.ser)
            except (OSError, ValueError, TimeoutError) as error:
                self._fail('HELLO: '+str(error))
                raise SessionError(self.failure) from error
            self._start_authenticated(self.session)

    def _start_authenticated(self, session):
        # Internal compatibility path for callers that already verified HELLO.
        with self._lock:
            self.session = session
            self.usable = True
            self._opened = True
            self._exchange(OPEN_SESSION)
            self._worker = threading.Thread(target=self._heartbeat_loop,
                                            name='armlink-heartbeat', daemon=True)
            self._worker.start()

    def arm(self):
        with self._lock:
            self._exchange(ARM)
            self.armed = True

    def move(self, targets):
        with self._lock:
            self._check()
            if not self.armed:
                raise SessionError('Explicit ARM required before MOVE')
            self._exchange(MOVE, targets)

    def stop(self):
        with self._lock:
            self._exchange(STOP)
            self.armed = False

    def _heartbeat_loop(self):
        while not self._closed.wait(0.005):
            with self._lock:
                if not self.usable:
                    return
                if self.armed and time.monotonic()-self.last_sent >= self.interval:
                    try:
                        self._exchange(HEARTBEAT)
                    except SessionError as error:
                        print(f'FAIL: {error}', file=sys.stderr)
                        return

    def check(self):
        with self._lock:
            self._check()

    def close(self):
        # Closing NEVER sends STOP: timeout must hold, whereas STOP detaches.
        with self._lock:
            self.usable = False
            self.armed = False
            self._closed.set()
        if self._worker is not None:
            self._worker.join()


def send_sequence(ser, session, targets, hold_seconds=1.0, ack_timeout=0.15):
    """Compatibility one-shot test using a HELLO already authenticated by caller."""
    client = ArmLinkSession(ser, ack_timeout=ack_timeout)
    # The existing entry point has already verified HELLO; no second read needed.
    try:
        client._start_authenticated(session)
        client.arm()
        client.move(targets)
        end = time.monotonic() + hold_seconds
        while time.monotonic() < end:
            client.check()
            time.sleep(min(0.02, max(0, end-time.monotonic())))
        client.stop()  # Successful explicit end only; never sent on failure.
    finally:
        client.close()



def wait_with_heartbeat(client, seconds):
    """Wait without blocking the session's heartbeat worker."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        client.check()
        time.sleep(min(0.02, max(0, end - time.monotonic())))


def send_waypoints(ser, session, angles_list, targets_list):
    """Send a sequence in one authenticated session, then explicitly STOP.

    Timing is based on the firmware's commanded 1 degree per 30 ms, not
    physical feedback. Only use after verifying the arm's initial pose and
    every waypoint's path and clearance. STOP may remove holding torque.
    """
    if len(angles_list) != len(targets_list):
        raise ValueError('Mismatched angles and targets')
    client = ArmLinkSession(ser)
    previous = [90.0] * 4  # Assumed startup command; NOT measured physical pose.
    try:
        client._start_authenticated(session)
        client.arm()
        for index, (angles, targets) in enumerate(zip(angles_list, targets_list), 1):
            client.check()
            print(f'Executing waypoint {index}/{len(targets_list)}')
            client.move(targets)
            max_delta = max(abs(a-b) for a, b in zip(angles, previous))
            # Firmware steps 1 degree / 30 ms, plus a settling margin.
            wait_seconds = max_delta * 0.030 + 0.75
            print(f'Estimated wait: {wait_seconds:.2f}s (not measured feedback)')
            wait_with_heartbeat(client, wait_seconds)
            previous = list(angles)
        client.stop()
    finally:
        # On failures, do NOT blindly send STOP: watchdog handles session loss.
        client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--angles', nargs=4, type=float, metavar=('BASE', 'SHOULDER', 'ELBOW', 'GRIPPER'))
    src.add_argument('--xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='meters; gripper uses --gripper')
    src.add_argument('--sequence', action='store_true', help='Execute the XYZ waypoints defined near the top of this file')
    p.add_argument('--gripper', type=float, default=90)
    p.add_argument('--port', help='ESP32 serial port')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--send-motion', action='store_true', help='Explicitly ARM and send a MOVE packet')
    args = p.parse_args()
    if args.sequence:
        # Validate every waypoint before any physical motion is requested.
        planned_angles = [angles_from_xyz(xyz, args.gripper) for xyz in waypoints]
    else:
        planned_angles = [args.angles if args.angles is not None
                          else angles_from_xyz(args.xyz, args.gripper)]
    targets_list = [payload_for_angles(angles) for angles in planned_angles]
    for index, (angles, targets) in enumerate(zip(planned_angles, targets_list), 1):
        print(f'Waypoint {index} servo targets:', dict(zip(JOINT_NAMES, angles)))
        print('MOVE payload:', targets.hex())
        demo_session = bytes(range(16))
        test_frame = cobs_encode(packet(MOVE, demo_session, 3, targets))
        assert cobs_decode(test_frame) == packet(MOVE, demo_session, 3, targets)
        print('Offline framing test passed:', len(test_frame), 'bytes')
    if args.dry_run:
        return
    if not args.port:
        p.error('--port is required for hardware mode')
    if not args.send_motion:
        p.error('hardware movement requires --send-motion; use --dry-run for packet tests')
    import serial
    with serial.Serial(args.port, args.baud, timeout=0.02, write_timeout=0.15) as ser:
        # Opening USB serial may reset the ESP32; HELLO must come from the actual receiver.
        session = wait_hello(ser)
        print('RX valid HELLO. Session:', session.hex())
        try:
            if args.sequence:
                send_waypoints(ser, session, planned_angles, targets_list)
            else:
                send_sequence(ser, session, targets_list[0])
        except (OSError, ValueError, TimeoutError, SessionError) as error:
            raise SystemExit(f'FAIL: {error}. No further commands sent; receiver watchdog handles timeout.')
    print('PASS: all commands acknowledged; receiver confirmed STOP/disarmed.')


if __name__ == '__main__':
    main()
