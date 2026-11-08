
"""EasyBot MK2 host sender for firmware/comms/armlink.c.

Packet-only test: python3 firmware/comms/send_ik.py --dry-run --angles 90 85 100 90
Hardware: python3 firmware/comms/send_ik.py --port /dev/cu.usbserial-10 --angles 90 85 100 90 --send-motion
Requires pyserial only for hardware mode. The ESP32 must implement HELLO and receive
OPEN_SESSION, ARM, MOVE, HEARTBEAT and return authenticated ACKs.
"""
import argparse
import hashlib
import hmac
import math
import struct
import sys
import time
from pathlib import Path

HEADER = struct.Struct('<2sBBBH16sI')
KEY = bytes([0x0B]) * 32  # Existing firmware TEST KEY, not secure for deployment.
HELLO, OPEN_SESSION, ARM, MOVE, STOP, HEARTBEAT = 1, 2, 3, 4, 5, 6
JOINT_NAMES = ('base', 'shoulder', 'elbow', 'gripper')
# Temporary conservative *command* limits. Replace after mechanical calibration.
LIMITS = ((60, 120), (60, 120), (60, 120), (70, 110))


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
    expected = (b'AR', 1, 1, 0x80, size, session, counter)
    if HEADER.unpack(raw[:HEADER.size]) != expected:
        raise ValueError('ACK header/session/counter mismatch')
    if not hmac.compare_digest(raw[-32:], hmac.digest(KEY, raw[:-32], 'sha256')):
        raise ValueError('ACK HMAC invalid')
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


def send_sequence(ser, session, targets, hold_seconds=1.0, ack_timeout=0.3):
    """One-shot motion test. Every command gates the next on an accepted ACK.

    On failure send nothing further; never retry ambiguous motion commands.
    The receiver watchdog then applies its agreed timeout policy.
    """
    counter = 0

    def exchange(kind, payload=b''):
        nonlocal counter
        counter += 1
        send_and_wait_ack(ser, kind, session, counter, payload, ack_timeout)

    exchange(OPEN_SESSION)
    exchange(ARM)
    exchange(MOVE, targets)
    end = time.monotonic() + hold_seconds
    while time.monotonic() < end:
        exchange(HEARTBEAT)
        time.sleep(0.1)
    exchange(STOP)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--angles', nargs=4, type=float, metavar=('BASE', 'SHOULDER', 'ELBOW', 'GRIPPER'))
    src.add_argument('--xyz', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='meters; gripper uses --gripper')
    p.add_argument('--gripper', type=float, default=90)
    p.add_argument('--port', help='ESP32 serial port')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--send-motion', action='store_true', help='Explicitly ARM and send a MOVE packet')
    args = p.parse_args()
    angles = args.angles if args.angles is not None else angles_from_xyz(args.xyz, args.gripper)
    targets = payload_for_angles(angles)
    print('Servo targets:', dict(zip(JOINT_NAMES, angles)))
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
    with serial.Serial(args.port, args.baud, timeout=0.15, write_timeout=1) as ser:
        # Opening USB serial may reset the ESP32; HELLO must come from the actual receiver.
        session = wait_hello(ser)
        print('RX valid HELLO. Session:', session.hex())
        try:
            send_sequence(ser, session, targets)
        except (OSError, ValueError, TimeoutError) as error:
            raise SystemExit(f'FAIL: {error}. No further commands sent; receiver watchdog handles timeout.')
    print('PASS: all commands acknowledged; receiver confirmed STOP/disarmed.')


if __name__ == '__main__':
    main()
