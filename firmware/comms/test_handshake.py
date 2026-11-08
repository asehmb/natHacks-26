"""Handshake only: Windows `py test_handshake.py --port COM5` (requires pyserial)."""
import argparse
import hmac
import time

from send_ik import (HEADER, KEY, OPEN_SESSION, cobs_decode, cobs_encode,
                     packet, unpack_authenticated_hello)


def verify_ack(raw, session, counter):
    if len(raw) != 62:
        raise ValueError('ACK length mismatch')
    expected = (b'AR', 1, 1, 0x80, 3, session, counter)
    if HEADER.unpack(raw[:27]) != expected:
        raise ValueError('ACK header/session/counter mismatch')
    if not hmac.compare_digest(raw[-32:], hmac.digest(KEY, raw[:-32], 'sha256')):
        raise ValueError('ACK authentication failed')
    if raw[27:30] != bytes([OPEN_SESSION, 1, 0]):
        raise ValueError('ACK must confirm OPEN_SESSION, active, disarmed')


def handshake(ser, timeout):
    deadline = time.monotonic() + timeout
    pending = bytearray()
    session = None
    while time.monotonic() < deadline:
        byte = ser.read(1)
        if not byte:
            continue
        if byte != b'\x00':
            pending.extend(byte)
            if len(pending) > 256:
                # Discard until delimiter, including any ROM startup text.
                while byte and byte != b'\x00' and time.monotonic() < deadline:
                    byte = ser.read(1)
                pending.clear()
            continue
        frame = bytes(pending) + b'\x00'
        pending.clear()
        try:
            raw = cobs_decode(frame)
        except ValueError:
            continue
        if session is None:
            try:
                session = unpack_authenticated_hello(raw)
            except ValueError:
                continue
            print('RX authenticated HELLO session=' + session.hex())
            outgoing = cobs_encode(packet(OPEN_SESSION, session, 1))
            if ser.write(outgoing) != len(outgoing):
                raise OSError('Incomplete OPEN_SESSION write')
            ser.flush()
            print('TX OPEN_SESSION counter=1')
        elif len(raw) >= 5 and raw[4] == 0x80:
            verify_ack(raw, session, 1)
            print('RX authenticated ACK session=' + session.hex() +
                  ' counter=1 command=OPEN_SESSION session_active=1 armed=0')
            print('PASS: USB handshake verified; no ARM or MOVE sent')
            return
    raise TimeoutError('Handshake timed out waiting for ' +
                       ('HELLO' if session is None else 'authenticated ACK'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--timeout', type=float, default=10)
    args = parser.parse_args()
    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=0.1, write_timeout=1) as ser:
            handshake(ser, args.timeout)
    except (OSError, ValueError, TimeoutError, serial.SerialException) as error:
        raise SystemExit('FAIL: ' + str(error))


if __name__ == '__main__':
    main()
