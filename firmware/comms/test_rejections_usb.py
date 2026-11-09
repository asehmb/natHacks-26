"""USB rejection/reconnect checkpoint. Outputs-disabled combined_only firmware.

Sends ARM/STOP and invalid packets, but NO MOVE commands. No servo movement.
"""
import argparse
import struct
import time

from send_ik import packet, cobs_encode
from test_combined import Client, frames


def write(ser, data):
    if ser.write(data) != len(data):
        raise OSError('Incomplete test packet write')
    ser.flush()


def snapshot(client):
    raw = client.request(7)
    values = struct.unpack('<BBB4hII', raw[27:-32])
    if values[0] != 1 or values[2] != 0:
        raise ValueError('Require active session and outputs_enabled=0 before testing')
    return values


def unchanged(client, baseline, armed):
    values = snapshot(client)
    if values[1] != int(armed) or values[3:] != baseline[3:]:
        raise ValueError('Unexpected arm/target/move/timeout change: '+str(values))


def silent(ser, frame):
    write(ser, frame)
    # Correct behavior is silence; follow with authenticated state to prove life.
    for _ in frames(ser, 0.06):
        raise ValueError('Invalid/replayed packet unexpectedly produced a reply')


def run(ser):
    client = Client(ser)
    client.open()
    baseline = snapshot(client)
    if baseline[1] != 0:
        raise ValueError('Expected disarmed receiver at start')
    print('PASS: preflight outputs_enabled=0; no MOVE will be sent')

    # Use the next legitimate counter without consuming it on the host. The
    # following GET_STATE reuses that counter, detecting illegal advancement.
    def candidate(session=None):
        return packet(3, session or client.session, client.counter+1)

    altered = bytearray(candidate())
    altered[-1] ^= 1
    silent(ser, cobs_encode(bytes(altered)))
    unchanged(client, baseline, False)
    print('PASS: invalid HMAC rejected without arming or consuming counter')

    wrong = bytes([client.session[0] ^ 1])+client.session[1:]
    silent(ser, cobs_encode(candidate(wrong)))
    unchanged(client, baseline, False)
    print('PASS: signed wrong-session packet rejected')

    silent(ser, b'\x05\x01\x00')  # Truncated COBS block.
    unchanged(client, baseline, False)
    print('PASS: malformed COBS discarded; valid state still responds')

    silent(ser, b'\x01'*200+b'\x00')
    unchanged(client, baseline, False)
    print('PASS: oversized frame discarded; receiver recovers at delimiter')

    client.ack(3, armed=True)
    client.ack(5)
    old_stop = packet(5, client.session, client.counter)
    client.ack(3, armed=True)
    silent(ser, cobs_encode(old_stop))
    unchanged(client, baseline, True)
    print('PASS: replayed STOP rejected; receiver remains armed')

    old_session = client.session
    # No valid traffic for 0.8 seconds; invalid packets must not feed watchdog.
    invalid = bytearray(packet(6, old_session, client.counter+1))
    invalid[-1] ^= 1
    until = time.monotonic()+0.8
    while time.monotonic() < until:
        write(ser, cobs_encode(bytes(invalid)))
        time.sleep(0.08)
    client.open()
    if client.session == old_session:
        raise ValueError('Invalid traffic kept the old session alive')
    after = snapshot(client)
    if after[1] != 0 or after[3:8] != baseline[3:8] or after[8] != baseline[8]+1:
        raise ValueError('Watchdog/reconnect state mismatch: '+str(after))
    print('PASS: invalid traffic does not feed watchdog; fresh session is disarmed')

    silent(ser, cobs_encode(packet(3, old_session, client.counter+1)))
    unchanged(client, after, False)
    print('PASS: prior-session ARM rejected after reconnect')
    client.ack(3, armed=True)
    unchanged(client, after, True)
    client.ack(5)
    unchanged(client, after, False)
    print('PASS: explicit ARM works in fresh session; STOP returns to disarmed')
    print('PASS: USB rejection/reconnect checkpoint; outputs disabled; no MOVE sent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    args = parser.parse_args()
    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=0.01, write_timeout=0.15) as ser:
            run(ser)
    except (OSError, ValueError, TimeoutError, serial.SerialException) as error:
        raise SystemExit('FAIL: '+str(error))


if __name__ == '__main__':
    main()
