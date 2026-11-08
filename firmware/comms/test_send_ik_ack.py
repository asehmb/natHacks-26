"""Laptop-only ACK tests. No serial port or physical commands."""
import hmac
import unittest
import send_ik as host

SESSION = bytes(range(16))


def ack(command, counter, session=SESSION):
    payload = bytes([command, 1, int(command not in (host.OPEN_SESSION, host.STOP))])
    if command != host.OPEN_SESSION:
        payload += b'\0'
    body = host.HEADER.pack(b'AR', 1, 1, 0x80, len(payload), session, counter) + payload
    return body + hmac.digest(host.KEY, body, 'sha256')


class FakeSerial:
    def __init__(self, change=lambda raw: raw, fail_command=None):
        self.change = change
        self.fail_command = fail_command
        self.rx = bytearray()
        self.commands = []

    def write(self, data):
        raw = host.cobs_decode(data)
        command, counter = raw[4], int.from_bytes(raw[23:27], 'little')
        self.commands.append(command)
        reply = ack(command, counter)
        if self.fail_command in (None, command):
            reply = self.change(reply)
        if reply is not None:
            self.rx.extend(b'\0' + host.cobs_encode(reply))
        return len(data)

    def read(self, count):
        result = bytes(self.rx[:count])
        del self.rx[:count]
        return result

    def flush(self):
        pass


def change_byte(offset, resign=True):
    def change(raw):
        altered = bytearray(raw)
        altered[offset] ^= 1
        if resign:
            altered[-32:] = hmac.digest(host.KEY, altered[:-32], 'sha256')
        return bytes(altered)
    return change


class AckTests(unittest.TestCase):
    def test_valid_sequence(self):
        serial = FakeSerial()
        host.send_sequence(serial, SESSION, host.payload_for_angles([90]*4), hold_seconds=0)
        self.assertEqual(serial.commands, [2, 3, 4, 5])

    def test_heartbeat_ack(self):
        host.send_and_wait_ack(FakeSerial(), host.HEARTBEAT, SESSION, 4)

    def test_command_ack_mismatches(self):
        for offset in (7, 23, 27, 28, 29):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                host.send_and_wait_ack(FakeSerial(change_byte(offset)), host.MOVE, SESSION, 3)

    def test_missing_heartbeat_ack(self):
        serial = FakeSerial(lambda raw: None)
        with self.assertRaises(TimeoutError):
            host.send_and_wait_ack(serial, host.HEARTBEAT, SESSION, 4, ack_timeout=0.001)
        self.assertEqual(serial.commands, [host.HEARTBEAT])

    def test_signed_mismatches_abort_before_arm(self):
        # Correctly signed mismatches exercise semantic checks independently.
        for offset in (0, 2, 3, 4, 5, 7, 23, 27, 28, 29):
            with self.subTest(offset=offset):
                serial = FakeSerial(change_byte(offset))
                with self.assertRaises(host.SessionError):
                    host.send_sequence(serial, SESSION, b'\0'*8, hold_seconds=0)
                self.assertEqual(serial.commands, [host.OPEN_SESSION])

    def test_tampering_and_truncation(self):
        for change in (change_byte(-1, False), lambda raw: raw[:-1]):
            with self.subTest(change=change), self.assertRaises(ValueError):
                host.send_and_wait_ack(FakeSerial(change), 2, SESSION, 1)

    def test_rejected_move_stops_sequence(self):
        serial = FakeSerial(change_byte(30), fail_command=host.MOVE)
        with self.assertRaisesRegex(host.SessionError, 'rejected'):
            host.send_sequence(serial, SESSION, b'\0'*8, hold_seconds=0)
        self.assertEqual(serial.commands, [2, 3, 4])

    def test_missing_ack_at_each_stage(self):
        for command, expected in ((2, [2]), (3, [2, 3]), (4, [2, 3, 4]), (5, [2, 3, 4, 5])):
            serial = FakeSerial(lambda raw: None, fail_command=command)
            with self.subTest(command=command), self.assertRaises(host.SessionError):
                host.send_sequence(serial, SESSION, b'\0'*8, hold_seconds=0, ack_timeout=0.001)
            self.assertEqual(serial.commands, expected)

    def test_partial_reads_persist_across_empty_reads(self):
        serial = FakeSerial()
        original = serial.read
        reads = 0

        def read(count):
            nonlocal reads
            reads += 1
            return b'' if reads % 2 else original(count)

        serial.read = read
        host.send_and_wait_ack(serial, 2, SESSION, 1)

    def test_queued_hello(self):
        serial = FakeSerial()
        body = host.HEADER.pack(b'AR', 1, 1, 1, 0, SESSION, 0)
        serial.rx.extend(host.cobs_encode(body+hmac.digest(host.KEY, body, 'sha256')))
        host.send_and_wait_ack(serial, 2, SESSION, 1)

    def test_short_write(self):
        serial = FakeSerial()
        serial.write = lambda data: len(data)-1
        with self.assertRaisesRegex(OSError, 'Incomplete'):
            host.send_and_wait_ack(serial, 2, SESSION, 1)


if __name__ == '__main__':
    unittest.main()
