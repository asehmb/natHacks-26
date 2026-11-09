"""Laptop-only session lifecycle tests; no serial port or physical outputs."""
import hmac
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
import io

import send_ik as host
from test_send_ik_ack import ack, change_byte, SESSION


class Serial:
    def __init__(self, mutation=lambda raw: raw, session=SESSION, fail=host.HEARTBEAT):
        self.session = session
        self.mutation = mutation
        self.fail = fail
        self.commands = []
        self.timestamps = []
        self.counters = []
        self.rx = bytearray()
        self.hello()

    def hello(self):
        body = host.HEADER.pack(b'AR', 1, 1, 1, 0, self.session, 0)
        self.rx.extend(host.cobs_encode(body+hmac.digest(host.KEY, body, 'sha256')))

    def read_until(self, delimiter, size):
        end = self.rx.index(0)+1
        data = bytes(self.rx[:end])
        del self.rx[:end]
        return data

    def read(self, count):
        if not self.rx:
            time.sleep(0.001)
        data = bytes(self.rx[:count])
        del self.rx[:count]
        return data

    def write(self, data):
        raw = host.cobs_decode(data)
        kind = raw[4]
        counter = int.from_bytes(raw[23:27], 'little')
        self.commands.append(kind)
        self.timestamps.append(time.monotonic())
        self.counters.append(counter)
        reply = ack(kind, counter, self.session)
        if kind == self.fail:
            reply = self.mutation(reply)
        if reply is not None:
            self.rx.extend(b'\0'+host.cobs_encode(reply))
        return len(data)

    def flush(self):
        pass


def wait_for(predicate, seconds=2):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('Timed out waiting for test condition')


class HeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.out = redirect_stdout(self.output)
        self.err = redirect_stderr(self.output)
        self.out.__enter__()
        self.err.__enter__()

    def tearDown(self):
        self.err.__exit__(None, None, None)
        self.out.__exit__(None, None, None)

    def client(self, serial, **kwargs):
        client = host.ArmLinkSession(serial, **kwargs)
        self.addCleanup(client.close)
        client.open()
        return client

    def test_continuous_idle_heartbeats_and_explicit_stop(self):
        serial = Serial()
        client = self.client(serial)
        time.sleep(0.03)
        self.assertEqual(serial.commands, [host.OPEN_SESSION])
        client.arm()
        client.move(host.payload_for_angles([90]*4))
        wait_for(lambda: serial.commands.count(host.HEARTBEAT) >= 4)
        client.move(host.payload_for_angles([91]*4))
        client.stop()
        count = len(serial.commands)
        time.sleep(0.3)
        self.assertEqual(len(serial.commands), count)
        self.assertFalse(client.armed)
        self.assertTrue(client.usable)
        beats = [stamp for kind, stamp in zip(serial.commands, serial.timestamps) if kind == 6]
        for first, second in zip(beats, beats[1:]):
            self.assertGreaterEqual(second-first, 0.24)
            self.assertLess(second-first, 0.45)
        self.assertEqual(serial.counters, list(range(1, count+1)))

    def test_heartbeat_failures_invalidate_and_block_all_commands(self):
        cases = {
            'rejected': (change_byte(30), 'rejected'),
            'missing': (lambda raw: None, 'timeout'),
            'malformed': (lambda raw: raw[:-1], 'length'),
            'hmac': (change_byte(-1, False), 'HMAC'),
            'wrong-session': (change_byte(7), 'session mismatch'),
            'wrong-counter': (change_byte(23), 'counter mismatch'),
            'wrong-command': (change_byte(27), 'command mismatch'),
            'wrong-armed': (change_byte(29), 'flags mismatch'),
            'wrong-active': (change_byte(28), 'flags mismatch'),
        }
        for name, (mutation, reason) in cases.items():
            with self.subTest(name=name):
                serial = Serial(mutation)
                client = self.client(serial, heartbeat_interval=0.01, ack_timeout=0.02)
                client.arm()
                client.move(b'\0'*8)
                wait_for(lambda: not client.usable)
                self.assertIn(reason, client.failure)
                self.assertIn('HEARTBEAT counter=', client.failure)
                before = list(serial.commands)
                for action in (lambda: client.move(b'\0'*8), client.arm, client.stop, client.open):
                    with self.assertRaises(host.SessionError):
                        action()
                client.close()
                self.assertEqual(serial.commands, before)
                self.assertNotIn(host.STOP, serial.commands)
                self.assertEqual(serial.commands.count(host.MOVE), 1)

    def test_rejected_move_aborts_and_no_automatic_stop(self):
        serial = Serial(change_byte(30), fail=host.MOVE)
        client = self.client(serial)
        client.arm()
        with self.assertRaisesRegex(host.SessionError, 'MOVE counter=3.*rejected'):
            client.move(b'\0'*8)
        with self.assertRaises(host.SessionError):
            client.move(b'\0'*8)
        self.assertEqual(serial.commands, [2, 3, 4])

    def test_malformed_cobs_heartbeat_ack(self):
        class BrokenSerial(Serial):
            def write(self, data):
                size = super().write(data)
                if self.commands[-1] == host.HEARTBEAT:
                    self.rx = bytearray(b'\x05\x01\x00')
                return size

        serial = BrokenSerial()
        client = self.client(serial, heartbeat_interval=0.01)
        client.arm()
        wait_for(lambda: not client.usable)
        self.assertIn('Invalid COBS', client.failure)
        with self.assertRaises(host.SessionError):
            client.move(b'\0'*8)
        self.assertNotIn(host.MOVE, serial.commands)
        self.assertNotIn(host.STOP, serial.commands)

    def test_recovery_requires_new_hello_open_and_explicit_arm(self):
        old = self.client(Serial(lambda raw: None), heartbeat_interval=0.01, ack_timeout=0.02)
        old.arm()
        wait_for(lambda: not old.usable)
        fresh = Serial(session=bytes(reversed(SESSION)))
        new = self.client(fresh)
        with self.assertRaisesRegex(host.SessionError, 'Explicit ARM'):
            new.move(b'\0'*8)
        self.assertEqual(fresh.commands, [2])
        new.arm()
        new.move(b'\0'*8)
        self.assertEqual(fresh.commands, [2, 3, 4])

    def test_close_never_sends_stop(self):
        serial = Serial()
        client = self.client(serial)
        client.arm()
        client.close()
        time.sleep(0.03)
        self.assertEqual(serial.commands, [2, 3])
        self.assertFalse(client._worker.is_alive())

    def test_no_send_after_local_watchdog_deadline(self):
        serial = Serial()
        client = self.client(serial)
        # Unarmed sessions are not kept alive; force a missed deadline.
        client.last_sent = time.monotonic()-0.51
        with self.assertRaisesRegex(host.SessionError, 'watchdog deadline'):
            client.arm()
        self.assertEqual(serial.commands, [2])


if __name__ == '__main__':
    unittest.main()
