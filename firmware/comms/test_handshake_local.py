"""Laptop C/Python interoperability and rejection checks; no serial device."""
import ctypes
import hmac
from pathlib import Path
import subprocess
import tempfile
import unittest

from send_ik import KEY, cobs_decode, cobs_encode, packet, OPEN_SESSION
from test_handshake import handshake, verify_ack


class State(ctypes.Structure):
    _fields_ = [('session_active', ctypes.c_bool), ('armed', ctypes.c_bool),
                ('last_valid_ms', ctypes.c_uint32), ('last_counter', ctypes.c_uint32),
                ('session_id', ctypes.c_ubyte * 16)]


class HandshakeTests(unittest.TestCase):
    def test_c_python_handshake_and_rejections(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder) / 'armlink.so'
            subprocess.run(['gcc', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                            str(root / 'armlink.c'), '-lmbedcrypto', '-o', str(library)], check=True)
            lib = ctypes.CDLL(str(library))
            buf = lambda data: (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
            session = bytes(range(16))
            state = State()
            lib.prepare_receiver_session(ctypes.byref(state), buf(session))
            ack = (ctypes.c_ubyte * 62)()
            self.assertEqual(lib.build_open_session_ack(ctypes.byref(state), buf(KEY), ack), 0)
            hello = (ctypes.c_ubyte * 59)()
            self.assertEqual(lib.build_hello_packet(buf(session), buf(KEY), hello), 1)
            testcase = self

            class SimulatedSerial:
                def __init__(self):
                    self.rx = bytearray(cobs_encode(bytes(hello)))
                    self.writes = []

                def read(self, count):
                    result = bytes(self.rx[:count])
                    del self.rx[:count]
                    return result

                def write(self, data):
                    self.writes.append(data)
                    raw = cobs_decode(data)
                    testcase.assertEqual(raw, packet(OPEN_SESSION, session, 1))
                    testcase.assertEqual(lib.receive_open_session(
                        ctypes.byref(state), buf(raw), len(raw), buf(KEY), 1000), 1)
                    testcase.assertEqual(lib.build_open_session_ack(
                        ctypes.byref(state), buf(KEY), ack), 1)
                    self.rx.extend(cobs_encode(bytes(ack)))
                    return len(data)

                def flush(self):
                    pass

            serial = SimulatedSerial()
            handshake(serial, 1)
            self.assertEqual(len(serial.writes), 1)
            self.assertFalse(state.armed)
            verify_ack(bytes(ack), session, 1)
            # Re-sign each semantic mismatch so HMAC cannot hide a missing check.
            for offset in (2, 3, 4, 5, 7, 23, 27, 28, 29):
                changed = bytearray(ack)
                changed[offset] ^= 1
                changed[-32:] = hmac.digest(KEY, changed[:-32], 'sha256')
                with self.subTest(offset=offset), self.assertRaises(ValueError):
                    verify_ack(bytes(changed), session, 1)
            changed = bytearray(ack)
            changed[-1] ^= 1
            with self.assertRaises(ValueError):
                verify_ack(bytes(changed), session, 1)
            with self.assertRaises(ValueError):
                verify_ack(bytes(ack)[:-1], session, 1)
            raw = packet(OPEN_SESSION, session, 1)
            self.assertEqual(lib.receive_open_session(
                ctypes.byref(state), buf(raw), len(raw), buf(KEY), 1100), 0)


if __name__ == '__main__':
    unittest.main()
