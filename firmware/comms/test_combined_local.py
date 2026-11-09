"""Run the actual combined sketch with mocked Arduino serial/clock/random APIs.

Uses real C HMAC/session helpers and the same Python test as the hardware run.
This is a LAPTOP SIMULATION, not ESP32 compilation or physical verification.
"""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import time
from test_combined import run, Client
import struct

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as directory:
    tmp = Path(directory)
    (tmp/'Arduino.h').write_text('''
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <deque>
#include <chrono>
struct MockSerial {
  std::deque<uint8_t> input, output;
  void begin(int) {}
  int available() { return input.size(); }
  int read() { int v=input.front(); input.pop_front(); return v; }
  void write(uint8_t b) { output.push_back(b); }
  void write(const uint8_t *p, size_t n) { while(n--) write(*p++); }
};
extern MockSerial Serial;
inline uint32_t millis() {
  static auto start=std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now()-start).count();
}
''')
    (tmp/'esp_system.h').write_text('''
#pragma once
#include <stdint.h>
#include <stddef.h>
inline void esp_fill_random(void *p, size_t n) {
  static uint8_t seed=0;
  uint8_t *b=(uint8_t*)p;
  while(n--) *b++=++seed;
}
''')
    (tmp/'runner.cpp').write_text('''
#include "Arduino.h"
MockSerial Serial;
#include "combined_only.ino"
extern "C" {
void start_sketch() { setup(); }
unsigned mock_writes() { return servos.backend.writes; }
unsigned mock_detaches() { return servos.backend.detachRequests; }
unsigned mock_attachment_mask() {
  unsigned mask=0;
  for(int i=0;i<4;++i) if(servos.backend.attached[i]) mask |= 1u<<i;
  return mask;
}
void seed_mock_attachment() {
  for(int i=0;i<4;++i) servos.backend.attached[i]=true;
}
void feed(uint8_t b) { Serial.input.push_back(b); loop(); }
int poll_byte() {
  loop();
  if (Serial.output.empty()) return -1;
  int b=Serial.output.front(); Serial.output.pop_front(); return b;
}
}
''')
    subprocess.run(['gcc', '-fPIC', '-Wall', '-Wextra', '-Werror', '-c',
                    str(root/'armlink.c'), '-o', str(tmp/'armlink.o')], check=True)
    subprocess.run(['g++', '-std=c++11', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                    '-I'+str(tmp), '-I'+str(root), '-I'+str(root/'combined_only'),
                    str(tmp/'runner.cpp'), str(tmp/'armlink.o'), '-lmbedcrypto',
                    '-o', str(tmp/'sketch.so')], check=True)
    lib = ctypes.CDLL(str(tmp/'sketch.so'))
    lib.start_sketch()

    class Serial:
        def read(self, count):
            b = lib.poll_byte()
            if b < 0:
                time.sleep(0.001)
                return b''
            return bytes([b])

        def write(self, data):
            for b in data:
                lib.feed(b)
            return len(data)

        def flush(self):
            pass

    print('LAPTOP SIMULATION: actual sketch, mocked Arduino APIs')
    run(Serial())
    assert lib.mock_writes() == 4, 'Rejected commands must cause zero writes'
    assert lib.mock_detaches() == 1, 'STOP must reach the backend'
    assert lib.mock_attachment_mask() == 0
    client = Client(Serial())
    client.open()
    client.ack(3, armed=True)
    assert lib.mock_attachment_mask() == 0, 'ARM must not invent reattachment policy'
    # Test fixture only: simulate already-attached outputs without any hardware.
    lib.seed_mock_attachment()
    client.ack(4, struct.pack('<4h', 9150, 8550, 10050, 9050), armed=True)
    client.state(True, (92, 86, 101, 91), 2)
    writes = lib.mock_writes()
    time.sleep(0.55)
    lib.poll_byte()
    assert lib.mock_attachment_mask() == 15, 'Timeout must preserve attachment'
    assert lib.mock_writes() == writes, 'Timeout must not write new targets'
    client.open()
    client.state(False, (92, 86, 101, 91), 2)
    client.ack(5)
    assert lib.mock_attachment_mask() == 0, 'STOP must detach every channel'
    time.sleep(0.55)
    lib.poll_byte()
    assert lib.mock_attachment_mask() == 0, 'Timeout after STOP must not reattach'
    client.open()
    client.state(False, (92, 86, 101, 91), 2)
    assert lib.mock_writes() == writes
    print('PASS: 3B sketch dispatch reaches mock writes/detach; timeout holds attachment and targets')
    # Exercise the production sender against the actual mocked C++ receiver.
    from send_ik import send_sequence, payload_for_angles
    time.sleep(0.55)
    lib.poll_byte()
    serial = Serial()
    # This mock serial has read(), while the original HELLO utility uses read_until().
    from test_combined import frames
    from send_ik import unpack_authenticated_hello
    for raw in frames(serial, 1):
        try:
            session = unpack_authenticated_hello(raw)
            break
        except ValueError:
            continue
    else:
        raise AssertionError('No simulated HELLO for production sender')
    send_sequence(serial, session, payload_for_angles([90, 85, 100, 90]), hold_seconds=0.8)
    print('PASS: production sender verified all ACKs from mocked combined firmware')
    from test_rejections_usb import run as run_rejections
    time.sleep(0.55)
    lib.poll_byte()
    writes_before = lib.mock_writes()
    run_rejections(Serial())
    assert lib.mock_writes() == writes_before, 'Rejection test must never write servo targets'
    print('PASS: rejection/reconnect USB harness verified in LAPTOP SIMULATION')
    from test_heartbeat_faults_usb import run as run_heartbeat_faults
    time.sleep(0.6)
    lib.poll_byte()
    writes_before = lib.mock_writes()
    run_heartbeat_faults(Serial())
    assert lib.mock_writes() == writes_before
    print('PASS: USB host heartbeat fault harness verified in LAPTOP SIMULATION')
    from test_heartbeat_reliability import run as run_reliability
    time.sleep(0.6)
    lib.poll_byte()
    writes_before = lib.mock_writes()
    run_reliability(Serial(), seconds=2)
    assert lib.mock_writes() == writes_before
    print('PASS: reliability harness short smoke test in LAPTOP SIMULATION (not a five-minute run)')
