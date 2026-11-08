import hashlib
import hmac

import struct
from armlink_host import build_move_payload

angles = [90, 85, 100, 90]

payload = build_move_payload(angles)

print("MOVE payload:", payload.hex())
print("Decoded targets:", struct.unpack("<4h", payload))


from armlink_host import build_move_packet

session_id = bytes(range(16))

packet = build_move_packet(
    [90, 85, 100, 90],
    session_id,
    2
)

print("Packet length:", len(packet))
print("MOVE command:", hex(packet[4]))
print("Payload:", packet[27:35].hex())
print("HMAC valid:", hmac.compare_digest(
    packet[-32:],
    hmac.digest(
        bytes([0x0B]) * 32,
        packet[:-32],
        hashlib.sha256
    )
))




from armlink_host import encode_serial_frame, decode_serial_frame

frame = encode_serial_frame(packet)
decoded = decode_serial_frame(frame)

print("Frame ends with 00:", frame[-1] == 0)
print("No internal zeros:", 0 not in frame[:-1])
print("Round trip successful:", decoded == packet)
print("Frame length:", len(frame))



import serial

PORT = "/dev/cu.usbserial-10"

with serial.Serial(PORT, 115200, timeout=1) as esp:
    esp.write(frame)
    esp.flush()
    print(f"Sent {len(frame)} bytes to ESP32")
