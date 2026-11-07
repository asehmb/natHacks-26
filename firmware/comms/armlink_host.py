import hashlib
import hmac
import struct

HEADER = struct.Struct("<2sBBBH16sI")
TEST_KEY = bytes([0x0B]) * 32

def decode_serial_frame(frame):
    if not frame or frame[-1] != 0 or 0 in frame[:-1]:
        raise ValueError("Expected one COBS frame ending in 00")

    encoded = frame[:-1]
    decoded = bytearray()
    index = 0

    while index < len(encoded):
        code = encoded[index]
        end = index + code
        if end > len(encoded):
            raise ValueError("Truncated COBS frame")

        decoded.extend(encoded[index + 1:end])
        if code != 0xFF and end < len(encoded):
            decoded.append(0)
        index = end

    return bytes(decoded)


try:
    frame = bytes.fromhex(input("Paste HELLO frame hex: "))
    hello = decode_serial_frame(frame)
except ValueError as error:
    raise SystemExit(f"Invalid HELLO frame: {error}")

if len(hello) != HEADER.size + 32:
    raise SystemExit("Invalid HELLO packet length")

header = hello[:HEADER.size]
tag = hello[HEADER.size:]

expected_tag = hmac.digest(TEST_KEY, header, hashlib.sha256)
if not hmac.compare_digest(tag, expected_tag):
    raise SystemExit("HELLO HMAC verification failed")

magic, version, direction, kind, payload_len, session_id, counter = (
    HEADER.unpack(header)
)
if (magic, version, direction, kind, payload_len, counter) != (
    b"AR", 1, 1, 0x01, 0, 0
):
    raise SystemExit("Unexpected HELLO header")

open_header = HEADER.pack(b"AR", 1, 0, 0x02, 0, session_id, 1)
open_tag = hmac.digest(TEST_KEY, open_header, hashlib.sha256)

print("PASS: Python verified the C HELLO")
print("OPEN_SESSION hex:", (open_header + open_tag).hex())
reply_frame = bytearray()
for block in (open_header + open_tag).split(b"\x00"):
    reply_frame.append(len(block) + 1)
    reply_frame.extend(block)
reply_frame.append(0)

print("OPEN_SESSION frame C bytes:",
      ", ".join(f"0x{byte:02x}" for byte in reply_frame))