import hashlib
import hmac
import struct

HEADER = struct.Struct("<2sBBBH16sI")
TEST_KEY = bytes([0x0B]) * 32



def build_move_payload(servo_angles):
    if len(servo_angles) != 4:
        raise ValueError("Expected 4 servo angles")

    targets = []

    for angle in servo_angles:
        value = round(angle * 100)

        if not -32768 <= value <= 32767:
            raise ValueError("Angle exceeds packet range")

        targets.append(value)

    return struct.pack("<4h", *targets)



def build_move_packet(servo_angles, session_id, counter):
    if len(session_id) != 16:
        raise ValueError("Session ID must be 16 bytes")

    payload = build_move_payload(servo_angles)

    header = HEADER.pack(
        b"AR",       # Magic bytes
        1,           # Protocol version
        0,           # Host -> ESP32
        0x04,        # MOVE command
        len(payload),
        session_id,
        counter
    )

    packet = header + payload

    tag = hmac.digest(TEST_KEY, packet, hashlib.sha256)

    return packet + tag



def encode_serial_frame(packet):
    encoded = bytearray()
    code_index = 0
    code = 1

    encoded.append(0)

    for byte in packet:
        if byte == 0:
            encoded[code_index] = code
            code_index = len(encoded)
            encoded.append(0)
            code = 1
        else:
            encoded.append(byte)
            code += 1

            if code == 0xFF:
                encoded[code_index] = code
                code_index = len(encoded)
                encoded.append(0)
                code = 1

    encoded[code_index] = code
    encoded.append(0)

    return bytes(encoded)

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

    if __name__ == "__main__":
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




        payload = build_move_payload([90, 85, 100, 90])

        print("MOVE payload:", payload.hex())
        print("Decoded targets:", struct.unpack("<4h", payload))
