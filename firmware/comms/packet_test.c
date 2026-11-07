#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static bool has_valid_magic(const uint8_t *packet, size_t length)
{
    if (length < 2) {
        return false;
    }

    return packet[0] == 0x41 && packet[1] == 0x52;
}

static bool has_valid_packet_layout(const uint8_t *packet, size_t length)
{
    const size_t fixed_length = 59;  // 27-byte header + 32-byte HMAC tag

    if (length < fixed_length || !has_valid_magic(packet, length)) {
        return false;
    }

    if (packet[2] != 1) {
        return false;
    }

    // The payload length is stored in bytes 5 and 6, low byte first.
    size_t payload_length = (size_t)packet[5] + 256u * packet[6];

    return payload_length <= 64 && length == fixed_length + payload_length;
}


static bool has_valid_move_layout(const uint8_t *packet, size_t length)
{
    if (!has_valid_packet_layout(packet, length)) {
        return false;
    }

    return packet[3] == 0       // Laptop to ESP32
        && packet[4] == 0x04    // MOVE
        && packet[5] == 8       // Payload length, low byte
        && packet[6] == 0;      // Payload length, high byte
}

static bool decode_move_targets(const uint8_t *packet, size_t length,
                                int16_t targets[4])
{
    if (!has_valid_move_layout(packet, length)) {
        return false;
    }

    for (size_t i = 0; i < 4; i++) {
        size_t offset = 27 + 2 * i;
        int32_t value = packet[offset] + 256 * (int32_t)packet[offset + 1];

        if (value >= 32768) {
            value -= 65536;  // Interpret the two bytes as a signed value.
        }

        targets[i] = (int16_t)value;
    }

    return true;
}


static uint32_t read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24);
}


int main(void)
{
    const uint8_t correct[] = {0x41, 0x52};
    const uint8_t incorrect[] = {0x41, 0x53};
    const uint8_t truncated[] = {0x41};


    uint8_t empty_payload_packet[59] = {0x41, 0x52, 1};
    uint8_t wrong_version[59] = {0x41, 0x52, 2};
    uint8_t missing_payload[59] = {0x41, 0x52, 1, 0, 0x07, 1, 0};

    if (!has_valid_packet_layout(empty_payload_packet, sizeof empty_payload_packet) ||
        has_valid_packet_layout(wrong_version, sizeof wrong_version) ||
        has_valid_packet_layout(missing_payload, sizeof missing_payload)) {
        puts("FAIL: a packet layout check returned the wrong result");
        return 1;
    }

    puts("PASS: packet layout checks");

    
    const uint8_t move[67] = {
    0x41, 0x52, 1, 0, 0x04, 8, 0,
    [23] = 0x78, 0x56, 0x34, 0x12,  // Example counter: 0x12345678
    [27] = 0x28, 0x23,  //  90.00 degrees
           0x70, 0x17,  //  60.00 degrees
           0x50, 0xFB,  // -12.00 degrees
           0xB8, 0x0B   //  30.00 degrees
};

    const uint8_t old_move[69] = {0x41, 0x52, 1, 0, 0x04, 10, 0};

    if (!has_valid_move_layout(move, sizeof move) ||
        has_valid_move_layout(old_move, sizeof old_move)) {
        puts("FAIL: MOVE layout checks");
        return 1;
    }

    puts("PASS: MOVE layout checks");


    int16_t targets[4];

    if (!decode_move_targets(move, sizeof move, targets) ||
        targets[0] != 9000 || targets[1] != 6000 ||
        targets[2] != -1200 || targets[3] != 3000) {
        puts("FAIL: MOVE angle decoding");
        return 1;
    }

    puts("PASS: MOVE angle decoding");


        uint32_t counter = read_u32_le(&move[23]);

    if (counter != UINT32_C(0x12345678)) {
        puts("FAIL: counter decoding");
        return 1;
    }

    puts("PASS: counter decoding");

    if (has_valid_magic(correct, sizeof correct) &&
        !has_valid_magic(incorrect, sizeof incorrect) &&
        !has_valid_magic(truncated, sizeof truncated)) {
        puts("PASS: all three magic-byte checks");
        return 0;
    }

    puts("FAIL: a magic-byte check returned the wrong result");
    return 1;
}