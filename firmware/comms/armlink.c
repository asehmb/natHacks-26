#include "armlink.h"
#include <string.h>
#include <mbedtls/md.h>
#include <mbedtls/constant_time.h>

bool has_valid_magic(const uint8_t *packet, size_t length)
{
    if (length < 2) {
        return false;
    }
    return packet[0] == 0x41 && packet[1] == 0x52;
}

bool has_valid_packet_layout(const uint8_t *packet, size_t length)
{
    const size_t fixed_length = 59; // 27-byte header + 32-byte HMAC tag

    if (length < fixed_length || !has_valid_magic(packet, length)) {
        return false;
    }

    if (packet[2] != 1) {
        return false;
    }

    size_t payload_length = (size_t)packet[5] + 256u * packet[6];
    return payload_length <= 64 && length == fixed_length + payload_length;
}

bool has_valid_move_layout(const uint8_t *packet, size_t length)
{
    if (!has_valid_packet_layout(packet, length)) {
        return false;
    }

    return packet[3] == 0      // Laptop to ESP32
        && packet[4] == 0x04   // MOVE
        && packet[5] == 8      // Payload length, low byte
        && packet[6] == 0;     // Payload length, high byte
}

bool decode_move_targets(const uint8_t *packet, size_t length, int16_t targets[4])
{
    if (!has_valid_move_layout(packet, length)) {
        return false;
    }

    for (size_t i = 0; i < 4; i++) {
        size_t offset = 27 + 2 * i;
        int32_t value = packet[offset] + 256 * (int32_t)packet[offset + 1];
        if (value >= 32768) {
            value -= 65536;
        }
        targets[i] = (int16_t)value;
    }

    return true;
}

uint32_t read_u32_le(const uint8_t *bytes)
{
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24);
}

bool calculate_packet_hmac(const uint8_t *packet, size_t length,
                           const uint8_t key[32], uint8_t tag[32])
{
    if (!has_valid_packet_layout(packet, length)) {
        return false;
    }

    const mbedtls_md_info_t *sha256 =
        mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256 == NULL) {
        return false;
    }

    size_t authenticated_length = length - 32;
    return mbedtls_md_hmac(
        sha256,
        key, 32,
        packet, authenticated_length,
        tag
    ) == 0;
}

bool verify_packet_hmac(const uint8_t *packet, size_t length,
                        const uint8_t key[32])
{
    uint8_t calculated_tag[32];
    if (!calculate_packet_hmac(packet, length, key, calculated_tag)) {
        return false;
    }

    const uint8_t *received_tag = packet + length - 32;
    return mbedtls_ct_memcmp(
        calculated_tag, received_tag, sizeof calculated_tag
    ) == 0;
}

bool accept_authenticated_packet(const uint8_t *packet, size_t length,
                                 const uint8_t key[32],
                                 const uint8_t expected_session[16],
                                 uint32_t *last_counter)
{
    if (!verify_packet_hmac(packet, length, key)) {
        return false;
    }

    if (mbedtls_ct_memcmp(packet + 7, expected_session, 16) != 0) {
        return false;
    }

    uint32_t incoming_counter = read_u32_le(packet + 23);
    if (incoming_counter <= *last_counter) {
        return false;
    }

    *last_counter = incoming_counter;
    return true;
}

bool communication_timed_out(uint32_t now_ms, uint32_t last_valid_ms)
{
    const uint32_t timeout_ms = 500;
    uint32_t elapsed_ms = now_ms - last_valid_ms;
    return elapsed_ms >= timeout_ms;
}

bool check_receiver_timeout(ReceiverState *state, uint32_t now_ms)
{
    if (!state->session_active) {
        return false;
    }

    if (!communication_timed_out(now_ms, state->last_valid_ms)) {
        return false;
    }

    state->armed = false;
    state->session_active = false;
    return true;
}


void prepare_receiver_session(
    ReceiverState *state,
    const uint8_t fresh_session_id[16])
{
    if (state == NULL || fresh_session_id == NULL) {
        return;
    }

    memcpy(state->session_id, fresh_session_id, sizeof state->session_id);
    state->session_active = false;
    state->armed = false;
    state->last_valid_ms = 0;
    state->last_counter = 0;
}

bool receive_open_session(
    ReceiverState *state,
    const uint8_t *packet,
    size_t length,
    const uint8_t key[32],
    uint32_t now_ms)
{
    if (state == NULL || packet == NULL || key == NULL ||
        state->session_active || state->last_counter != 0) {
        return false;
    }

    if (!has_valid_packet_layout(packet, length) ||
        packet[3] != 0 ||
        packet[4] != ARMLINK_OPEN_SESSION ||
        packet[5] != 0 || packet[6] != 0 ||
        read_u32_le(packet + 23) != 1) {
        return false;
    }

    if (!accept_authenticated_packet(
            packet, length, key, state->session_id, &state->last_counter)) {
        return false;
    }

    state->session_active = true;
    state->armed = false;
    state->last_valid_ms = now_ms;
    return true;
}


bool build_hello_packet(
    const uint8_t session_id[16],
    const uint8_t key[32],
    uint8_t packet[59])
{
    if (session_id == NULL || key == NULL || packet == NULL) {
        return false;
    }

    // Version 1, ESP32 to laptop, HELLO, empty payload.
    // The counter bytes are initialized to zero.
    uint8_t hello[59] = {0x41, 0x52, 1, 1, ARMLINK_HELLO, 0, 0};
    memcpy(hello + 7, session_id, 16);

    if (!calculate_packet_hmac(hello, sizeof hello, key, hello + 27)) {
        return false;
    }

    memcpy(packet, hello, sizeof hello);
    return true;
}

bool build_open_session_ack(const ReceiverState *state, const uint8_t key[32],
                            uint8_t packet[62])
{
    if (!state || !key || !packet || !state->session_active || state->armed ||
        state->last_counter != 1) return false;
    uint8_t ack[62] = {0x41, 0x52, 1, 1, ARMLINK_ACK, 3, 0};
    memcpy(ack + 7, state->session_id, 16);
    for (size_t i = 0; i < 4; ++i)
        ack[23 + i] = (uint8_t)(state->last_counter >> (8 * i));
    ack[27] = ARMLINK_OPEN_SESSION;
    ack[28] = 1;
    ack[29] = 0;
    if (!calculate_packet_hmac(ack, sizeof ack, key, ack + 30)) return false;
    memcpy(packet, ack, sizeof ack);
    return true;
}

bool receive_heartbeat(ReceiverState *state, const uint8_t *packet, size_t length,
                       const uint8_t key[32], uint32_t now_ms)
{
    if (!state->session_active ||
        communication_timed_out(now_ms, state->last_valid_ms)) {
        return false;
    }

    if (!has_valid_packet_layout(packet, length)) {
        return false;
    }

    if (packet[3] != 0 || packet[4] != 0x06 || length != 59) {
        return false;
    }

    if (!accept_authenticated_packet(
            packet, length, key, state->session_id, &state->last_counter)) {
        return false;
    }

    state->last_valid_ms = now_ms;
    return true;
}

bool receive_arm_or_stop(ReceiverState *state, const uint8_t *packet, size_t length,
                         const uint8_t key[32], uint32_t now_ms)
{
    if (!state->session_active ||
        communication_timed_out(now_ms, state->last_valid_ms)) {
        return false;
    }

    if (!has_valid_packet_layout(packet, length)) {
        return false;
    }

    if (packet[3] != 0 || length != 59) {
        return false;
    }

    uint8_t command = packet[4];
    if (command != 0x03 && command != 0x05) {
        return false;
    }

    if (!accept_authenticated_packet(
            packet, length, key, state->session_id, &state->last_counter)) {
        return false;
    }

    state->armed = (command == 0x03);
    state->last_valid_ms = now_ms;
    return true;
}

bool receive_move(ReceiverState *state, const uint8_t *packet, size_t length,
                  const uint8_t key[32], uint32_t now_ms,
                  int16_t output_targets[4])
{
    if (!state->session_active || !state->armed ||
        communication_timed_out(now_ms, state->last_valid_ms)) {
        return false;
    }

    int16_t decoded_targets[4];
    if (!decode_move_targets(packet, length, decoded_targets)) {
        return false;
    }

    if (!accept_authenticated_packet(
            packet, length, key, state->session_id, &state->last_counter)) {
        return false;
    }

    memcpy(output_targets, decoded_targets, sizeof decoded_targets);
    state->last_valid_ms = now_ms;
    return true;
}

size_t cobs_encode(const uint8_t *input, size_t length,
                   uint8_t *output, size_t max_output)
{
    if (length == 0 || max_output == 0) {
        return 0;
    }

    size_t read_index = 0;
    size_t write_index = 1;
    size_t code_index = 0;
    uint8_t code = 1;

    while (read_index < length) {
        if (input[read_index] == 0) {
            output[code_index] = code;
            code = 1;
            code_index = write_index++;
            if (code_index >= max_output && read_index + 1 < length) {
                return 0;
            }
            read_index++;
        } else {
            if (write_index >= max_output) {
                return 0;
            }
            output[write_index++] = input[read_index++];
            code++;
            if (code == 0xFF) {
                output[code_index] = code;
                code = 1;
                code_index = write_index++;
                if (code_index >= max_output && read_index < length) {
                    return 0;
                }
            }
        }
    }

    if (code_index >= max_output) {
        return 0;
    }
    output[code_index] = code;
    return write_index;
}

size_t cobs_decode(const uint8_t *input, size_t length,
                   uint8_t *output, size_t max_output)
{
    if (length == 0) {
        return 0;
    }

    size_t read_index = 0;
    size_t write_index = 0;

    while (read_index < length) {
        uint8_t code = input[read_index++];
        if (code == 0 || read_index + code - 1 > length) {
            return 0;
        }

        for (uint8_t i = 1; i < code; i++) {
            uint8_t byte = input[read_index++];
            if (byte == 0 || write_index >= max_output) {
                return 0;
            }
            output[write_index++] = byte;
        }

        if (code < 0xFF && read_index < length) {
            if (write_index >= max_output) {
                return 0;
            }
            output[write_index++] = 0;
        }
    }

    return write_index;
}
