#ifndef ARMLINK_H
#define ARMLINK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    ARMLINK_HELLO        = 0x01,
    ARMLINK_OPEN_SESSION = 0x02,
    ARMLINK_ARM          = 0x03,
    ARMLINK_MOVE         = 0x04,
    ARMLINK_STOP         = 0x05,
    ARMLINK_HEARTBEAT    = 0x06,
    ARMLINK_GET_STATE    = 0x07,
    ARMLINK_ACK          = 0x80,
    ARMLINK_STATE        = 0x81
};

typedef struct {
    bool session_active;
    bool armed;
    uint32_t last_valid_ms;
    uint32_t last_counter;
    uint8_t session_id[16];
} ReceiverState;

bool has_valid_magic(const uint8_t *packet, size_t length);
bool has_valid_packet_layout(const uint8_t *packet, size_t length);
bool has_valid_move_layout(const uint8_t *packet, size_t length);
bool decode_move_targets(const uint8_t *packet, size_t length, int16_t targets[4]);
uint32_t read_u32_le(const uint8_t *bytes);

bool calculate_packet_hmac(const uint8_t *packet, size_t length,const uint8_t key[32], uint8_t tag[32]);
bool verify_packet_hmac(const uint8_t *packet, size_t length,const uint8_t key[32]);
bool accept_authenticated_packet(const uint8_t *packet, size_t length,const uint8_t key[32],
const uint8_t expected_session[16],
uint32_t *last_counter);




bool communication_timed_out(uint32_t now_ms, uint32_t last_valid_ms);
bool check_receiver_timeout(ReceiverState *state, uint32_t now_ms);

bool receive_heartbeat(ReceiverState *state, const uint8_t *packet, size_t length,
                       const uint8_t key[32], uint32_t now_ms);
bool receive_arm_or_stop(ReceiverState *state, const uint8_t *packet, size_t length,
                         const uint8_t key[32], uint32_t now_ms);
bool receive_move(ReceiverState *state, const uint8_t *packet, size_t length,
                  const uint8_t key[32], uint32_t now_ms,
                  int16_t output_targets[4]);


void prepare_receiver_session(
    ReceiverState *state,
    const uint8_t fresh_session_id[16]);

bool receive_open_session(
    ReceiverState *state,
    const uint8_t *packet,
    size_t length,
    const uint8_t key[32],
    uint32_t now_ms);

    
size_t cobs_encode(const uint8_t *input, size_t length,
                   uint8_t *output, size_t max_output);
size_t cobs_decode(const uint8_t *input, size_t length,
                   uint8_t *output, size_t max_output);

bool build_hello_packet(
    const uint8_t session_id[16],
    const uint8_t key[32],
    uint8_t packet[59]);

#endif