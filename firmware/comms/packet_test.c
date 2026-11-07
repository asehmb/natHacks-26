#include "armlink.h"
#include <stdio.h>
#include <string.h>
#include <mbedtls/md.h>

static bool test_hmac_sha256(void)
{
    uint8_t key[20];
    memset(key, 0x0B, sizeof key);

    const uint8_t message[] = "Hi There";
    const uint8_t expected[32] = {
        0xB0, 0x34, 0x4C, 0x61, 0xD8, 0xDB, 0x38, 0x53,
        0x5C, 0xA8, 0xAF, 0xCE, 0xAF, 0x0B, 0xF1, 0x2B,
        0x88, 0x1D, 0xC2, 0x00, 0xC9, 0x83, 0x3D, 0xA7,
        0x26, 0xE9, 0x37, 0x6C, 0x2E, 0x32, 0xCF, 0xF7
    };
    uint8_t actual[32];

    const mbedtls_md_info_t *sha256 =
        mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256 == NULL) {
        return false;
    }

    int result = mbedtls_md_hmac(
        sha256,
        key, sizeof key,
        message, sizeof message - 1,
        actual
    );
    if (result != 0) {
        return false;
    }

    return memcmp(actual, expected, sizeof expected) == 0;
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


    if (!test_hmac_sha256()) {
        puts("FAIL: HMAC-SHA256 known-answer test");
        return 1;
    }

    puts("PASS: HMAC-SHA256 known-answer test");


    // Public test key. Use a separately provisioned secret for the real arm.
    uint8_t test_key[32];
    memset(test_key, 0x0B, sizeof test_key);




    const uint8_t test_session_id[16] = {
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
};

ReceiverState open_state = {0};
prepare_receiver_session(&open_state, test_session_id);

uint8_t open_packet[59] = {
    0x41, 0x52, 1, 0, ARMLINK_OPEN_SESSION, 0, 0
};
memcpy(open_packet + 7, test_session_id, sizeof test_session_id);
open_packet[23] = 1; // Counter 1; remaining counter bytes are already zero.

uint8_t open_tag[32];
if (!calculate_packet_hmac(
        open_packet, sizeof open_packet, test_key, open_tag)) {
    puts("FAIL: could not calculate OPEN_SESSION HMAC");
    return 1;
}
memcpy(open_packet + 27, open_tag, sizeof open_tag);

uint8_t wrong_key[32];
memset(wrong_key, 0x0C, sizeof wrong_key);

if (receive_open_session(
        &open_state, open_packet, sizeof open_packet, wrong_key, 1200) ||
    open_state.session_active || open_state.last_counter != 0) {
    puts("FAIL: wrong-key OPEN_SESSION accepted or changed state");
    return 1;
}

if (!receive_open_session(
        &open_state, open_packet, sizeof open_packet, test_key, 1200) ||
    !open_state.session_active || open_state.armed ||
    open_state.last_counter != 1 || open_state.last_valid_ms != 1200) {
    puts("FAIL: valid OPEN_SESSION did not establish the session");
    return 1;
}

if (receive_open_session(
        &open_state, open_packet, sizeof open_packet, test_key, 1300) ||
    open_state.last_counter != 1) {
    puts("FAIL: duplicate OPEN_SESSION was accepted");
    return 1;
}

puts("PASS: OPEN_SESSION authentication and counter handling");


    uint8_t hello_packet[59];

    if (!build_hello_packet(test_session_id, test_key, hello_packet) ||
        !has_valid_packet_layout(hello_packet, sizeof hello_packet) ||
        hello_packet[3] != 1 ||
        hello_packet[4] != ARMLINK_HELLO ||
        memcmp(hello_packet + 7, test_session_id, sizeof test_session_id) != 0 ||
        read_u32_le(hello_packet + 23) != 0 ||
        !verify_packet_hmac(hello_packet, sizeof hello_packet, test_key)) {
        puts("FAIL: HELLO packet construction");
        return 1;
    }

puts("PASS: HELLO packet construction");


     uint8_t hello_frame[61];
    size_t hello_frame_len = cobs_encode(
        hello_packet, sizeof hello_packet, hello_frame, sizeof hello_frame - 1);

    if (hello_frame_len == 0) {
        puts("FAIL: could not frame HELLO");
        return 1;
    }

    hello_frame[hello_frame_len++] = 0;

    printf("HELLO frame hex: ");
    for (size_t i = 0; i < hello_frame_len; i++) {
        printf("%02x", (unsigned)hello_frame[i]);
    }
    putchar('\n');

        const uint8_t python_reply_frame[] = {
        0x04, 0x41, 0x52, 0x01, 0x02, 
        0x02, 0x01, 0x12, 0x10, 0x11, 
        0x12, 0x13, 0x14, 0x15, 0x16, 
        0x17, 0x18, 0x19, 0x1a, 0x1b, 
        0x1c, 0x1d, 0x1e, 0x1f, 0x01, 
        0x01, 0x01, 0x21, 0x09, 0xfb, 
        0x91, 0x43,0xda, 0xf7, 0xf0, 
        0x30, 0xd7, 0xfd, 0xaa, 0xbe,
        0x98, 0x5a, 0x3e, 0x31, 0x9c, 
        0xe1, 0xdd, 0x12, 0x6c, 0x96, 
        0x2a, 0xbc, 0x34, 0x0a, 0x8c, 
        0x4f, 0x5d, 0xbf, 0xd0, 0x06, 
        0x00
    };

    uint8_t python_reply[59];
    size_t frame_len = sizeof python_reply_frame;

    if (python_reply_frame[frame_len - 1] != 0 ||
        cobs_decode(python_reply_frame, frame_len - 1,
                    python_reply, sizeof python_reply) != sizeof python_reply) {
        puts("FAIL: could not decode Python OPEN_SESSION frame");
        return 1;
    }

    ReceiverState python_receiver = {0};
    prepare_receiver_session(&python_receiver, test_session_id);

    if (!receive_open_session(
            &python_receiver, python_reply, sizeof python_reply, test_key, 1200) ||
        !python_receiver.session_active ||
        python_receiver.armed ||
        python_receiver.last_counter != 1) {
        puts("FAIL: C receiver rejected Python OPEN_SESSION or set the wrong state");
        return 1;
    }

    puts("PASS: C receiver accepted Python OPEN_SESSION");

const uint8_t expected_packet_tag[32] = {
        0x02, 0x8A, 0xC0, 0x0F, 0xB4, 0xE5, 0x4F, 0x6F,
        0x0D, 0xAE, 0x94, 0xEF, 0x15, 0xF8, 0x70, 0x5D,
        0x19, 0xF8, 0x2C, 0x26, 0xC2, 0xAD, 0x21, 0x0B,
        0x65, 0x22, 0xDC, 0x82, 0x5D, 0xA3, 0xF6, 0x84
    };

    uint8_t packet_tag[32];

    if (!calculate_packet_hmac(move, sizeof move, test_key, packet_tag) ||
        memcmp(packet_tag, expected_packet_tag, sizeof packet_tag) != 0) {
        puts("FAIL: MOVE packet HMAC calculation");
        return 1;
    }

puts("PASS: MOVE packet HMAC calculation");

    uint8_t authenticated_move[sizeof move];
    memcpy(authenticated_move, move, sizeof move);

    // Attach the tag we calculated in the previous test.
    memcpy(authenticated_move + sizeof authenticated_move - 32,
           packet_tag, sizeof packet_tag);

    if (!verify_packet_hmac(authenticated_move,
                            sizeof authenticated_move, test_key)) {
        puts("FAIL: authentic packet rejected");
        return 1;
    }

    

    if (verify_packet_hmac(authenticated_move,
                           sizeof authenticated_move, wrong_key)) {
        puts("FAIL: wrong key accepted");
        return 1;
    }

    // Change an angle without updating the tag.
    authenticated_move[27] ^= 0x01;

    if (verify_packet_hmac(authenticated_move,
                           sizeof authenticated_move, test_key)) {
        puts("FAIL: altered angle accepted");
        return 1;
    }

    // Restore the angle, then change the tag itself.
    authenticated_move[27] ^= 0x01;
    authenticated_move[sizeof authenticated_move - 1] ^= 0x01;

    if (verify_packet_hmac(authenticated_move,
                           sizeof authenticated_move, test_key)) {
        puts("FAIL: altered tag accepted");
        return 1;
    }

    puts("PASS: packet authentication and tampering checks");


        // Simulate an established session matching our example packet.
    const uint8_t expected_session[16] = {0};
    uint32_t last_counter = 0;

    // Make a fresh copy because the previous test modified its packet.
    uint8_t replay_packet[sizeof move];
    memcpy(replay_packet, move, sizeof move);
    memcpy(replay_packet + sizeof replay_packet - 32,
           packet_tag, sizeof packet_tag);

    if (!accept_authenticated_packet(
            replay_packet, sizeof replay_packet, test_key,
            expected_session, &last_counter) ||
        last_counter != UINT32_C(0x12345678)) {
        puts("FAIL: fresh authenticated packet rejected");
        return 1;
    }

    // Send the identical packet again.
    if (accept_authenticated_packet(
            replay_packet, sizeof replay_packet, test_key,
            expected_session, &last_counter)) {
        puts("FAIL: replayed packet accepted");
        return 1;
    }

    // Forge a higher counter without updating its HMAC.
    replay_packet[23] = 0x79;

    if (accept_authenticated_packet(
            replay_packet, sizeof replay_packet, test_key,
            expected_session, &last_counter) ||
        last_counter != UINT32_C(0x12345678)) {
        puts("FAIL: forged counter accepted or changed receiver state");
        return 1;
    }

    puts("PASS: replay protection checks");


    


        const struct {
        const char *name;
        uint8_t counter_low;
        uint8_t session_first;
    } rejection_cases[] = {
        {"older authenticated counter", 0x77, 0x00},
        {"different authenticated session", 0x79, 0x01}
    };

    for (size_t i = 0; i < 2; i++) {
        uint8_t rejected_packet[sizeof move];
        uint8_t new_tag[32];

        memcpy(rejected_packet, move, sizeof move);

        rejected_packet[23] = rejection_cases[i].counter_low;
        rejected_packet[7] = rejection_cases[i].session_first;

        // Give the modified packet a valid HMAC.
        if (!calculate_packet_hmac(
                rejected_packet, sizeof rejected_packet, test_key, new_tag)) {
            puts("FAIL: could not create authenticated test packet");
            return 1;
        }

        memcpy(rejected_packet + sizeof rejected_packet - 32,
               new_tag, sizeof new_tag);

        // Confirm authentication succeeds before testing the other rules.
        if (!verify_packet_hmac(
                rejected_packet, sizeof rejected_packet, test_key)) {
            puts("FAIL: test packet has an invalid HMAC");
            return 1;
        }

        if (accept_authenticated_packet(
                rejected_packet, sizeof rejected_packet, test_key,
                expected_session, &last_counter) ||
            last_counter != UINT32_C(0x12345678)) {
            printf("FAIL: %s\n", rejection_cases[i].name);
            return 1;
        }

        printf("PASS: rejected %s\n", rejection_cases[i].name);
    }


        if (communication_timed_out(1499, 1000)) {
        puts("FAIL: watchdog triggered before 500 ms");
        return 1;
    }

    if (!communication_timed_out(1500, 1000)) {
        puts("FAIL: watchdog missed the 500 ms boundary");
        return 1;
    }

    if (!communication_timed_out(1600, 1000)) {
        puts("FAIL: watchdog missed an overdue timeout");
        return 1;
    }

    // Simulate the 32-bit timer wrapping back to zero.
    if (!communication_timed_out(400, UINT32_MAX - 99)) {
        puts("FAIL: watchdog did not handle timer rollover");
        return 1;
    }

    puts("PASS: communication watchdog timing");


        // Simulate a session that is armed and last communicated at 1000 ms.
    ReceiverState receiver = {
        .session_active = true,
        .armed = true,
        .last_valid_ms = 1000,
        .last_counter = 0,
        .session_id = {0}
    };

    if (check_receiver_timeout(&receiver, 1499) ||
        !receiver.armed || !receiver.session_active) {
        puts("FAIL: receiver disarmed before timeout");
        return 1;
    }

    if (!check_receiver_timeout(&receiver, 1500) ||
        receiver.armed || receiver.session_active) {
        puts("FAIL: timeout did not disarm and invalidate the session");
        return 1;
    }

    // The timeout event should be reported only once.
    if (check_receiver_timeout(&receiver, 1600)) {
        puts("FAIL: inactive session generated another timeout event");
        return 1;
    }

    puts("PASS: timeout updates receiver state");


        ReceiverState heartbeat_receiver = {
        .session_active = true,
        .armed = true,
        .last_valid_ms = 1000,
        .last_counter = 1,
        .session_id = {0}
    };

    uint8_t heartbeat[59] = {
        0x41, 0x52, 1, 0, 0x06, 0, 0,
        [23] = 2
    };
    uint8_t heartbeat_tag[32];

    if (!calculate_packet_hmac(
            heartbeat, sizeof heartbeat, test_key, heartbeat_tag)) {
        puts("FAIL: could not authenticate heartbeat");
        return 1;
    }

    memcpy(heartbeat + 27, heartbeat_tag, sizeof heartbeat_tag);

    // A fresh authenticated heartbeat refreshes the timer.
    if (!receive_heartbeat(
            &heartbeat_receiver, heartbeat, sizeof heartbeat, test_key, 1200) ||
        heartbeat_receiver.last_valid_ms != 1200 ||
        heartbeat_receiver.last_counter != 2) {
        puts("FAIL: valid heartbeat did not update receiver");
        return 1;
    }

    // Replaying that heartbeat must not refresh the timer.
    if (receive_heartbeat(
            &heartbeat_receiver, heartbeat, sizeof heartbeat, test_key, 1400) ||
        heartbeat_receiver.last_valid_ms != 1200 ||
        heartbeat_receiver.last_counter != 2) {
        puts("FAIL: replayed heartbeat changed receiver state");
        return 1;
    }

    // Increase the counter without updating the HMAC.
       
    heartbeat[23] = 3;

    if (receive_heartbeat(
            &heartbeat_receiver, heartbeat, sizeof heartbeat, test_key, 1500) ||
        heartbeat_receiver.last_valid_ms != 1200 ||
        heartbeat_receiver.last_counter != 2) {
        puts("FAIL: forged heartbeat changed receiver state");
        return 1;
    }

     // Now give counter 3 a valid HMAC, but deliver it at expiry.
    if (!calculate_packet_hmac(
            heartbeat, sizeof heartbeat, test_key, heartbeat_tag)) {
        puts("FAIL: could not authenticate late heartbeat");
        return 1;
    }

    memcpy(heartbeat + 27, heartbeat_tag, sizeof heartbeat_tag);

    if (receive_heartbeat(
            &heartbeat_receiver, heartbeat, sizeof heartbeat, test_key, 1700) ||
        heartbeat_receiver.last_valid_ms != 1200 ||
        heartbeat_receiver.last_counter != 2) {
        puts("FAIL: late heartbeat revived expired session");
        return 1;
    }

    if (!check_receiver_timeout(&heartbeat_receiver, 1700) ||
        heartbeat_receiver.armed || heartbeat_receiver.session_active) {
        puts("FAIL: receiver did not disarm after heartbeat timeout");
        return 1;
    }

    puts("PASS: heartbeat refresh and timeout enforcement");



        ReceiverState control_receiver = {
        .session_active = true,
        .armed = false,
        .last_valid_ms = 1000,
        .last_counter = 1,
        .session_id = {0}
    };

    const struct {
        const char *name;
        uint8_t command;
        uint8_t counter;
        uint32_t time_ms;
        bool should_accept;
        bool should_be_armed;
        uint32_t expected_counter;
        uint32_t expected_time;
    } control_cases[] = {
        {"ARM",          0x03, 2, 1100, true,  true,  2, 1100},
        {"STOP",         0x05, 3, 1200, true,  false, 3, 1200},
        {"replayed ARM", 0x03, 2, 1300, false, false, 3, 1200}
    };

    for (size_t i = 0; i < 3; i++) {
        uint8_t control_packet[59] = {0x41, 0x52, 1, 0};
        uint8_t control_tag[32];

        control_packet[4] = control_cases[i].command;
        control_packet[23] = control_cases[i].counter;

        if (!calculate_packet_hmac(
                control_packet, sizeof control_packet, test_key, control_tag)) {
            puts("FAIL: could not authenticate control packet");
            return 1;
        }

        memcpy(control_packet + 27, control_tag, sizeof control_tag);

        bool accepted = receive_arm_or_stop(
            &control_receiver, control_packet, sizeof control_packet,
            test_key, control_cases[i].time_ms);

        if (accepted != control_cases[i].should_accept ||
            control_receiver.armed != control_cases[i].should_be_armed ||
            control_receiver.last_counter != control_cases[i].expected_counter ||
            control_receiver.last_valid_ms != control_cases[i].expected_time) {
            printf("FAIL: control command %s\n", control_cases[i].name);
            return 1;
        }
    }

    puts("PASS: ARM, STOP, and replayed ARM handling");


    ReceiverState move_receiver = {
        .session_active = true,
        .armed = false,
        .last_valid_ms = 1000,
        .last_counter = 0,
        .session_id = {0}
    };

    uint8_t valid_move[sizeof move];
    memcpy(valid_move, move, sizeof move);
    memcpy(valid_move + sizeof valid_move - 32,
           packet_tag, sizeof packet_tag);

    const int16_t unchanged_targets[4] = {111, 222, 333, 444};
    int16_t received_targets[4];
    memcpy(received_targets, unchanged_targets, sizeof received_targets);

    // A disarmed receiver must reject movement without changing its output.
    if (receive_move(
            &move_receiver, valid_move, sizeof valid_move,
            test_key, 1100, received_targets) ||
        memcmp(received_targets, unchanged_targets,
               sizeof received_targets) != 0 ||
        move_receiver.last_counter != 0 ||
        move_receiver.last_valid_ms != 1000) {
        puts("FAIL: disarmed receiver accepted movement or changed state");
        return 1;
    }

    // Simulate successful arming, which we tested separately.
    move_receiver.armed = true;

    if (!receive_move(
            &move_receiver, valid_move, sizeof valid_move,
            test_key, 1200, received_targets) ||
        received_targets[0] != 9000 ||
        received_targets[1] != 6000 ||
        received_targets[2] != -1200 ||
        received_targets[3] != 3000 ||
        move_receiver.last_counter != UINT32_C(0x12345678) ||
        move_receiver.last_valid_ms != 1200) {
        puts("FAIL: armed receiver did not accept valid MOVE");
        return 1;
    }

    // A duplicate must not execute again or refresh the watchdog.
    if (receive_move(
            &move_receiver, valid_move, sizeof valid_move,
            test_key, 1300, received_targets) ||
        move_receiver.last_valid_ms != 1200) {
        puts("FAIL: replayed MOVE accepted or refreshed watchdog");
        return 1;
    }

    puts("PASS: armed MOVE reception");

    uint8_t encoded_frame[128];
    size_t encoded_len = cobs_encode(
        valid_move, sizeof valid_move, encoded_frame, sizeof encoded_frame);
    if (encoded_len == 0) {
        puts("FAIL: COBS encode failed on valid_move");
        return 1;
    }

    // Encoded contents must contain no 0x00 bytes before the delimiter.
    for (size_t i = 0; i < encoded_len; i++) {
        if (encoded_frame[i] == 0x00) {
            puts("FAIL: COBS encoded frame contained a zero byte");
            return 1;
        }
    }

    uint8_t decoded_packet[128];
    size_t decoded_len = cobs_decode(
        encoded_frame, encoded_len, decoded_packet, sizeof decoded_packet);
    if (decoded_len != sizeof valid_move ||
        memcmp(decoded_packet, valid_move, sizeof valid_move) != 0) {
        puts("FAIL: COBS decode did not match original MOVE packet");
        return 1;
    }

    // Truncated or zero-corrupted frames must be rejected.
    if (cobs_decode(encoded_frame, encoded_len - 1, decoded_packet, sizeof decoded_packet) != 0) {
        puts("FAIL: truncated COBS frame accepted");
        return 1;
    }
    encoded_frame[3] = 0x00;
    if (cobs_decode(encoded_frame, encoded_len, decoded_packet, sizeof decoded_packet) != 0) {
        puts("FAIL: COBS frame with embedded zero accepted");
        return 1;
    }
    puts("PASS: COBS framing encode and decode");


    


    if (has_valid_magic(correct, sizeof correct) &&
        !has_valid_magic(incorrect, sizeof incorrect) &&
        !has_valid_magic(truncated, sizeof truncated)) {
        puts("PASS: all three magic-byte checks");
        return 0;
    }

    puts("FAIL: a magic-byte check returned the wrong result");
    return 1;
}