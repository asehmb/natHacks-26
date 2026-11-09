#include <Arduino.h>
#include <esp_system.h>
#include "armlink.h"
#include "servo_bridge.h"

// Checkpoint 3B: mock backend ONLY. No servo attachment or GPIO writes.
// F1-confirmed policy: STOP disarms and detaches; timeout invalidates the
// session and disarms, preserving existing attachment and last valid targets.
// Physical output and ARM reattachment remain unimplemented.
static ReceiverState receiver;
static uint8_t key[32], incoming[128];
static size_t used;
static bool dropping;
static uint32_t lastHello, timeoutCount;
static ServoBridge<MockServoBackend> servos;
static uint32_t appliedMoves;

static void freshSession() {
  uint8_t session[16];
  esp_fill_random(session, sizeof session);
  prepare_receiver_session(&receiver, session);
}
static void transmit(const uint8_t *packet, size_t length) {
  uint8_t frame[128];
  size_t n = cobs_encode(packet, length, frame, sizeof frame);
  if (n) { Serial.write((uint8_t)0); Serial.write(frame, n); Serial.write((uint8_t)0); }
}
static void put32(uint8_t *out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out[i] = (uint8_t)(value >> (8*i));
}
static void reply(uint8_t kind, uint8_t command, uint8_t result, uint32_t counter) {
  // ACK payload: command, active, armed, result (0 accepted, 1 rejected).
  // STATE payload: active, armed, outputs_enabled, four commanded integer
  // degrees as int16, appliedMoves u32, timeoutCount u32. No measured feedback.
  uint8_t out[78] = {0x41, 0x52, 1, 1, kind, 0, 0};
  out[5] = kind == ARMLINK_STATE ? 19 : 4;
  memcpy(out + 7, receiver.session_id, 16);
  put32(out + 23, counter);
  if (kind == ARMLINK_STATE) {
    out[27] = receiver.session_active; out[28] = receiver.armed; out[29] = 0;
    for (int i=0; i<4; ++i) {
      out[30+2*i] = servos.positions[i] & 255;
      out[31+2*i] = servos.positions[i] >> 8;
    }
    put32(out+38, appliedMoves); put32(out+42, timeoutCount);
  } else {
    out[27] = command; out[28] = receiver.session_active;
    out[29] = receiver.armed; out[30] = result;
  }
  size_t length = 59 + out[5];
  if (calculate_packet_hmac(out, length, key, out + length-32)) transmit(out, length);
}
static void expire(uint32_t now) {
  if (check_receiver_timeout(&receiver, now)) {
    servos.timeout(); ++timeoutCount; freshSession();
  }
}
static void dispatch(uint8_t *packet, size_t length) {
  uint32_t now = millis();
  expire(now);
  if (receive_open_session(&receiver, packet, length, key, now)) {
    uint8_t ack[62];
    if (build_open_session_ack(&receiver, key, ack)) transmit(ack, sizeof ack);
    return;
  }
  if (!receiver.session_active || !has_valid_packet_layout(packet, length) ||
      packet[3] != 0) return;
  // Authenticate once on a candidate state. Commit only accepted commands.
  ReceiverState next = receiver;
  if (!accept_authenticated_packet(packet, length, key, receiver.session_id,
                                    &next.last_counter)) return;
  uint8_t kind = packet[4];
  uint32_t counter = next.last_counter;
  next.last_valid_ms = now;
  bool accepted = false;
  if (length == 59 && (kind == ARMLINK_ARM || kind == ARMLINK_STOP)) {
    if (kind == ARMLINK_STOP) servos.stop();
    next.armed = kind == ARMLINK_ARM; accepted = true;
  } else if (length == 59 && (kind == ARMLINK_GET_STATE || kind == ARMLINK_HEARTBEAT)) {
    accepted = true;
  } else if (kind == ARMLINK_MOVE && receiver.armed) {
    int16_t targets[4];
    if (decode_move_targets(packet, length, targets) && servos.apply(targets)) {
      ++appliedMoves;
      accepted = true;
    }
  }
  if (accepted) receiver = next;
  reply(kind == ARMLINK_GET_STATE && accepted ? ARMLINK_STATE : ARMLINK_ACK,
        kind, accepted ? 0 : 1, counter);
}
void setup() {
  memset(key, 0x0b, sizeof key); Serial.begin(115200);
  freshSession(); lastHello = millis()-250;
}
void loop() {
  uint32_t now = millis(); expire(now);
  if (!receiver.session_active && (uint32_t)(now-lastHello) >= 250) {
    uint8_t hello[59];
    if (build_hello_packet(receiver.session_id, key, hello)) transmit(hello, sizeof hello);
    lastHello = now;
  }
  while (Serial.available()) {
    uint8_t byte = (uint8_t)Serial.read();
    if (!byte) {
      if (!dropping && used) {
        uint8_t decoded[123];
        size_t n = cobs_decode(incoming, used, decoded, sizeof decoded);
        dispatch(decoded, n);
      }
      used=0; dropping=false;
    } else if (!dropping) {
      if (used < sizeof incoming) incoming[used++]=byte;
      else { used=0; dropping=true; }
    }
  }
}
