#include <Arduino.h>
#include <esp_system.h>
#include "armlink.h"

// No servo library, GPIO outputs, ARM dispatcher, or MOVE dispatcher.
static ReceiverState receiver;
static uint8_t key[32];
static uint8_t incoming[128];
static size_t used;
static bool dropping;
static uint32_t lastHello;

static void freshSession() {
  uint8_t session[16];
  esp_fill_random(session, sizeof session);
  prepare_receiver_session(&receiver, session);
}

static void transmit(const uint8_t *packet, size_t length) {
  uint8_t frame[128];
  size_t count = cobs_encode(packet, length, frame, sizeof frame);
  if (count) {
    // Leading delimiter also separates ROM boot text from binary frames.
    Serial.write((uint8_t)0);
    Serial.write(frame, count);
    Serial.write((uint8_t)0);
  }
}

void setup() {
  memset(key, 0x0b, sizeof key); // Shared development test key.
  Serial.begin(115200);
  freshSession();
  lastHello = millis() - 250;
}

void loop() {
  uint32_t now = millis();
  if (check_receiver_timeout(&receiver, now)) freshSession();
  if (!receiver.session_active && (uint32_t)(now - lastHello) >= 250) {
    uint8_t hello[59];
    if (build_hello_packet(receiver.session_id, key, hello))
      transmit(hello, sizeof hello);
    lastHello = now;
  }
  while (Serial.available()) {
    uint8_t byte = (uint8_t)Serial.read();
    if (byte == 0) {
      if (!dropping && used) {
        uint8_t packet[123];
        size_t length = cobs_decode(incoming, used, packet, sizeof packet);
        now = millis();
        if (check_receiver_timeout(&receiver, now)) freshSession();
        if (receive_open_session(&receiver, packet, length, key, now)) {
          uint8_t ack[62];
          if (build_open_session_ack(&receiver, key, ack)) transmit(ack, sizeof ack);
        }
      }
      used = 0;
      dropping = false;
    } else if (!dropping) {
      if (used < sizeof incoming) incoming[used++] = byte;
      else { dropping = true; used = 0; }
    }
  }
}
