#ifndef SERVO_BRIDGE_H
#define SERVO_BRIDGE_H
#include <stdint.h>
// F1 order: base GPIO18, shoulder GPIO19, elbow GPIO21, gripper GPIO22.
// Validate every centidegree target BEFORE converting or applying any target.
static inline bool validate_targets(const int16_t targets[4], int degrees[4]) {
  const int lower[4] = {3000, 6000, 4500, 7000};
  const int upper[4] = {15000, 12000, 13500, 11000};
  for (int i = 0; i < 4; ++i)
    if (targets[i] < lower[i] || targets[i] > upper[i]) return false;
  // ESP32Servo.write uses integer degrees. Round positive values to nearest.
  for (int i = 0; i < 4; ++i) degrees[i] = (targets[i] + 50) / 100;
  return true;
}

// Mock backend only. Records interface calls; never touches GPIO or PWM.
// attached[] represents simulated attachment, never physical attachment.
struct MockServoBackend {
  bool attached[4] = {false, false, false, false};
  int targets[4] = {90, 90, 90, 90};
  uint32_t writes = 0;
  uint32_t detachRequests = 0;
  void write(int joint, int degrees) {
    targets[joint] = degrees;
    ++writes;
  }
  void detachAll() {
    ++detachRequests;
    for (int i = 0; i < 4; ++i) attached[i] = false;
  }
};

template<class Backend>
class ServoBridge {
 public:
  Backend backend;
  int positions[4] = {90, 90, 90, 90};
  bool apply(const int16_t targets[4]) {
    int degrees[4];
    if (!validate_targets(targets, degrees)) return false;
    // Validation completes before the first write. F1 joint order is 0..3.
    for (int i = 0; i < 4; ++i) backend.write(i, degrees[i]);
    for (int i = 0; i < 4; ++i) positions[i] = degrees[i];
    return true;
  }
  void stop() { backend.detachAll(); }
  // Preserve attachment AND targets. In particular, never reattach after STOP.
  void timeout() {}
};
#endif
