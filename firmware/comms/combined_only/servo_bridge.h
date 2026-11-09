
#ifndef SERVO_BRIDGE_H
#define SERVO_BRIDGE_H

#include <stdint.h>
#include <ESP32Servo.h>

// Joint order shared by ArmLink MOVE and STATE:
// 0 = base, 1 = shoulder, 2 = elbow, 3 = gripper.
// These limits are *commanded servo angles*, not kinematic joint radians.
// Confirm the limits against the assembled arm before enabling real PWM.
static constexpr int kJointCount = 4;
static constexpr int kReferenceDegrees = 90;
static constexpr int kMinDegrees[kJointCount] = {0, 20, 15, 0};
static constexpr int kMaxDegrees[kJointCount] = {180, 160, 130, 180};

// MOVE transmits four signed little-endian centidegree values.
// All four limits must pass before any backend writes occur.
static inline bool validate_targets(const int16_t targets[4], int degrees[4]) {
  if (!targets || !degrees) return false;
  for (int i = 0; i < kJointCount; ++i) {
    const int32_t value = targets[i];
    if (value < kMinDegrees[i] * 100L || value > kMaxDegrees[i] * 100L)
      return false;
  }
  for (int i = 0; i < kJointCount; ++i) {
    // All configured ranges are nonnegative. Round to nearest whole degree.
    degrees[i] = (static_cast<int32_t>(targets[i]) + 50) / 100;
  }
  return true;
}

// The combined_only firmware currently uses this mock backend.
// It has NO connections to ESP32Servo, PWM, or physical GPIO.
struct MockServoBackend {
  bool attached[kJointCount] = {false, false, false, false};
  int targets[kJointCount] = {90, 90, 90, 90};
  uint32_t writes = 0;
  uint32_t detachRequests = 0;

  bool enabled[kJointCount] = {
    true, true, true, true
  };

  void write(int joint, int degrees) {
    targets[joint] = degrees;
    ++writes;
  }
  void detachAll() {
    ++detachRequests;
    for (int i = 0; i < kJointCount; ++i) attached[i] = false;
  }
};


struct Esp32ServoBackend {
  Servo motors[kJointCount];
  bool motionAllowed = false;
  const int pins[kJointCount] = {
    18, 19, 21, 22
  };


  int currentAngles[kJointCount] = {
      90, 90, 90, 90
  };

  int desiredAngles[kJointCount] = {
      90, 90, 90, 90
  };

  unsigned long lastUpdate = 0;

  // One commanded degree every 30 milliseconds
  const unsigned long STEP_INTERVAL = 30;

  // Keep all motors disabled initially.
  // Base must remain disabled until assembled.
  bool enabled[kJointCount] = {
    true, true, true, true
  };

  bool attached[kJointCount] = {
    false, false, false, false
  };


  void write(int joint, int degrees) {
      if (joint < 0 || joint >= kJointCount)
          return;

      if (!enabled[joint])
          return;

      if (degrees < kMinDegrees[joint] ||
          degrees > kMaxDegrees[joint])
          return;

      // Store target instead of moving immediately.
      desiredAngles[joint] = degrees;
  }




void update() {
    if (!motionAllowed)
        return;

    for (int i = 0; i < kJointCount; i++) {
        if (!enabled[i])
            continue;

        if (currentAngles[i] == desiredAngles[i])
            continue;

        if (!attached[i]) {
            motors[i].setPeriodHertz(50);
            motors[i].attach(pins[i], 1000, 2000);
            attached[i] = true;
        }

        // Send the destination directly.
        motors[i].write(desiredAngles[i]);

        currentAngles[i] = desiredAngles[i];
    }
}





  void detachAll() {
    motionAllowed = false;

    for (int i = 0; i < kJointCount; ++i) {
      desiredAngles[i] = currentAngles[i];

      if (attached[i]) {
        motors[i].detach();
        attached[i] = false;
      }
    }
  }

};


template <class Backend>
class ServoBridge {
 public:
  Backend backend;

  // Kept for compatibility with combined_only.ino STATE replies.
  // These are LAST COMMANDED angles, NOT servo encoder measurements.
  int positions[kJointCount] = {90, 90, 90, 90};
  int targetAngles[kJointCount] = {90, 90, 90, 90};

  // True only after someone establishes a trustworthy physical 90-degree
  // reference for that joint (e.g., a verified calibration procedure).
  // Startup's write(90) by itself does NOT establish such a reference.
  bool referenceEstablished[kJointCount] = {false, false, false, false};

  // The receiver's current STATE format already carries positions[4].
  // A Python client can calculate deviation = reported_angle - 90.
  int deviationFrom90(int joint) const {
    if (joint < 0 || joint >= kJointCount) return 0;
    return positions[joint] - kReferenceDegrees;
  }

  // Call only after a human-verified homing/calibration step, never just
  // because a command was sent. The base should remain unreferenced until
  // its assembly is finished.
  void markReferenceEstablished(int joint) {
    if (joint >= 0 && joint < kJointCount)
      referenceEstablished[joint] = true;
  }


  void syncPositions() {
      for (int i = 0; i < kJointCount; i++) {
          if (backend.enabled[i] && backend.attached[i]) {
              positions[i] = backend.currentAngles[i];
          }
      }
  }


  void invalidateReference(int joint) {
    if (joint >= 0 && joint < kJointCount)
      referenceEstablished[joint] = false;
  }

  // Accepts one validated four-joint target frame atomically.
  // Records positions only after issuing each backend command.
  // In the mock build, commands do NOT drive physical servos.

bool apply(const int16_t targets[kJointCount]) {
    int degrees[kJointCount];

    if (!validate_targets(targets, degrees))
        return false;

    for (int i = 0; i < kJointCount; ++i) {
        // Store the requested target separately.
        targetAngles[i] = degrees[i];

        if (backend.enabled[i]) {
            backend.write(i, degrees[i]);
        }
    }

    return true;
}


  // Do not overwrite the last commanded values on STOP.
  // The physical pose may change after PWM output is removed.
  void stop() {
    backend.detachAll();
    for (int i = 0; i < kJointCount; ++i)
      referenceEstablished[i] = false;
  }

  // Retain the pre-existing behavior: timeout preserves attachment and
  // last commanded values. The receiver handles disarming and session expiry.
  void timeout() {
    backend.detachAll();

    for (int i = 0; i < kJointCount; i++) {
        referenceEstablished[i] = false;
    }
}
};

#endif // SERVO_BRIDGE_H
