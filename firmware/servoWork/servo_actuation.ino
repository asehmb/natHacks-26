
#include <ESP32Servo.h>

const int N = 4;

const char* jointNames[N] = {
  "base", "shoulder", "elbow", "gripper"
};

const int pins[N] = {
  18, 19, 21, 22
};

Servo joints[N];

int positions[N] = {
  90, 90, 90, 90
};

// For them mf angles
const int minAngles[N] = {
  30, 60, 45, 70
};

const int maxAngles[N] = {
  150, 120, 135, 110
};

// Enable additional joints only after
// wiring and testing them individually.
bool enabled[N] = {
  true, true, true, true
};

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);

  for (int i = 0; i < N; i++) {
    if (enabled[i]) {
      joints[i].setPeriodHertz(50);
      joints[i].attach(pins[i], 1000, 2000);
      joints[i].write(positions[i]);
    }
  }

  Serial.println("EasyBot MK2 Controller");
  Serial.println("Commands:");
  Serial.println("base 90");
  Serial.println("shoulder 90");
  Serial.println("elbow 90");
  Serial.println("gripper 90");
  Serial.println("status");
}

void loop() {
  if (!Serial.available()) return;

  String input = Serial.readStringUntil('\n');
  input.trim();

  if (input == "limits") {
  for (int i = 0; i < N; i++) {
    Serial.printf(
      "%s: %d to %d degrees\n",
      jointNames[i],
      minAngles[i],
      maxAngles[i]
    );
  }
  return;
}

  if (input == "status") {
    for (int i = 0; i < N; i++) {
      Serial.printf(
        "%s: %d degrees (%s)\n",
        jointNames[i],
        positions[i],
        enabled[i] ? "ON" : "OFF"
      );
    }
    return;
  }

  int space = input.indexOf(' ');

  if (space == -1) {
    Serial.println("Invalid command");
    return;
  }

  String name = input.substring(0, space);
  String value = input.substring(space + 1);
  value.trim();

  if (value.length() == 0) {
    Serial.println("Missing angle");
    return;
  }

  for (unsigned int k = 0; k < value.length(); k++) {
    if (!isDigit(value[k])) {
      Serial.println("Invalid angle");
      return;
    }
  }

  int angle = value.toInt();

  int id = -1;

  for (int i = 0; i < N; i++) {
    if (name == jointNames[i]) {
      id = i;
      break;
    }
  }

  if (id == -1) {
    Serial.println("Unknown joint");
    return;
  }

  // if (!enabled[id]) {
  //   Serial.println("Joint not enabled yet");
  //   return;
  // }

  positions[id] = angle;


if (angle < minAngles[id] || angle > maxAngles[id]) {
  Serial.printf(
    "Rejected: %s must be between %d and %d degrees\n",
    jointNames[id],
    minAngles[id],
    maxAngles[id]
  );
  return;
}

  joints[id].write(angle);
  positions[id] = angle;

  Serial.printf(
    "%s moving to %d degrees\n",
    jointNames[id],
    angle
  );

    // Serial.printf(
    // "SIMULATED: %s accepted %d degrees\n",
    // jointNames[id],
    // angle
    // );
}
