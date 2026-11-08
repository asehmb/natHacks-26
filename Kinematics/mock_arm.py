"""Deterministic joint-space mock, with an explicit F2 packet adapter seam.

No sockets, PWM, signing implementation, FK, IK, or real-time background thread.
Call advance(dt) explicitly so teammate tests do not have to sleep.
"""

import json
import math

import numpy as np

from config import JOINT_NAMES, get_joint_limits, load_config


class MockArm:
    def __init__(self, config=None, initial_q=None):
        config = load_config() if config is None else config
        self._limits = get_joint_limits(config)
        self._velocity = np.minimum(
            [j["max_velocity"] for j in config["joints"]],
            config["safety"]["max_speed"],
        )
        # Unknown physical home: choose valid interval midpoints for simulation.
        self._q = self._limits.mean(axis=1) if initial_q is None else self._check_q(initial_q)
        self._target = self._q.copy()

    def _check_q(self, q):
        try:
            values = list(q)
        except TypeError as exc:
            raise ValueError("q: expected five joint angles in radians") from exc
        if len(values) != 5:
            raise ValueError("q: expected five joint angles in radians")
        if any(isinstance(x, (bool, str)) for x in values):
            raise ValueError("q: angles must be finite numbers, not strings/booleans")
        try:
            q = np.asarray(values, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("q: angles must be finite numbers") from exc
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            raise ValueError("q: expected five finite joint angles")
        for i, name in enumerate(JOINT_NAMES):
            if not self._limits[i, 0] <= q[i] <= self._limits[i, 1]:
                raise ValueError(f"q[{i}] ({name}): outside configured joint limits")
        return q.copy()

    def move(self, q):
        """Queue a model-space target (five radians); reject instead of clamp."""
        self._target = self._check_q(q)
        return self.get_state()

    def advance(self, dt):
        """Advance simulation by dt seconds, applying global/per-joint speed caps."""
        if isinstance(dt, (bool, str)):
            raise ValueError("dt: expected a finite nonnegative number of seconds")
        try:
            dt = float(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt: expected a finite nonnegative number of seconds") from exc
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt: expected a finite nonnegative number of seconds")
        step = self._velocity * dt
        self._q += np.clip(self._target - self._q, -step, step)
        return self.get_state()

    def get_state(self):
        """Return a fresh JSON-serializable snapshot, in model radians."""
        return {"q": self._q.tolist(), "target_q": self._target.tolist(),
                "moving": bool(np.any(np.abs(self._target - self._q) > 1e-9))}

    def exchange(self, packet, *, decode_targets, encode_state):
        """F2 adapter: validate/decode request -> move -> encode current state.

        decode_targets(packet) must perform F2's length/version/HMAC/counter
        validation AND wire servo-space -> model-space conversion, returning q.
        encode_state(state) must emit F2's actual response bytes. Exceptions from
        the decoder propagate BEFORE changing any target. No invented wire format.
        A response acknowledges the target; advance(dt) performs simulated motion.
        """
        q = decode_targets(packet)
        self.move(q)
        return encode_state(self.get_state())


def main():
    arm = MockArm()
    state = arm.get_state()
    print(json.dumps({"simulation_only": True, "initial": state}))
    target = state["q"][:]
    target[0] += 0.05
    arm.move(target)
    print(json.dumps(arm.advance(0.1)))
    print(json.dumps(arm.advance(1.0)))


if __name__ == "__main__":
    main()
