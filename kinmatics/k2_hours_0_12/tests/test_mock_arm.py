import json

import numpy as np
import pytest

from config import load_config
from Kinematics.mock_arm import MockArm


def test_mock_moves_under_speed_caps():
    cfg = load_config()
    cfg["safety"]["max_speed"] = 0.2
    cfg["joints"][1]["max_velocity"] = 0.1
    arm = MockArm(cfg, initial_q=[0, 0, 0, 0, 0.5])
    state = arm.move([0.5, 0.5, -0.5, 0.5, 0.8])
    assert state["q"] == [0, 0, 0, 0, 0.5]
    assert state["moving"] is True
    np.testing.assert_allclose(arm.advance(1)["q"], [0.2, 0.1, -0.2, 0.2, 0.7])
    state = arm.advance(10)
    np.testing.assert_allclose(state["q"], [0.5, 0.5, -0.5, 0.5, 0.8])
    assert state["moving"] is False
    assert arm.advance(0) == state


@pytest.mark.parametrize("bad_q", [
    [2, 0, 0, 0, 0.5], [0] * 4, [0] * 6, [float("nan")] * 5,
    [float("inf")] * 5, [True] * 5, ["0"] * 5, None, [[0]] * 5,
])
def test_invalid_move_does_not_change_state(bad_q):
    arm = MockArm()
    before = arm.get_state()
    with pytest.raises(ValueError, match="q"):
        arm.move(bad_q)
    assert arm.get_state() == before


@pytest.mark.parametrize("bad_dt", [-1, float("nan"), float("inf"), True, "1", None])
def test_invalid_time_rejected(bad_dt):
    with pytest.raises(ValueError, match="dt"):
        MockArm().advance(bad_dt)


def test_returned_state_is_a_copy():
    arm = MockArm()
    state = arm.get_state()
    before = arm.get_state()
    state["q"][0] = 999
    state["target_q"][0] = 999
    assert arm.get_state() == before


def test_packet_adapter_calls_supplied_codec():
    arm = MockArm()
    packet = b"test fixture only - NOT the team's wire format"

    def decode_targets(received):
        assert received == packet
        return [0.1, 0, 0, 0, 0.5]

    response = arm.exchange(packet, decode_targets=decode_targets,
                            encode_state=lambda state: json.dumps(state).encode())
    state = json.loads(response)
    assert state["target_q"] == [0.1, 0, 0, 0, 0.5]
    assert state["moving"]


def test_packet_rejected_by_decoder_cannot_move_arm():
    arm = MockArm()
    before = arm.get_state()

    def reject(_packet):
        raise ValueError("bad authentication")

    def never_encode(_state):
        pytest.fail("must not encode after failed decode")

    with pytest.raises(ValueError, match="authentication"):
        arm.exchange(b"bad", decode_targets=reject, encode_state=never_encode)
    assert arm.get_state() == before
