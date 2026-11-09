import math
import sys
from pathlib import Path
import numpy as np
import pytest

_KINEMATICS_DIR = Path(__file__).resolve().parent.parent.parent / "Kinematics"
if str(_KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(_KINEMATICS_DIR))

from armvision.pipeline import ArmState, TrackedObject, WorldState
from armvision.planner_bridge import (
    TargetValidationError,
    check_reachability,
    execute_pick_and_place,
    target_to_planner_coords,
    validate_target,
)
from mock_adapter import MockArmAdapter
from mock_arm import MockArm
from planner import Planner, State


def make_object(
    obj_id=1,
    color="red",
    center_mm=(240.0, 20.0, 0.0),
    grasp_mm=(240.0, 20.0, 120.0),
    grasp_yaw_deg=0.0,
    size_mm=(40.0, 20.0),
    occluded=False,
    age=5,
    seen=True,
    missed=0,
):
    return TrackedObject(
        id=obj_id,
        color=color,
        label="object",
        center_mm=np.array(center_mm, dtype=float),
        grasp_mm=np.array(grasp_mm, dtype=float),
        grasp_yaw_deg=grasp_yaw_deg,
        size_mm=size_mm,
        occluded=occluded,
        age=age,
        seen=seen,
        missed=missed,
    )


def test_target_validation_requires_fresh_mat():
    obj = make_object()
    # 1. mat not visible
    state_no_mat = WorldState(mat_visible=False, pose_age=0, objects=[obj])
    with pytest.raises(TargetValidationError, match="Mat localization is not fresh"):
        validate_target(state_no_mat)

    # 2. pose_age > 0 (stale pose)
    state_stale_pose = WorldState(mat_visible=True, pose_age=1, objects=[obj])
    with pytest.raises(TargetValidationError, match="Mat localization is not fresh"):
        validate_target(state_stale_pose)

    # 3. fresh pose succeeds
    state_fresh = WorldState(mat_visible=True, pose_age=0, objects=[obj])
    assert validate_target(state_fresh).id == 1


def test_target_validation_requires_seen_in_current_frame():
    obj_stale = make_object(seen=False, missed=1)
    state = WorldState(mat_visible=True, pose_age=0, objects=[obj_stale])
    with pytest.raises(TargetValidationError, match="was not seen in current frame"):
        validate_target(state)


def test_target_validation_requires_no_occlusion():
    obj_occluded = make_object(occluded=True)
    state = WorldState(mat_visible=True, pose_age=0, objects=[obj_occluded])
    with pytest.raises(TargetValidationError, match="is occluded"):
        validate_target(state)


def test_target_validation_rejects_nonfinite_coordinates():
    obj_nan = make_object(grasp_mm=(240.0, float("nan"), 120.0))
    state = WorldState(mat_visible=True, pose_age=0, objects=[obj_nan])
    with pytest.raises(TargetValidationError, match="non-finite"):
        validate_target(state)

    obj_inf = make_object(center_mm=(float("inf"), 20.0, 0.0))
    state_inf = WorldState(mat_visible=True, pose_age=0, objects=[obj_inf])
    with pytest.raises(TargetValidationError, match="non-finite"):
        validate_target(state_inf)


def test_target_selection_by_color_and_position():
    red = make_object(obj_id=1, color="red", center_mm=(220.0, -30.0, 0.0), grasp_mm=(220.0, -30.0, 120.0))
    blue = make_object(obj_id=2, color="blue", center_mm=(240.0, 40.0, 0.0), grasp_mm=(240.0, 40.0, 120.0))
    state = WorldState(mat_visible=True, pose_age=0, objects=[red, blue])

    # Select by color
    assert validate_target(state, color="blue").id == 2
    assert validate_target(state, color="red").id == 1

    # Select non-existent color
    with pytest.raises(TargetValidationError, match="No object found matching color='green'"):
        validate_target(state, color="green")


def test_stale_object_does_not_block_valid_object():
    # Stale red object nearer the base (x=200)
    stale_red = make_object(
        obj_id=1, color="red", center_mm=(200.0, 0.0, 0.0), grasp_mm=(200.0, 0.0, 120.0),
        seen=False, missed=1,
    )
    # Fresh red object farther from base (x=250)
    fresh_red = make_object(
        obj_id=2, color="red", center_mm=(250.0, 0.0, 0.0), grasp_mm=(250.0, 0.0, 120.0),
        seen=True, missed=0,
    )
    state = WorldState(mat_visible=True, pose_age=0, objects=[stale_red, fresh_red])

    # 1. Selection by color selects the valid fresh red object (#2), not blocked by stale (#1)
    target = validate_target(state, color="red")
    assert target.id == 2

    # 2. General selection without color also selects the valid fresh red object
    assert validate_target(state).id == 2

    # 3. Explicitly requesting the stale object ID must still be rejected
    with pytest.raises(TargetValidationError, match="was not seen in current frame"):
        validate_target(state, obj_id=1)


def test_occluded_object_does_not_block_valid_object():
    # Occluded blue object nearer the base
    occluded_blue = make_object(
        obj_id=10, color="blue", center_mm=(200.0, 0.0, 0.0), grasp_mm=(200.0, 0.0, 120.0),
        occluded=True,
    )
    # Clear blue object farther from base
    clear_blue = make_object(
        obj_id=20, color="blue", center_mm=(250.0, 0.0, 0.0), grasp_mm=(250.0, 0.0, 120.0),
        occluded=False,
    )
    state = WorldState(mat_visible=True, pose_age=0, objects=[occluded_blue, clear_blue])

    # Selection by color selects the clear blue object (#20)
    assert validate_target(state, color="blue").id == 20

    # Explicitly requesting the occluded object ID is rejected
    with pytest.raises(TargetValidationError, match="is occluded by the arm"):
        validate_target(state, obj_id=10)


def test_conflicting_id_and_color_rejected():
    red = make_object(obj_id=1, color="red")
    blue = make_object(obj_id=2, color="blue")
    state = WorldState(mat_visible=True, pose_age=0, objects=[red, blue])

    # 1. Requesting a blue object by a red object's ID must be strictly rejected
    with pytest.raises(TargetValidationError, match="Conflicting criteria: target #1 has color 'red', which does not match requested color 'blue'"):
        validate_target(state, obj_id=1, color="blue")

    # 2. Requesting matching ID and color succeeds
    assert validate_target(state, obj_id=1, color="red").id == 1
    assert validate_target(state, obj_id=2, color="blue").id == 2

    # 3. Case-insensitive and alias normalization via COLOR_DICT
    assert validate_target(state, obj_id=1, color="RED").id == 1
    gray_obj = make_object(obj_id=3, color="gray")
    state_gray = WorldState(mat_visible=True, pose_age=0, objects=[gray_obj])
    assert validate_target(state_gray, obj_id=3, color="grey").id == 3

    # 4. Requesting ID without color accepts the object regardless of color
    assert validate_target(state, obj_id=1).id == 1


def test_coordinate_conversion_to_meters():
    obj = make_object(grasp_mm=(240.0, 20.0, 120.0))
    coords_m = target_to_planner_coords(obj)
    assert coords_m == pytest.approx((0.24, 0.02, 0.12), abs=1e-6)

    # With explicit grasp_z_m override
    coords_override = target_to_planner_coords(obj, grasp_z_m=0.10)
    assert coords_override == pytest.approx((0.24, 0.02, 0.10), abs=1e-6)


def test_reachability_checks():
    # Reachable coordinate
    sol = check_reachability((0.24, 0.02, 0.12))
    assert sol is not None and len(sol) == 3

    # Out of reach coordinate
    with pytest.raises(TargetValidationError, match="is unreachable"):
        check_reachability((0.55, 0.0, 0.12))


def test_execute_pick_and_place_with_mock_planner():
    # Target in reachable workspace
    obj = make_object(grasp_mm=(240.0, 20.0, 120.0))
    state = WorldState(mat_visible=True, pose_age=0, objects=[obj])

    arm = MockArmAdapter(MockArm())
    planner = Planner(arm)
    planner.go_home()

    dest_m = (0.24, -0.05, 0.12)
    target, grasp_m = execute_pick_and_place(planner, state, destination_m=dest_m)

    assert target.id == 1
    assert grasp_m == pytest.approx((0.24, 0.02, 0.12), abs=1e-6)
    assert planner.state == State.DONE


def test_execute_pick_and_place_rejects_unreachable_target():
    # Target too far away (x=500mm -> 0.5m)
    obj_unreachable = make_object(grasp_mm=(500.0, 0.0, 120.0))
    state = WorldState(mat_visible=True, pose_age=0, objects=[obj_unreachable])

    arm = MockArmAdapter(MockArm())
    planner = Planner(arm)
    planner.go_home()

    with pytest.raises(TargetValidationError, match="is unreachable"):
        execute_pick_and_place(planner, state)

    # Arm state remains HOMING / DONE with go_home, pick_and_place was never commanded
    assert planner.state == State.DONE
