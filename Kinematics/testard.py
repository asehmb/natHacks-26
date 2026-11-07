"""Tests for the patched kinematic_model.py (original class, mm and degrees). Run: pytest -q"""
import contextlib
import io
import math
import os
import random

os.environ.setdefault("MPLBACKEND", "Agg")  # no GUI needed to import the module

import pytest

from kinematic_model import EEZYbotARM_Mk2

RNG = random.Random(7)
arm = EEZYbotARM_Mk2(0, 80, -90)


def random_valid_q():
    q1 = RNG.uniform(arm.q1_min, arm.q1_max)
    q2 = RNG.uniform(arm.q2_min, arm.q2_max)
    lo, hi = arm.q3CalcLimits(q2=q2)
    return q1, q2, RNG.uniform(lo, hi)


def fk(q):
    return arm.forwardKinematics(q1=q[0], q2=q[1], q3=q[2])


def test_fk_known_pose():
    # Straight up (q2 = 90, q3 = 0): tip is L4 out, at the top of both links
    x, y, z = fk((0, 90, 0))
    assert (x, y, z) == pytest.approx((arm.L4, 0, arm.L1 + arm.L2 + arm.L3), abs=1e-9)


def test_roundtrip_10000():
    """q -> fk -> ik -> same q, and fk(ik(p)) == p to well under a micron."""
    for _ in range(10_000):
        q = random_valid_q()
        p = fk(q)
        sol = arm.inverseKinematics(*p)
        assert sol is not None, f"IK rejected a reachable pose {q}"
        assert sol == pytest.approx(q, abs=1e-6)
        assert fk(sol) == pytest.approx(p, abs=1e-6)


def test_ik_is_silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        arm.inverseKinematics(250, 0, 150)
    assert buf.getvalue() == ""


@pytest.mark.parametrize(
    "p",
    [
        (500, 0, 100),     # too far: used to crash with math domain error
        (100, 0, -300),    # too low: used to crash
        (0, 0, 0),         # used to return q2 = 21.7 deg, outside the limits
        (10, 0, 92),       # closer to the axis than L4: used to return wrong angles
        (60, 0, 150),      # used to return q2 = 152 deg, outside the limits
        (-200, 0, 100),    # behind the base
        (200, 200, 100),   # outside the +/-30 deg base sweep
        (0, 0, 350),       # straight above the base
    ],
)
def test_unreachable_returns_none(p):
    assert arm.inverseKinematics(*p) is None


def test_ik_never_raises_and_never_violates_limits():
    accepted = 0
    for _ in range(20_000):
        p = (RNG.uniform(-100, 450), RNG.uniform(-300, 300), RNG.uniform(-50, 400))
        sol = arm.inverseKinematics(*p)
        if sol is None:
            continue
        accepted += 1
        arm.checkErrorJointLimits(q1=sol[0], q2=sol[1], q3=sol[2])  # must not raise
        assert fk(sol) == pytest.approx(p, abs=1e-6)
    assert accepted > 500


def test_ik_rejects_nan_and_inf():
    assert arm.inverseKinematics(float("nan"), 0, 100) is None
    assert arm.inverseKinematics(float("inf"), 0, 100) is None


def test_units_mistake_fails_loudly():
    """Meters passed to a mm function must give None, not plausible angles."""
    assert arm.inverseKinematics(0.255, 0.0, 0.199) is None


def test_ik_always_elbow_up():
    for _ in range(1000):
        sol = arm.inverseKinematics(*fk(random_valid_q()))
        assert sol[2] < 0


def test_limit_check_has_float_tolerance():
    """A pose exactly on a limit must still round-trip."""
    q = (arm.q1_max, arm.q2_max, arm.q3CalcLimits(q2=arm.q2_max)[0])
    assert arm.inverseKinematics(*fk(q)) is not None


# ======================================================================
# workspaceBounds (in memory, mm / degrees)
# ======================================================================


def test_workspace_bounds_keys_units_and_shrink():
    raw = arm.workspaceBounds(margin_mm=0, margin_deg=0)
    safe = arm.workspaceBounds()
    assert set(raw) == {"r_min", "r_max", "x_min", "x_max", "y_min", "y_max",
                        "z_min", "z_max", "q1_min", "q1_max"}
    for k in ("r", "x", "y", "z"):
        assert safe[f"{k}_min"] == pytest.approx(raw[f"{k}_min"] + 10.0)
        assert safe[f"{k}_max"] == pytest.approx(raw[f"{k}_max"] - 10.0)
    assert safe["q1_min"] == pytest.approx(arm.q1_min + 2.0)
    assert safe["q1_max"] == pytest.approx(arm.q1_max - 2.0)
    assert 200 < raw["r_max"] < 500          # millimeters
    assert raw["q1_min"] == arm.q1_min and raw["q1_max"] == arm.q1_max


def test_workspace_bounds_writes_no_files():
    """The old plot_EEZYbotARM_workspace wrote to /data; this must not touch the disk."""
    before = set(os.listdir("."))
    arm.workspaceBounds()
    assert set(os.listdir(".")) == before


def test_every_ik_solution_is_inside_raw_bounds():
    b = arm.workspaceBounds(margin_mm=0, margin_deg=0)
    eps = 2.0  # mm; the box comes from a sampled grid
    for _ in range(5000):
        x, y, z = fk(random_valid_q())
        assert b["x_min"] - eps <= x <= b["x_max"] + eps
        assert b["y_min"] - eps <= y <= b["y_max"] + eps
        assert b["z_min"] - eps <= z <= b["z_max"] + eps
        assert b["r_min"] - eps <= math.hypot(x, y) <= b["r_max"] + eps


def test_bounds_box_is_only_a_quick_reject():
    """Points can be inside the box and still be unreachable: ik() is the real check."""
    b = arm.workspaceBounds(margin_mm=0, margin_deg=0)
    corner = (b["x_min"] + 1.0, b["y_max"] - 1.0, b["z_max"] - 1.0)
    assert b["x_min"] <= corner[0] <= b["x_max"] and b["z_min"] <= corner[2] <= b["z_max"]
    assert arm.inverseKinematics(*corner) is None


# ======================================================================
# plan_move (unit-agnostic; degrees and deg/s here)
# ======================================================================

from kinematic_model import plan_move


def _as_list(value, n):
    return list(value) if isinstance(value, (list, tuple)) else [value] * n


def assert_within_limits(traj, dt, vmax, amax):
    """Finite-difference check of speed and acceleration on the sampled trajectory."""
    n = len(traj[0])
    vmax, amax = _as_list(vmax, n), _as_list(amax, n)
    tol = 1 + 1e-9
    vel = [[(b[i] - a[i]) / dt for i in range(n)] for a, b in zip(traj, traj[1:])]
    for v in vel:
        for i in range(n):
            assert abs(v[i]) <= vmax[i] * tol, f"joint {i} speed {abs(v[i])} > {vmax[i]}"
    acc = [[(v1[i] - v0[i]) / dt for i in range(n)] for v0, v1 in zip(vel, vel[1:])]
    acc.append([2 * vel[0][i] / dt for i in range(n)])    # start from rest (half-step)
    acc.append([2 * vel[-1][i] / dt for i in range(n)])   # stop at rest (half-step)
    for a in acc:
        for i in range(n):
            assert abs(a[i]) <= amax[i] * tol, f"joint {i} accel {abs(a[i])} > {amax[i]}"


def test_plan_move_limits_endpoints_random():
    for _ in range(300):
        q0, q1 = random_valid_q(), random_valid_q()
        vmax, amax = RNG.uniform(20, 120), RNG.uniform(50, 500)
        dt = RNG.choice([0.005, 0.01, 0.02, 0.05])
        traj = plan_move(q0, q1, vmax, amax, dt)
        assert traj[0] == tuple(q0)
        assert traj[-1] == tuple(q1)             # exactly the goal, no drift
        assert_within_limits(traj, dt, vmax, amax)


def test_plan_move_per_joint_limits():
    q0, q1 = (0, 50, -60), (25, 110, -140)
    vmax, amax = [20, 60, 90], [100, 150, 400]
    traj = plan_move(q0, q1, vmax, amax, 0.01)
    assert traj[-1] == q1
    assert_within_limits(traj, 0.01, vmax, amax)


def test_all_joints_finish_together():
    q0, q1 = (0, 50, -60), (25, 110, -140)
    traj = plan_move(q0, q1, 40, 200, 0.01)
    for k, q in enumerate(traj):
        fractions = [(q[i] - q0[i]) / (q1[i] - q0[i]) for i in range(3)]
        assert fractions == pytest.approx([fractions[0]] * 3, abs=1e-9)  # same progress every step
    # they are also monotonic (no overshoot / reversal)
    for i in range(3):
        sign = 1 if q1[i] > q0[i] else -1
        assert all(sign * (b[i] - a[i]) >= -1e-12 for a, b in zip(traj, traj[1:]))


def test_slowest_joint_sets_the_duration():
    dt, v, a = 0.01, 50.0, 200.0
    big = plan_move((0, 40, -90), (0, 140, -90), v, a, dt)        # 100 deg on one joint
    both = plan_move((0, 40, -90), (10, 140, -100), v, a, dt)     # plus two small moves
    T_expected = 100 / v + v / a                                  # trapezoid time for 100 deg
    assert len(big) == len(both)                                  # small joints don't add time
    assert T_expected <= (len(big) - 1) * dt < T_expected + dt


def test_short_move_is_triangular_long_move_is_trapezoidal():
    dt, v, a = 0.005, 100.0, 100.0
    short = plan_move((0, 60, -90), (0, 62, -90), v, a, dt)       # 2 deg: never reaches 100 deg/s
    peak_short = max(abs(b[1] - c[1]) / dt for b, c in zip(short, short[1:]))
    assert peak_short < 0.5 * v

    long_ = plan_move((0, 40, -90), (0, 130, -90), 60.0, 300.0, dt)  # 90 deg: cruises at 60 deg/s
    speeds = [abs(b[1] - c[1]) / dt for b, c in zip(long_, long_[1:])]
    assert max(speeds) >= 0.99 * 60.0
    assert sum(s >= 0.99 * 60.0 for s in speeds) > 10               # a real cruise phase


def test_zero_length_move_does_not_divide_by_zero():
    q = (10.0, 60.0, -90.0)
    assert plan_move(q, q, 50, 100, 0.01) == [q]
    # one joint still, the others moving: the still joint must not move
    traj = plan_move((0, 50, -80), (0, 90, -100), 50, 100, 0.01)
    assert all(p[0] == 0 for p in traj)


def test_tiny_move_is_fine():
    traj = plan_move((0, 60, -90), (0, 60.0000001, -90), 50, 100, 0.01)
    assert traj[-1] == (0, 60.0000001, -90)
    assert_within_limits(traj, 0.01, 50, 100)


def test_plan_move_accepts_lists_and_tuples():
    assert plan_move([0, 60, -90], [10, 70, -100], 50, 100, 0.02)[-1] == (10, 70, -100)


@pytest.mark.parametrize(
    "args",
    [
        ((0, 0, 0), (1, 1, 1), 50, 100, 0.0),            # dt = 0
        ((0, 0, 0), (1, 1, 1), 50, 100, -0.01),
        ((0, 0, 0), (1, 1, 1), 0, 100, 0.01),            # vmax = 0
        ((0, 0, 0), (1, 1, 1), 50, -5, 0.01),            # negative amax
        ((0, 0, 0), (1, 1), 50, 100, 0.01),              # length mismatch
        ((0, 0, 0), (1, 1, 1), [50, 50], 100, 0.01),     # wrong number of vmax entries
        ((0, 0, 0), (float("nan"), 1, 1), 50, 100, 0.01),
        ((), (), 50, 100, 0.01),
    ],
)
def test_plan_move_rejects_bad_arguments(args):
    with pytest.raises(ValueError):
        plan_move(*args)


def test_planned_moves_stay_inside_joint_limits():
    """Both ends valid => every sample valid (the allowed joint region is convex)."""
    for _ in range(200):
        traj = plan_move(random_valid_q(), random_valid_q(), 60, 300, 0.02)
        for q in traj:
            arm.checkErrorJointLimits(q1=q[0], q2=q[1], q3=q[2])  # must not raise

