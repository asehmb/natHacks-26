import numpy as np
import pytest

from armvision.grasp import compute_grasp


def rect(center, length, width, angle_deg, n=20):
    a = np.radians(angle_deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array(center, float)
    corners = [c + sx * length / 2 * u + sy * width / 2 * v for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    pts = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        pts += [p + (q - p) * t for t in np.linspace(0, 1, n, endpoint=False)]
    return np.array(pts)


@pytest.mark.parametrize("angle", [0, 30, 72, -45, 90])
def test_near_end_inset_and_yaw(angle):
    c = np.array([200.0, 100.0])
    g = compute_grasp(rect(c, 60, 20, angle), base_xy=(0, 100), inset_mm=15)
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    ends = [c + d * 30, c - d * 30]
    near = min(ends, key=lambda e: np.linalg.norm(e - [0, 100]))
    expected = near + (c - near) / 30 * 15
    assert np.allclose(g.point, expected, atol=1.0)
    assert g.length == pytest.approx(60, abs=1.5) and g.width == pytest.approx(20, abs=1.5)
    diff = (g.yaw_deg - g.axis_deg) % 180
    assert diff == pytest.approx(90, abs=1.0)


def test_square_uses_centroid():
    c = np.array([150.0, 50.0])
    g = compute_grasp(rect(c, 30, 28, 20), base_xy=(0, 0))
    assert np.allclose(g.point, c, atol=0.5)


def test_inset_clamped_to_half_length():
    c = np.array([100.0, 0.0])
    g = compute_grasp(rect(c, 20, 6, 0), base_xy=(0, 0), inset_mm=50)
    assert np.allclose(g.point, c, atol=0.5)
