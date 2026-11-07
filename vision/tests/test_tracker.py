import numpy as np

from armvision.tracker import Observation, Tracker


def ob(x, y=0.0):
    p = np.array([x, y], float)
    return Observation(p, p.copy(), 0.0, (50.0, 20.0), "red")


def test_ids_stable_while_moving():
    t = Tracker(ema=1.0)
    ids = None
    for k in range(10):
        tr = t.update([ob(100 + 5 * k), ob(300 - 5 * k, 100)])
        cur = sorted((x.id, round(x.obs.center[0])) for x in tr)
        ids = ids or [i for i, _ in cur]
        assert [i for i, _ in cur] == ids
    assert len(tr) == 2


def test_missing_dropped_after_n_frames():
    t = Tracker(max_missed=3)
    t.update([ob(100)])
    for _ in range(3):
        assert len(t.update([])) == 1
    assert t.update([]) == []


def test_occluded_kept():
    t = Tracker(max_missed=2)
    t.update([ob(100)])
    for _ in range(20):
        tr = t.update([], in_arm=lambda p: True)
    assert len(tr) == 1 and tr[0].occluded
    tr = t.update([ob(102)], in_arm=lambda p: True)
    assert tr[0].id == 1 and not tr[0].occluded


def test_far_detection_is_new_track():
    t = Tracker(gate_mm=40)
    t.update([ob(100)])
    tr = t.update([ob(300)])
    assert sorted(x.id for x in tr) == [1, 2]
