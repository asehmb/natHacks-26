import numpy as np
import pytest

from nlp import ObjectRef
from vision.detector import COCO_LABELS, decode_yolov8, nms
from vision.grounding import Detection, Grounder, label_matches
from vision.hsv import color_fraction, color_mask, rgb_to_hsv
from vision.stream import TemporalSmoother


class Fixed:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame):
        return self.dets


def blank(h=100, w=200):
    return np.full((h, w, 3), 240, np.uint8)


def test_rgb_to_hsv_known_values():
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], np.uint8)
    h, s, v = rgb_to_hsv(px)
    assert np.allclose(h[0, :3], [0, 120, 240]) and np.allclose(s[0], [1, 1, 1, 0, 0]) and np.allclose(v[0], [1, 1, 1, 1, 0])


def test_unknown_color_raises():
    with pytest.raises(ValueError):
        color_mask(blank(), "chartreuse")


def test_empty_crop_fraction_is_zero():
    assert color_fraction(np.zeros((0, 0, 3), np.uint8), "red") == 0.0


def test_half_red_half_blue():
    f = np.zeros((10, 10, 3), np.uint8)
    f[:, :5] = (220, 20, 20)
    f[:, 5:] = (20, 20, 220)
    assert color_fraction(f, "red") == 0.5 and color_fraction(f, "blue") == 0.5 and color_fraction(f, "green") == 0.0


@pytest.mark.parametrize("noun,label,ok", [
    ("cup", "cup", True), ("cup", "Cup", True), ("wine glass", "glass", True), ("cup", "dog", False),
])
def test_label_matches(noun, label, ok):
    assert label_matches(noun, label) is ok


def test_size_words_pick_big_or_small():
    dets = [Detection("cup", (0, 0, 10, 10), 0.9), Detection("cup", (20, 0, 80, 60), 0.5)]
    g = Grounder(Fixed(dets))
    assert g.find(blank(), ObjectRef("cup", size="big")).bbox == (20, 0, 80, 60)
    assert g.find(blank(), ObjectRef("cup", size="small")).bbox == (0, 0, 10, 10)
    assert g.find(blank(), ObjectRef("cup")).bbox == (0, 0, 10, 10)  # default: highest score


@pytest.mark.parametrize("pos,expected", [
    ("left", (0, 40, 20, 60)), ("right", (180, 40, 200, 60)), ("top", (90, 0, 110, 20)),
    ("bottom", (90, 80, 110, 100)), ("middle", (90, 40, 110, 60)),
])
def test_position_words(pos, expected):
    dets = [Detection("cup", b, 0.8) for b in
            [(0, 40, 20, 60), (180, 40, 200, 60), (90, 0, 110, 20), (90, 80, 110, 100), (90, 40, 110, 60)]]
    assert Grounder(Fixed(dets)).find(blank(), ObjectRef("cup", position=pos)).bbox == expected


def test_color_filter_threshold():
    f = blank()
    f[0:100, 0:10] = (220, 20, 20)  # exactly 10% of the 100x100 box is red
    d = [Detection("cup", (0, 0, 100, 100), 0.9)]
    assert Grounder(Fixed(d), min_color_fraction=0.15, blob_fallback=False).find(f, ObjectRef("cup", "red")) is None
    assert Grounder(Fixed(d), min_color_fraction=0.05).find(f, ObjectRef("cup", "red")) is not None


def test_no_detector_and_no_color_returns_none():
    assert Grounder(None).find(blank(), ObjectRef("cup")) is None


def test_blob_fallback_without_detector_picks_largest_blob():
    f = blank()
    f[5:15, 5:15] = (220, 20, 20)
    f[40:80, 100:150] = (220, 20, 20)
    d = Grounder(None).find(f, ObjectRef("screwdriver", "red"))
    assert d.bbox == (100, 40, 150, 80)


def test_blob_fallback_does_not_need_opencv(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)  # makes `import cv2` raise ImportError
    f = blank()
    f[5:15, 5:15] = (220, 20, 20)
    f[40:80, 100:150] = (220, 20, 20)
    assert Grounder(None).find(f, ObjectRef("screwdriver", "red")).bbox == (100, 40, 150, 80)


def test_blob_below_min_fraction_is_ignored():
    f = blank()
    f[0:2, 0:2] = (220, 20, 20)
    assert Grounder(None, min_blob_fraction=0.01).find(f, ObjectRef("pen", "red")) is None


def test_detections_expose_area_and_center():
    d = Detection("x", (10, 20, 30, 60), 1.0)
    assert d.area == 800 and d.center == (20.0, 40.0)
    assert Detection("x", (30, 30, 10, 10), 1).area == 0  # inverted box


def test_nms_keeps_separate_boxes_and_orders_by_score():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    assert nms(boxes, np.array([0.2, 0.9]), 0.5) == [1, 0]
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def test_decode_nothing_above_threshold_and_per_class_nms():
    nc = len(COCO_LABELS)
    out = np.zeros((1, 4 + nc, 3), np.float32)
    assert decode_yolov8(out, COCO_LABELS, 0.25, 0.45, 1.0, (0, 0), (100, 100)) == []
    for i, c in enumerate(["cup", "cup", "dog"]):  # two overlapping cups + a dog on the same spot
        out[0, :4, i] = (50, 50, 40, 40)
        out[0, 4 + COCO_LABELS.index(c), i] = 0.9 - i * 0.1
    dets = decode_yolov8(out, COCO_LABELS, 0.25, 0.45, 1.0, (0, 0), (100, 100))
    assert sorted(d.label for d in dets) == ["cup", "dog"]


def test_decode_clips_to_frame():
    nc = len(COCO_LABELS)
    out = np.zeros((1, 4 + nc, 1), np.float32)
    out[0, :4, 0] = (5, 5, 100, 100)
    out[0, 4, 0] = 0.9
    (d,) = decode_yolov8(out, COCO_LABELS, 0.25, 0.45, 1.0, (0, 0), (50, 60))
    assert d.bbox == (0, 0, 55, 50) or (d.bbox[0] == 0 and d.bbox[1] == 0 and d.bbox[2] <= 60 and d.bbox[3] <= 50)


def test_smoother_blends_and_recovers():
    s = TemporalSmoother(alpha=0.5, max_misses=1)
    s.update(Detection("c", (0, 0, 100, 100), 0.9))
    assert s.update(Detection("c", (100, 100, 200, 200), 0.9)).bbox == (50, 50, 150, 150)
    assert s.update(None) is not None
    assert s.update(None) is None
    assert s.update(Detection("c", (10, 10, 20, 20), 0.5)).bbox == (10, 10, 20, 20)  # restarts cleanly


def test_label_count():
    assert len(COCO_LABELS) == 80 and "traffic light" in COCO_LABELS and "cell phone" in COCO_LABELS


def test_providers_always_end_with_cpu():
    pytest.importorskip("onnxruntime")
    from vision.providers import select_providers

    assert select_providers()[-1] == "CPUExecutionProvider"
    assert select_providers(allow_accel=False) == ["CPUExecutionProvider"]


def test_frames_reads_a_video_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from vision.stream import frames, track

    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not w.isOpened():
        pytest.skip("no MJPG writer available")
    for _ in range(5):
        bgr = np.zeros((48, 64, 3), np.uint8)
        bgr[:, :] = (0, 0, 255)  # BGR red
        w.write(bgr)
    w.release()
    got = list(frames(path))
    assert len(got) == 5 and got[0].shape == (48, 64, 3)
    assert color_fraction(got[0], "red") > 0.9  # frames are converted to RGB
    results = list(track(Grounder(None), ObjectRef("thing", "red"), path))
    assert len(results) == 5 and results[-1][1] is not None


def test_frames_on_missing_source_yields_nothing():
    pytest.importorskip("cv2")
    from vision.stream import frames

    assert list(frames("/nonexistent/video.mp4")) == []
