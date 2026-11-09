import numpy as np
import pytest

from armvision.pipeline import VisionSystem
from armvision.yolo import Box, decode, find_models

from conftest import draw_rect, mat_canvas, to_camera


class FakeLabeler:
    """Stands in for YoloWorld: returns boxes for fixed mat-plane rectangles, in camera pixels."""

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def detect(self, frame, queries):
        self.calls += 1
        return self.boxes


def img_box(label, xyxy, score=0.9):
    return Box(label, xyxy, score)


def scene(cfg, with_object=True):
    canvas = mat_canvas(cfg)
    if with_object:
        draw_rect(canvas, (200, 150), (50, 30), 0, (0, 0, 220))
    return to_camera(canvas)


def box_around_mat_point(vs, center, half=40):
    cx, cy = vs.pose.mat_to_img(center)[0]
    return img_box("red cup", (cx - half, cy - half, cx + half, cy + half))


def test_decode_scales_back_to_frame():
    out = np.zeros((1, 4 + 2, 1), np.float32)
    out[0, :, 0] = [320, 320, 100, 200, 0.1, 0.8]          # cx, cy, w, h, class scores
    boxes = decode(out, ["a", "b"], 0.25, 0.45, scale=0.5, pad=(0, 0), frame_hw=(1280, 1280))
    assert len(boxes) == 1 and boxes[0].label == "b"
    assert boxes[0].xyxy == pytest.approx((540, 440, 740, 840))


def test_find_models_missing(tmp_path):
    assert find_models(tmp_path) is None


def test_diff_mode_labels_objects(cfg):
    vs = VisionSystem(cfg, track_arm=False)
    assert vs.capture_reference(scene(cfg, False))
    vs.labeler = FakeLabeler([])
    vs.process(scene(cfg))                                  # pose is known now
    vs.labeler = FakeLabeler([box_around_mat_point(vs, (200, 150))])
    vs._n = 0
    state = vs.process(scene(cfg))
    assert [o.label for o in state.objects] == ["red cup"]
    assert [o.label for o in state.find(label="cup")] == ["red cup"]
    # labels stay when YOLO is skipped on later frames (yolo_every) or returns nothing
    vs.labeler = FakeLabeler([])
    vs._n = 1
    assert vs.process(scene(cfg)).objects[0].label == "red cup"


def test_yolo_mode_places_box_on_mat(cfg):
    vs = VisionSystem(cfg, track_arm=False)
    vs.update_pose(scene(cfg, False))
    # box whose bottom edge sits on mat point (200, 150) -> base frame (260, 0)
    cx, cy = vs.pose.mat_to_img((200, 150))[0]
    lab = FakeLabeler([img_box("bottle", (cx - 20, cy - 80, cx + 20, cy))])
    ys = VisionSystem(cfg, labeler=lab, mode="yolo", track_arm=False)
    state = ys.process(scene(cfg))
    assert len(state.objects) == 1 and state.objects[0].label == "bottle"
    assert np.allclose(state.objects[0].center_mm[:2], [260, 0], atol=3)


def test_yolo_mode_needs_labeler(cfg):
    with pytest.raises(ValueError):
        VisionSystem(cfg, mode="yolo")


def test_label_voting_resists_one_bad_frame(cfg):
    vs = VisionSystem(cfg, track_arm=False)
    assert vs.capture_reference(scene(cfg, False))
    vs.labeler = FakeLabeler([])
    vs.process(scene(cfg))
    good = box_around_mat_point(vs, (200, 150))
    bad = img_box("scissors", good.xyxy, 0.5)
    for lab in (good, good, good, bad):                      # shadow makes one frame read "scissors"
        vs.labeler = FakeLabeler([lab if lab is not good else img_box("screwdriver", good.xyxy, 0.8)])
        vs._n = 0
        state = vs.process(scene(cfg))
    o = state.objects[0]
    assert o.label == "screwdriver"
    assert o.confidence < 0.8 * 0.6 + 0.4          # dissent costs confidence


def test_default_vocab_has_screwdriver():
    from armvision.yolo import DEFAULT_VOCAB
    assert "screwdriver" in DEFAULT_VOCAB


class FakeClipYolo:
    """YoloWorld with the ONNX sessions stubbed: CLIP gives fixed probabilities for each crop."""

    def __init__(self, probs):
        from armvision.yolo import YoloWorld
        self.y = YoloWorld.__new__(YoloWorld)
        self.y.clip, self.y.clip_min, self.y.max_rerank = object(), 0.35, 8
        self.y._clip_probs = lambda crops, queries: np.tile(probs, (len(crops), 1))


def test_clip_overrides_confident_wrong_label():
    y = FakeClipYolo([0.1, 0.8, 0.1]).y
    frame = np.zeros((200, 200, 3), np.uint8)
    out = y.rerank_boxes(frame, [Box("scissors", (20, 20, 120, 120), 0.6)], ["scissors", "screwdriver", "pen"])
    assert out[0].label == "screwdriver"
    assert out[0].score == pytest.approx((0.6 * 0.8) ** 0.5)


def test_clip_keeps_yolo_label_when_unsure():
    y = FakeClipYolo([0.34, 0.33, 0.33]).y                      # no class reaches clip_min
    frame = np.zeros((200, 200, 3), np.uint8)
    out = y.rerank_boxes(frame, [Box("scissors", (20, 20, 120, 120), 0.6)], ["scissors", "screwdriver", "pen"])
    assert out[0].label == "scissors" and out[0].score < 0.6


def test_clip_preprocess_shape_and_padding():
    from armvision.yolo import clip_preprocess
    x = clip_preprocess(np.full((40, 100, 3), 255, np.uint8))
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_diff_mode_clip_fallback_labels_box_less_object(cfg):
    vs = VisionSystem(cfg, track_arm=False)
    assert vs.capture_reference(scene(cfg, False))

    class ClipOnly(FakeLabeler):
        clip_min = 0.35

        def classify_crop(self, frame, xyxy, queries):
            return "screwdriver", 0.9
    vs.labeler = ClipOnly([])
    vs._n = 0
    assert vs.process(scene(cfg)).objects[0].label == "screwdriver"
