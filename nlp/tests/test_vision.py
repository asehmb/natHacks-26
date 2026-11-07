import numpy as np
import pytest

pytest.importorskip("numpy")

from nlp import ObjectRef
from vision.detector import COCO_LABELS, decode_yolov8, nms
from vision.grounding import Detection, Grounder
from vision.hsv import color_fraction
from vision.stream import TemporalSmoother


def solid(rgb, h=20, w=20):
    return np.full((h, w, 3), rgb, np.uint8)


@pytest.mark.parametrize("rgb,color", [
    ((220, 20, 20), "red"), ((220, 20, 60), "red"),  # hue just below 360 wraps to red
    ((20, 20, 220), "blue"), ((20, 200, 20), "green"),
    ((250, 250, 250), "white"), ((10, 10, 10), "black"), ((128, 128, 128), "gray"),
])
def test_color_match(rgb, color):
    assert color_fraction(solid(rgb), color) == 1.0


def test_color_mismatch():
    assert color_fraction(solid((20, 20, 220)), "red") == 0.0


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame):
        return self.dets


def scene():
    f = np.full((100, 200, 3), 240, np.uint8)
    f[10:40, 10:40] = (220, 20, 20)      # red thing on the left
    f[10:40, 150:180] = (20, 20, 220)    # blue thing on the right
    return f


def test_color_picks_between_same_label():
    dets = [Detection("cup", (10, 10, 40, 40), 0.9), Detection("cup", (150, 10, 180, 40), 0.9)]
    g = Grounder(FakeDetector(dets))
    assert g.find(scene(), ObjectRef("cup", "red")).bbox == (10, 10, 40, 40)
    assert g.find(scene(), ObjectRef("cup", "blue")).bbox == (150, 10, 180, 40)


def test_position_picks_between_same_label():
    dets = [Detection("cup", (10, 10, 40, 40), 0.5), Detection("cup", (150, 10, 180, 40), 0.9)]
    g = Grounder(FakeDetector(dets))
    assert g.find(scene(), ObjectRef("cup", position="left")).bbox == (10, 10, 40, 40)


def test_blob_fallback_for_unknown_noun():
    g = Grounder(FakeDetector([]))
    d = g.find(scene(), ObjectRef("screwdriver", "red"))
    assert d is not None and d.label == "screwdriver"
    x1, y1, x2, y2 = d.bbox
    assert x2 <= 60 and (x1, y1) == (10, 10)


def test_no_match_returns_none():
    g = Grounder(FakeDetector([]), blob_fallback=False)
    assert g.find(scene(), ObjectRef("screwdriver", "red")) is None
    assert Grounder(FakeDetector([])).find(scene(), ObjectRef("screwdriver", "purple")) is None


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_decode_yolov8_maps_back_to_frame():
    nc = len(COCO_LABELS)
    out = np.zeros((1, 4 + nc, 2), np.float32)
    # one box in letterboxed 640 space: center (340, 320), 100x100, class "cup"; scale 0.5, pad (20, 0)
    out[0, :4, 0] = (340, 320, 100, 100)
    out[0, 4 + COCO_LABELS.index("cup"), 0] = 0.8
    dets = decode_yolov8(out, COCO_LABELS, 0.25, 0.45, 0.5, (20, 0), (640, 1280))
    assert len(dets) == 1 and dets[0].label == "cup"
    # x: (290-20)/0.5=540 .. (390-20)/0.5=740; y: 540 .. 740 clipped to frame height 640
    assert dets[0].bbox == (540, 540, 740, 640)


def test_smoother_holds_then_drops():
    s = TemporalSmoother(alpha=0.5, max_misses=2)
    d = Detection("cup", (0, 0, 10, 10), 0.9)
    assert s.update(d).bbox == (0, 0, 10, 10)
    assert s.update(None) is not None and s.update(None) is not None
    assert s.update(None) is None


def test_known_noun_wrong_color_does_not_fall_back_to_blob():
    # detector recognises "cup" but it is blue; a red blob elsewhere must not be reported as the cup
    f = scene()
    dets = [Detection("cup", (150, 10, 180, 40), 0.9)]
    assert Grounder(FakeDetector(dets)).find(f, ObjectRef("cup", "red")) is None


class FakeOpenVocab:
    open_vocab = True

    def __init__(self):
        self.queries = None

    def detect(self, frame, queries=None):
        self.queries = queries
        return [Detection(queries[0], (10, 10, 40, 40), 0.8)]


def test_open_vocab_gets_noun_and_never_blob_falls_back():
    det = FakeOpenVocab()
    g = Grounder(det)
    assert g.find(scene(), ObjectRef("screwdriver", "red")).label == "screwdriver"
    assert det.queries == ["screwdriver"]
    assert g.find(scene(), ObjectRef("screwdriver", "blue")) is None  # wrong colour, no blob fallback
    assert g.find(scene(), ObjectRef(None, "red")) is None  # no noun: nothing to query


# --- real models (skipped unless `python -m vision.download` has been run) ---
import os

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
HAVE_MODELS = all(os.path.exists(os.path.join(MODELS, f)) for f in
                  ("yolov8s-worldv2.onnx", "onnx/text_model.onnx", "tokenizer.json", "samples/bus.jpg"))


@pytest.mark.skipif(not HAVE_MODELS, reason="models not downloaded")
def test_open_vocab_end_to_end():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from vision.openvocab import ClipTextEncoder, YoloWorldDetector

    img = cv2.cvtColor(cv2.imread(os.path.join(MODELS, "samples/bus.jpg")), cv2.COLOR_BGR2RGB)
    enc = ClipTextEncoder(os.path.join(MODELS, "onnx/text_model.onnx"), os.path.join(MODELS, "tokenizer.json"))
    e = enc.encode(["cup", "mug", "dog"])
    assert abs(np.linalg.norm(e[0]) - 1) < 1e-4 and e[0] @ e[1] > e[0] @ e[2]
    g = Grounder(YoloWorldDetector(os.path.join(MODELS, "yolov8s-worldv2.onnx"), enc,
                                   providers=["CPUExecutionProvider"]))
    bus = g.find(img, ObjectRef("bus"))
    assert bus and bus.score > 0.5 and bus.area > 0.3 * img.shape[0] * img.shape[1]
    assert g.find(img, ObjectRef("dog")) is None
    assert g.find(img, ObjectRef("bus", "red")) is None
    left, right = g.find(img, ObjectRef("person", position="left")), g.find(img, ObjectRef("person", position="right"))
    assert left.center[0] < right.center[0]
