"""ONNX Runtime object detector for Ultralytics YOLOv8-style exports (closed-set, CPU friendly).

Export a model with `yolo export model=yolov8n.pt format=onnx`. Output shape is (1, 4 + nc, N).
For open-vocabulary detection, implement the `Detector` protocol (vision.grounding) yourself.
"""
from __future__ import annotations

import numpy as np

from .grounding import Detection

COCO_LABELS = tuple(
    l.replace("_", " ") for l in (
        "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush"
    ).split()
)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to size x size. Returns (image, scale, (pad_x, pad_y))."""
    import cv2

    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, (px, py)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS on (N, 4) x1y1x2y2 boxes."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / np.maximum(area(boxes[i]) + area(boxes[rest]) - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode_yolov8(output: np.ndarray, labels, conf: float, iou: float, scale: float,
                  pad: tuple[int, int], frame_hw: tuple[int, int]) -> list[Detection]:
    """Decode a (1, 4+nc, N) YOLOv8-style output (class channels are probabilities) into detections
    in original-frame pixels."""
    preds = output[0].T  # (N, 4+nc)
    cls_scores = preds[:, 4:]
    cls = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls)), cls]
    m = scores >= conf
    preds, cls, scores = preds[m], cls[m], scores[m]
    if not len(scores):
        return []
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    H, W = frame_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    out = []
    for c in np.unique(cls):  # per-class NMS
        idx = np.nonzero(cls == c)[0]
        for k in nms(boxes[idx], scores[idx], iou):
            j = idx[k]
            out.append(Detection(labels[c], tuple(int(v) for v in boxes[j]), float(scores[j])))
    return out


class YoloOnnxDetector:
    def __init__(self, model_path: str, labels=COCO_LABELS, conf: float = 0.25, iou: float = 0.45,
                 input_size: int = 640, providers=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.labels, self.conf, self.iou, self.size = labels, conf, iou, input_size

    open_vocab = False

    def detect(self, frame: np.ndarray, queries=None) -> list[Detection]:
        """`queries` is ignored: this model only knows its trained classes."""
        img, scale, pad = letterbox(frame, self.size)
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        return decode_yolov8(out, self.labels, self.conf, self.iou, scale, pad, frame.shape[:2])
