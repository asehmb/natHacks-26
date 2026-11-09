"""Open-vocabulary YOLO-World (ONNX) + CLIP text encoder, used to label objects and to detect them directly.

Models live in vision/models (`python -m armvision.download` fetches them). Needs the optional extra:
pip install -e '.[yolo]'  (onnxruntime, tokenizers, huggingface_hub).

Two stages: YOLO-World proposes labelled boxes, then CLIP re-scores each crop against the vocabulary (crops exclude
most of the shadow/background that fools the detector's whole-image view).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Open vocabulary is a closed choice per run: if the true object isn't listed the model picks the nearest listed
# one (a screwdriver becomes "scissors"). Keep look-alikes in the list so they compete with each other.
DEFAULT_VOCAB = (
    "screwdriver", "wrench", "pliers", "hammer", "scissors", "knife", "pen", "marker", "pencil", "ruler",
    "tape", "cup", "bottle", "can", "box", "ball", "toy", "phone", "book", "block", "spoon", "fork", "key",
)
DETECTORS = ("yolov8l-worldv2.onnx", "yolov8s-worldv2.onnx")   # first one present wins; $ARMVISION_DETECTOR overrides
SHARED_FILES = ("onnx/text_model.onnx", "tokenizer.json")
CLIP_IMAGE = "onnx/vision_model.onnx"                           # optional: enables the crop re-classifier
VISION_MODELS = Path(__file__).resolve().parents[1] / "models"
REPO_MODELS = Path(__file__).resolve().parents[2] / "nlp" / "models"
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


@dataclass
class Box:
    label: str
    xyxy: tuple[float, float, float, float]   # camera-image pixels
    score: float

    @property
    def bottom_center(self):
        return ((self.xyxy[0] + self.xyxy[2]) / 2, self.xyxy[3])


def detector_file(d: Path) -> str | None:
    pref = os.environ.get("ARMVISION_DETECTOR")
    for name in ([pref] if pref else []) + list(DETECTORS):
        if (d / name).exists():
            return name
    return None


def find_models(path: str | Path | None = None) -> Path | None:
    """Directory with a detector and the CLIP text files: given path, $ARMVISION_MODELS, vision/models, ./models,
    then ../nlp/models."""
    cands = [path] if path else [os.environ.get("ARMVISION_MODELS"), VISION_MODELS, "models", REPO_MODELS]
    for c in cands:
        if c and detector_file(Path(c)) and all((Path(c) / f).exists() for f in SHARED_FILES):
            return Path(c)
    return None


def clip_preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> (3, 224, 224) float CLIP input. Pads to a square (whole object kept) instead of centre-cropping."""
    h, w = bgr.shape[:2]
    side = max(h, w)
    pad = np.full((side, side, 3), (CLIP_MEAN[::-1] * 255).astype(np.uint8), np.uint8)
    pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = bgr
    rgb = cv2.cvtColor(cv2.resize(pad, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return out, scale, (px, py)


def decode(output: np.ndarray, labels, conf: float, iou: float, scale: float, pad, frame_hw) -> list[Box]:
    """(1, 4+nc, N) YOLOv8-style output -> boxes in original-frame pixels (per-class NMS)."""
    preds = output[0].T
    cls_scores = preds[:, 4:]
    cls = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls)), cls]
    m = scores >= conf
    preds, cls, scores = preds[m], cls[m], scores[m]
    if not len(scores):
        return []
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad[0]) / scale).clip(0, frame_hw[1])
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad[1]) / scale).clip(0, frame_hw[0])
    out = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes[idx]]
        keep = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), 0.0, iou)
        for k in np.array(keep).reshape(-1):
            j = idx[int(k)]
            out.append(Box(labels[c], tuple(float(v) for v in boxes[j]), float(scores[j])))
    return out


class YoloWorld:
    def __init__(self, models_dir: str | Path, conf: float = 0.25, iou: float = 0.45, size: int = 640,
                 rerank: bool = True, clip_min: float = 0.35, max_rerank: int = 8):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("YOLO needs onnxruntime and tokenizers: pip install -e '.[yolo]'") from exc
        d = Path(models_dir)
        prov = ["CPUExecutionProvider"]
        self.detector_name = detector_file(d)
        self.detector = ort.InferenceSession(str(d / self.detector_name), providers=prov)
        self.text = ort.InferenceSession(str(d / SHARED_FILES[0]), providers=prov)
        self.tok = Tokenizer.from_file(str(d / SHARED_FILES[1]))
        self.clip = ort.InferenceSession(str(d / CLIP_IMAGE), providers=prov) if rerank and (d / CLIP_IMAGE).exists() else None
        self.clip_min, self.max_rerank = clip_min, max_rerank
        self._cemb: dict[str, np.ndarray] = {}
        self.tok.enable_truncation(max_length=77)
        self.conf, self.iou, self.size = conf, iou, size
        self._emb: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path=None, **kw) -> "YoloWorld | None":
        d = find_models(path)
        return cls(d, **kw) if d else None

    def _encode(self, phrases) -> np.ndarray:
        for p in phrases:
            if p not in self._emb:
                ids = np.array([self.tok.encode(p).ids], dtype=np.int64)
                e = self.text.run(None, {"input_ids": ids})[0][0]
                self._emb[p] = (e / np.linalg.norm(e)).astype(np.float32)
        return np.stack([self._emb[p] for p in phrases])[None]

    def _clip_text(self, queries) -> np.ndarray:
        phrases = [f"a photo of a {q}" for q in queries]
        for q, p in zip(queries, phrases):
            if q not in self._cemb:
                ids = np.array([self.tok.encode(p).ids], dtype=np.int64)
                e = self.text.run(None, {"input_ids": ids})[0][0]
                self._cemb[q] = (e / np.linalg.norm(e)).astype(np.float32)
        return np.stack([self._cemb[q] for q in queries])

    def _clip_probs(self, crops, queries) -> np.ndarray:
        """(n_crops, n_queries) softmax of CLIP image-text similarity."""
        x = np.stack([clip_preprocess(c) for c in crops])
        e = self.clip.run(None, {"pixel_values": x})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        logits = 100.0 * e @ self._clip_text(queries).T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)

    @staticmethod
    def _crop(frame, xyxy, margin=0.1):
        x1, y1, x2, y2 = xyxy
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        h, w = frame.shape[:2]
        x1, x2 = int(max(0, x1 - mx)), int(min(w, x2 + mx))
        y1, y2 = int(max(0, y1 - my)), int(min(h, y2 + my))
        return frame[y1:y2, x1:x2] if x2 - x1 >= 4 and y2 - y1 >= 4 else None

    def rerank_boxes(self, frame, boxes: list[Box], queries) -> list[Box]:
        """Second opinion from CLIP on each box crop. CLIP's pick replaces YOLO's when it is confident (clip_min);
        the score is the geometric mean of the detector score and CLIP's probability for the final label."""
        queries = list(queries)
        order = sorted(range(len(boxes)), key=lambda i: -boxes[i].score)[:self.max_rerank]
        crops = [(i, self._crop(frame, boxes[i].xyxy)) for i in order]
        crops = [(i, c) for i, c in crops if c is not None]
        if not crops:
            return boxes
        probs = self._clip_probs([c for _, c in crops], queries)
        out = list(boxes)
        for (i, _), p in zip(crops, probs):
            b = boxes[i]
            top = int(p.argmax())
            label = queries[top] if p[top] >= self.clip_min else b.label
            pl = float(p[queries.index(label)]) if label in queries else 0.0
            out[i] = Box(label, b.xyxy, float(np.sqrt(b.score * pl)))
        return out

    def classify_crop(self, frame, xyxy, queries) -> tuple[str, float] | None:
        """CLIP-only label for a region (used for objects the detector produced no box for). None without CLIP."""
        if self.clip is None:
            return None
        crop = self._crop(frame, xyxy)
        if crop is None:
            return None
        queries = list(queries)
        p = self._clip_probs([crop], queries)[0]
        return queries[int(p.argmax())], float(p.max())

    def detect(self, frame_bgr: np.ndarray, queries) -> list[Box]:
        queries = list(queries)
        if not queries:
            return []
        img, scale, pad = letterbox(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), self.size)
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.detector.run(None, {"images": x, "txt_feats": self._encode(queries)})[0]
        boxes = decode(out, queries, self.conf, self.iou, scale, pad, frame_bgr.shape[:2])
        return self.rerank_boxes(frame_bgr, boxes, queries) if self.clip is not None and boxes else boxes
