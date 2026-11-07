"""Open-vocabulary detection: YOLO-World (ONNX, dynamic text prompts) + CLIP text encoder (ONNX).

Models (see README): yolov8s-worldv2.onnx (Instemic/yolo-world-onnx, AGPL-3.0) and CLIP ViT-B/32
text_model.onnx + tokenizer.json (Xenova/clip-vit-base-patch32). Only onnxruntime, tokenizers,
numpy and opencv are needed at runtime; run `python -m vision.download` once to fetch them.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np

from .detector import decode_yolov8, letterbox
from .grounding import Detection
from .providers import select_providers


class ClipTextEncoder:
    """Phrase -> L2-normalised 512-d CLIP text embedding, with an LRU cache."""

    def __init__(self, model_path: str, tokenizer_path: str, providers=None, cache_size: int = 256):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_length=77)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def encode(self, phrases: Sequence[str]) -> np.ndarray:
        missing = [p for p in dict.fromkeys(phrases) if p not in self._cache]
        for p in missing:  # one at a time: avoids pad-token pooling issues, phrases are tiny
            ids = np.array([self.tok.encode(p).ids], dtype=np.int64)
            e = self.session.run(None, {"input_ids": ids})[0][0]
            self._cache[p] = (e / np.linalg.norm(e)).astype(np.float32)
        out = np.stack([self._cache[p] for p in phrases])
        for p in phrases:
            self._cache.move_to_end(p)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out


class YoloWorldDetector:
    """Detect arbitrary text-described objects. `queries` are the phrases to look for."""

    open_vocab = True

    def __init__(self, model_path: str, text_encoder: ClipTextEncoder, conf: float = 0.25,
                 iou: float = 0.45, input_size: int = 640, providers=None):
        import onnxruntime as ort

        self._ort, self._model_path = ort, model_path
        self.session = ort.InferenceSession(model_path, providers=providers or select_providers())
        self.encoder, self.conf, self.iou, self.size = text_encoder, conf, iou, input_size

    def _run(self, feeds: dict) -> np.ndarray:
        try:
            return self.session.run(None, feeds)[0]
        except Exception:
            if self.session.get_providers() == ["CPUExecutionProvider"]:
                raise
            # An accelerator that loads but fails at inference: drop to CPU for good.
            self.session = self._ort.InferenceSession(self._model_path, providers=["CPUExecutionProvider"])
            return self.session.run(None, feeds)[0]

    def detect(self, frame: np.ndarray, queries: Optional[Sequence[str]] = None) -> list[Detection]:
        if not queries:
            return []
        queries = list(queries)
        txt = self.encoder.encode(queries)[None]  # (1, N, 512)
        img, scale, pad = letterbox(frame, self.size)
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self._run({"images": x, "txt_feats": txt})
        return decode_yolov8(out, queries, self.conf, self.iou, scale, pad, frame.shape[:2])
