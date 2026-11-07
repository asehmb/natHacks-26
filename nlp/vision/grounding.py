"""Ground an ObjectRef (noun + color + size/position) in a frame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from nlp.actions import ObjectRef

from .blobs import largest_blob
from .hsv import color_fraction, color_mask


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    score: float

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """frame: RGB uint8 (H, W, 3)."""


def label_matches(noun: str, label: str) -> bool:
    n, l = noun.lower(), label.lower()
    return n == l or n in l or l in n


def _crop(frame: np.ndarray, d: Detection) -> np.ndarray:
    x1, y1, x2, y2 = d.bbox
    return frame[max(0, y1):y2, max(0, x1):x2]


def _pick_by_layout(cands: list[Detection], ref: ObjectRef, shape) -> Detection:
    """Apply position / size words to choose among equally good candidates."""
    keys = {
        "left": lambda d: d.center[0], "leftmost": lambda d: d.center[0],
        "right": lambda d: -d.center[0], "rightmost": lambda d: -d.center[0],
        "top": lambda d: d.center[1], "bottom": lambda d: -d.center[1],
        "nearest": lambda d: -d.area, "closest": lambda d: -d.area, "front": lambda d: -d.area,
        "farthest": lambda d: d.area,
        "middle": lambda d: abs(d.center[0] - shape[1] / 2) + abs(d.center[1] - shape[0] / 2),
    }
    if ref.position in keys:
        return min(cands, key=keys[ref.position])
    if ref.size in ("big", "large", "huge", "long"):
        return max(cands, key=lambda d: d.area)
    if ref.size in ("small", "tiny", "short"):
        return min(cands, key=lambda d: d.area)
    return max(cands, key=lambda d: d.score)


class Grounder:
    """Find the object an ObjectRef describes.

    Detector candidates are filtered by label, then by color (HSV). If the detector has no class
    for the noun (e.g. "screwdriver" with a COCO model) and a color was given, falls back to the
    largest blob of that color so colored objects can still be located.
    """

    def __init__(self, detector: Optional[Detector] = None, min_color_fraction: float = 0.15,
                 blob_fallback: bool = True, min_blob_fraction: float = 0.002):
        self.detector = detector
        self.min_color_fraction = min_color_fraction
        self.blob_fallback = blob_fallback
        self.min_blob_fraction = min_blob_fraction

    def find(self, frame: np.ndarray, ref: ObjectRef) -> Optional[Detection]:
        cands = self._detect(frame, ref)
        if ref.noun:
            cands = [d for d in cands if label_matches(ref.noun, d.label)]
        known_noun = bool(cands)  # the detector recognised the noun (even if colour rules it out)
        if ref.color:
            scored = []
            for d in cands:
                frac = color_fraction(_crop(frame, d), ref.color)
                if frac >= self.min_color_fraction:
                    scored.append(Detection(d.label, d.bbox, d.score * (0.5 + min(frac, 0.5))))
            cands = scored
        if cands:
            return _pick_by_layout(cands, ref, frame.shape)
        # Blob fallback is only for nouns the detector can't recognise at all; for an open-vocab
        # detector "nothing found" is a real answer, and a stray colour blob would be a false match.
        open_vocab = getattr(self.detector, "open_vocab", False)
        if self.blob_fallback and ref.color and not known_noun and not open_vocab:
            return self._largest_blob(frame, ref)
        return None

    def _detect(self, frame: np.ndarray, ref: ObjectRef) -> list[Detection]:
        if self.detector is None:
            return []
        if getattr(self.detector, "open_vocab", False):
            # Open-vocab models need the noun to search for; colour is applied afterwards via HSV.
            return self.detector.detect(frame, [ref.noun]) if ref.noun else []
        return self.detector.detect(frame)

    def _largest_blob(self, frame: np.ndarray, ref: ObjectRef) -> Optional[Detection]:
        mask = color_mask(frame, ref.color)
        found = largest_blob(mask)
        if found is None:
            return None
        bbox, area = found
        if area / mask.size < self.min_blob_fraction:
            return None
        return Detection(ref.noun or ref.color, bbox, float(area / mask.size))
