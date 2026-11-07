"""Video stream helpers: frame source and temporal smoothing of detections."""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from nlp.actions import ObjectRef

from .grounding import Detection, Grounder


def frames(source=0) -> Iterator[np.ndarray]:
    """Yield RGB frames from a camera index, file path or stream URL."""
    import cv2

    cap = cv2.VideoCapture(source)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


class TemporalSmoother:
    """Exponentially smooth the box over frames; drop the target after `max_misses` misses."""

    def __init__(self, alpha: float = 0.6, max_misses: int = 5):
        self.alpha, self.max_misses = alpha, max_misses
        self._box: Optional[np.ndarray] = None
        self._misses = 0
        self._last: Optional[Detection] = None

    def update(self, det: Optional[Detection]) -> Optional[Detection]:
        if det is None:
            self._misses += 1
            if self._misses > self.max_misses:
                self._box = self._last = None
            return self._last
        self._misses = 0
        new = np.array(det.bbox, dtype=np.float32)
        self._box = new if self._box is None else self.alpha * new + (1 - self.alpha) * self._box
        self._last = Detection(det.label, tuple(int(round(v)) for v in self._box), det.score)
        return self._last


def track(grounder: Grounder, ref: ObjectRef, source=0, smoother: Optional[TemporalSmoother] = None
          ) -> Iterator[tuple[np.ndarray, Optional[Detection]]]:
    """Yield (frame, smoothed detection or None) for every frame of the stream."""
    smoother = smoother or TemporalSmoother()
    for frame in frames(source):
        yield frame, smoother.update(grounder.find(frame, ref))
