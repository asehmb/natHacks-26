"""Greedy nearest-neighbour tracking in mat mm with EMA smoothing and occlusion awareness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Observation:
    center: np.ndarray       # mat mm
    grasp: np.ndarray        # mat mm
    yaw_deg: float           # mat frame, period 180
    size: tuple[float, float]
    color: str


@dataclass
class Track:
    id: int
    obs: Observation
    age: int = 1
    missed: int = 0
    occluded: bool = False


def _ema_yaw(old: float, new: float, w: float) -> float:
    a, b = np.radians(2 * old), np.radians(2 * new)   # double angle handles the 180-degree period
    x, y = (1 - w) * np.cos(a) + w * np.cos(b), (1 - w) * np.sin(a) + w * np.sin(b)
    d = np.degrees(np.arctan2(y, x)) / 2
    return 90.0 if d == -90.0 else d


class Tracker:
    def __init__(self, gate_mm: float = 40.0, max_missed: int = 15, ema: float = 0.5):
        self.gate, self.max_missed, self.ema = gate_mm, max_missed, ema
        self.tracks: list[Track] = []
        self._next = 1

    def reset(self):
        self.tracks, self._next = [], 1

    def update(self, observations: list[Observation], in_arm: Callable[[np.ndarray], bool] = lambda p: False):
        pairs = sorted(
            ((float(np.linalg.norm(t.obs.center - o.center)), ti, oi)
             for ti, t in enumerate(self.tracks) for oi, o in enumerate(observations)),
            key=lambda x: x[0])
        used_t, used_o = set(), set()
        for d, ti, oi in pairs:
            if d > self.gate:
                break
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            t, o, w = self.tracks[ti], observations[oi], self.ema
            old = t.obs
            t.obs = Observation(
                (1 - w) * old.center + w * o.center, (1 - w) * old.grasp + w * o.grasp,
                _ema_yaw(old.yaw_deg, o.yaw_deg, w),
                ((1 - w) * old.size[0] + w * o.size[0], (1 - w) * old.size[1] + w * o.size[1]), o.color)
            t.age += 1
            t.missed = 0
            t.occluded = False
        keep = []
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.age += 1
                t.occluded = bool(in_arm(t.obs.center))
                if t.occluded:
                    t.missed = 0      # kept while the arm is over it
                else:
                    t.missed += 1
                    if t.missed > self.max_missed:
                        continue
            keep.append(t)
        self.tracks = keep
        for oi, o in enumerate(observations):
            if oi not in used_o:
                self.tracks.append(Track(self._next, o))
                self._next += 1
        return self.tracks
