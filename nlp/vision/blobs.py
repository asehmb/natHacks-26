"""Largest connected region of a boolean mask, numpy + stdlib only (no OpenCV needed)."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

MAX_GRID = 96  # the mask is labelled on a grid at most this many cells on its short side


def largest_blob(mask: np.ndarray) -> Optional[tuple[tuple[int, int, int, int], int]]:
    """Return ((x1, y1, x2, y2), pixel_count) of the largest 8-connected region, or None if empty.

    Regions are found on a coarse grid (cells holding any set pixel), so two blobs separated by less
    than one cell merge. Small masks use a 1-pixel grid and are exact. The box is then tightened to
    the real pixels of the chosen region.
    """
    if not mask.any():
        return None
    h, w = mask.shape
    b = max(1, min(h, w) // MAX_GRID)
    gh, gw = -(-h // b), -(-w // b)
    padded = np.zeros((gh * b, gw * b), bool)
    padded[:h, :w] = mask
    counts = padded.reshape(gh, b, gw, b).sum(axis=(1, 3))  # set pixels per cell
    cells = counts > 0

    labels = np.zeros((gh, gw), np.int32)
    best, best_count, n = 0, 0, 0
    for sy, sx in zip(*np.nonzero(cells)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        total, queue = 0, deque([(sy, sx)])
        while queue:
            y, x = queue.popleft()
            total += int(counts[y, x])
            for ny in range(max(0, y - 1), min(gh, y + 2)):
                for nx in range(max(0, x - 1), min(gw, x + 2)):
                    if cells[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        queue.append((ny, nx))
        if total > best_count:
            best, best_count = n, total

    region = np.repeat(np.repeat(labels == best, b, axis=0), b, axis=1)[:h, :w] & mask
    ys, xs = np.nonzero(region)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), best_count
