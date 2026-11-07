import time

import numpy as np
import pytest

from vision.blobs import largest_blob


def mask_with(shape, *rects):
    m = np.zeros(shape, bool)
    for x1, y1, x2, y2 in rects:
        m[y1:y2, x1:x2] = True
    return m


def test_empty_mask():
    assert largest_blob(np.zeros((50, 50), bool)) is None
    assert largest_blob(np.zeros((0, 0), bool)) is None


def test_single_blob_is_exact():
    assert largest_blob(mask_with((100, 200), (30, 20, 70, 60))) == ((30, 20, 70, 60), 1600)


def test_picks_largest_of_several_blobs():
    m = mask_with((100, 200), (5, 5, 15, 15), (100, 40, 150, 80), (160, 5, 170, 12))
    assert largest_blob(m) == ((100, 40, 150, 80), 2000)


def test_largest_by_pixels_not_by_box_size():
    # a thin L-shape has a huge bounding box but few pixels; a solid square has more pixels
    m = mask_with((100, 100), (0, 0, 2, 100), (0, 98, 100, 100), (60, 10, 90, 40))
    assert largest_blob(m) == ((60, 10, 90, 40), 900)


def test_diagonal_pixels_are_connected():
    m = np.zeros((10, 10), bool)
    m[2, 2] = m[3, 3] = m[4, 4] = True
    assert largest_blob(m) == ((2, 2, 5, 5), 3)


def test_separated_blobs_do_not_merge_on_exact_grid():
    m = mask_with((20, 40), (0, 0, 5, 5), (7, 0, 12, 5))  # 2px gap
    assert largest_blob(m)[0] in {(0, 0, 5, 5), (7, 0, 12, 5)}


def test_blobs_on_frame_borders():
    m = mask_with((30, 30), (0, 0, 10, 10), (25, 25, 30, 30))
    assert largest_blob(m) == ((0, 0, 10, 10), 100)
    full = np.ones((7, 9), bool)
    assert largest_blob(full) == ((0, 0, 9, 7), 63)


def test_single_pixel():
    m = np.zeros((20, 20), bool)
    m[13, 4] = True
    assert largest_blob(m) == ((4, 13, 5, 14), 1)


def test_large_frame_is_fast_and_tight_on_coarse_grid():
    m = mask_with((480, 640), (100, 200, 190, 330), (400, 50, 420, 70))
    t = time.time()
    box, area = largest_blob(m)
    assert time.time() - t < 1.0
    assert box == (100, 200, 190, 330) and area == 90 * 130  # tightened to real pixels, not grid cells


def test_speckle_noise_on_large_frame_is_fast():
    rng = np.random.default_rng(0)
    m = rng.random((480, 640)) < 0.02  # 2% random single pixels
    m |= mask_with((480, 640), (300, 300, 380, 360))
    t = time.time()
    largest_blob(m)
    assert time.time() - t < 2.0


@pytest.mark.parametrize("shape", [(37, 53), (64, 64), (90, 40)])
def test_matches_opencv_on_small_masks(shape):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(shape[0])
    for _ in range(25):
        m = rng.random(shape) < 0.25
        if not m.any():
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        best = int(stats[1:, cv2.CC_STAT_AREA].max())
        box, area = largest_blob(m)
        assert area == best  # exact grid (min side < MAX_GRID), so identical pixel counts
        x1, y1, x2, y2 = box
        assert m[y1:y2, x1:x2].sum() >= area  # the box contains the whole blob
