import json

import cv2
import numpy as np
import pytest

from config import load_config
from Vision.detection import detect_blocks, draw_detections


def blank():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.mark.parametrize("color,bgr", [("red", (0, 0, 255)), ("green", (0, 255, 0)), ("blue", (255, 0, 0))])
def test_colored_block_center_and_contract(color, bgr):
    frame = blank()
    cv2.rectangle(frame, (60, 80), (120, 140), bgr, -1)
    objects, mask = detect_blocks(frame, color=color)
    assert len(objects) == 1
    obj = objects[0]
    assert set(obj) == {"class", "color", "pixel_xy", "table_xy", "angle", "confidence"}
    assert obj["class"] == "block"
    assert obj["color"] == color
    np.testing.assert_allclose(obj["pixel_xy"], [90, 110], atol=1)
    assert obj["table_xy"] is None
    assert obj["angle"] is None
    assert 0 <= obj["confidence"] <= 1
    assert mask.shape == frame.shape[:2]
    json.dumps(objects, allow_nan=False)
    before = frame.copy()
    display = draw_detections(frame, objects, mask)
    assert np.any(display != frame)
    np.testing.assert_array_equal(frame, before)


def test_red_wraparound_upper_hue_band():
    hsv = blank()
    hsv[80:140, 60:120] = [175, 255, 255]
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    objects, _ = detect_blocks(frame)
    assert len(objects) == 1


def test_empty_frame_has_no_detections():
    objects, mask = detect_blocks(blank())
    assert objects == []
    assert not np.any(mask)


def test_small_specks_ignored_and_multiple_blocks_detected():
    frame = blank()
    frame[10:12, 10:12] = [0, 0, 255]
    frame[30:60, 30:60] = [0, 0, 255]
    frame[100:160, 150:210] = [0, 0, 255]
    objects, _ = detect_blocks(frame)
    assert len(objects) == 2
    # Largest area is returned first, making demo selection deterministic.
    np.testing.assert_allclose(objects[0]["pixel_xy"], [179.5, 129.5], atol=1)


def test_wrong_color_ignored():
    frame = blank()
    frame[20:100, 20:100] = [255, 0, 0]
    assert detect_blocks(frame, color="red")[0] == []


@pytest.mark.parametrize("frame", [None, np.zeros((0, 0, 3), dtype=np.uint8),
                                     np.zeros((10, 10), dtype=np.uint8),
                                     np.zeros((10, 10, 3), dtype=float)])
def test_bad_image_has_clear_error(frame):
    with pytest.raises(ValueError, match="frame"):
        detect_blocks(frame)


@pytest.mark.parametrize("ranges", [[], [[[0, 0], [10, 255, 255]]],
                                    [[[20, 0, 0], [10, 255, 255]]],
                                    [[[0, 0, 0], [180, 255, 255]]],
                                    [[[0, True, 0], [10, 255, 255]]]])
def test_bad_thresholds_have_clear_error(ranges):
    cfg = load_config()
    cfg["vision"]["hsv_ranges"]["red"] = ranges
    with pytest.raises(ValueError, match="vision.hsv_ranges.red"):
        detect_blocks(blank(), config=cfg)


def test_unknown_color_has_clear_error():
    with pytest.raises(ValueError, match="vision"):
        detect_blocks(blank(), color="purple")


def test_configurable_minimum_area():
    frame = blank()
    frame[30:50, 30:50] = [0, 0, 255]
    cfg = load_config()
    cfg["vision"]["min_area_px"] = 500
    assert detect_blocks(frame, config=cfg)[0] == []


@pytest.mark.parametrize("area", [-1, 0, True, "300", float("nan")])
def test_invalid_minimum_area(area):
    cfg = load_config()
    cfg["vision"]["min_area_px"] = area
    with pytest.raises(ValueError, match="vision.min_area_px"):
        detect_blocks(blank(), config=cfg)
