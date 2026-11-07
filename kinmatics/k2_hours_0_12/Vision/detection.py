"""Color-blob detection for hours 0-12. No learned classifier or table geometry."""

import argparse
import json
import math
import time

import cv2
import numpy as np

from config import load_config
from Vision.camera import open_camera, read_frame


def _settings(config, color):
    """Validate vision-specific options while the shared loader permits extensions."""
    try:
        settings = config["vision"]
        ranges = settings["hsv_ranges"][color]
        area = settings["min_area_px"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"vision: missing settings or unsupported color {color!r}") from exc
    if isinstance(area, bool) or not isinstance(area, (int, float)) or not math.isfinite(area) or area <= 0:
        raise ValueError("vision.min_area_px: expected a finite number > 0")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError(f"vision.hsv_ranges.{color}: expected one or more [lower, upper] HSV ranges")
    checked = []
    for i, pair in enumerate(ranges):
        path = f"vision.hsv_ranges.{color}[{i}]"
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{path}: expected [lower_HSV, upper_HSV]")
        for endpoint in pair:
            if not isinstance(endpoint, list) or len(endpoint) != 3:
                raise ValueError(f"{path}: HSV endpoints must have three integers")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in endpoint):
                raise ValueError(f"{path}: HSV endpoints must be integers")
        low, high = np.asarray(pair[0], dtype=int), np.asarray(pair[1], dtype=int)
        if np.any(low < 0) or np.any(high > [179, 255, 255]) or np.any(low > high):
            raise ValueError(f"{path}: require 0 <= lower <= upper <= [179, 255, 255]")
        checked.append((low.astype(np.uint8), high.astype(np.uint8)))
    return checked, float(area)


def detect_blocks(frame, *, color="red", config=None):
    """Return a list of six-key vision messages and a cleaned uint8 mask.

    Input is an OpenCV BGR uint8 frame. pixel_xy uses u right, v down in the
    ORIGINAL frame. table_xy and angle are None until calibrated; no placeholder
    homography is applied. Confidence is contour rectangularity, NOT probability.
    Assumes the selected-color blobs are blocks in a controlled bench scene.
    """
    config = load_config() if config is None else config
    ranges, min_area = _settings(config, color)
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or frame.size == 0:
        raise ValueError("frame: expected a nonempty HxWx3 uint8 BGR image")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        rectangle_area = width * height
        if rectangle_area <= 0:
            continue
        message = {
            "class": "block", "color": color,
            "pixel_xy": [float(moments["m10"] / moments["m00"]),
                         float(moments["m01"] / moments["m00"])],
            "table_xy": None, "angle": None,
            "confidence": float(np.clip(area / rectangle_area, 0.0, 1.0)),
        }
        detections.append((area, message))
    detections.sort(key=lambda item: item[0], reverse=True)
    return [message for _, message in detections], mask


def draw_detections(frame, detections, mask):
    """Draw contour bounding boxes and centroids, leaving the input frame intact."""
    display = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for message in detections:
        u, v = message["pixel_xy"]
        # Locate the contour containing this detection without adding bbox fields
        # to the locked six-key vision/planner message.
        for contour in contours:
            if cv2.pointPolygonTest(contour, (u, v), False) >= 0:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break
        center = (round(u), round(v))
        cv2.circle(display, center, 4, (255, 255, 255), -1)
        cv2.putText(display, f"{message['color']} ({u:.0f}, {v:.0f})",
                    (center[0] + 6, max(15, center[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index, stream URL, or video path")
    parser.add_argument("--color", default="red", help="color key from arm_config.yaml vision.hsv_ranges")
    parser.add_argument("--image", help="test a saved image instead of opening a camera")
    parser.add_argument("--output", help="save annotated still image; requires --image")
    args = parser.parse_args()
    if args.output and not args.image:
        parser.error("--output requires --image")
    config = load_config()
    capture = None
    try:
        if args.image:
            frame = cv2.imread(args.image)
            detections, mask = detect_blocks(frame, color=args.color, config=config)
            print(json.dumps(detections, allow_nan=False))
            if args.output and not cv2.imwrite(args.output, draw_detections(frame, detections, mask)):
                raise RuntimeError(f"Could not save image to {args.output}")
            return
        capture = open_camera(args.source, resolution=config["camera"]["resolution"])
        last_print = 0.0
        first = True
        while True:
            frame = read_frame(capture)
            if first:
                h, w = frame.shape[:2]
                print(f"Actual capture {w}x{h}. q quits; JSON is pixel-only/unlocalized.", flush=True)
                first = False
            detections, mask = detect_blocks(frame, color=args.color, config=config)
            now = time.monotonic()
            if now - last_print >= 0.5:
                print(json.dumps(detections, allow_nan=False), flush=True)
                last_print = now
            cv2.imshow("K2 blocks", draw_detections(frame, detections, mask))
            cv2.imshow("K2 color mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Detection error: {exc}\n")
    except KeyboardInterrupt:
        pass
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
