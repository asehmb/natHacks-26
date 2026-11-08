"""Open a camera index or an existing video URL/file, and preview RGB frames."""

import argparse
import sys

import cv2

from Kinematics.config import load_config


def parse_source(source):
    """Convert CLI '0' to camera 0; keep URLs and file paths as strings."""
    if isinstance(source, bool):
        raise ValueError("source: expected a nonnegative camera index or URL/path")
    if isinstance(source, int):
        if source < 0:
            raise ValueError("source: camera index must be nonnegative")
        return source
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source: expected a camera index or URL/path")
    source = source.strip()
    return int(source) if source.isdecimal() else source


def open_camera(source=0, *, resolution=None):
    """Return an opened cv2.VideoCapture. Caller must release it.

    Resolution is a request, not a guarantee. Never silently resize frames.
    On macOS, numeric camera indexes use AVFoundation, including an iPhone if
    macOS exposes it. A stream URL must already be provided by a streaming app.
    """
    source = parse_source(source)
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and isinstance(source, int) else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Cannot open camera source {source!r}. Check camera permissions, "
            "try another index (0, 1, 2), or provide a working stream URL."
        )
    if resolution is not None:
        width, height = resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def read_frame(capture):
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError("Camera returned no frame. Check the phone connection and camera selection.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index, stream URL, or video path")
    args = parser.parse_args()
    config = load_config()
    capture = None
    try:
        capture = open_camera(args.source, resolution=config["camera"]["resolution"])
        first = True
        while True:
            frame = read_frame(capture)
            if first:
                h, w = frame.shape[:2]
                print(f"Receiving {w}x{h}. Confirm this is the iPhone view. Press q to quit.")
                first = False
            cv2.imshow("K2 camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (RuntimeError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Camera error: {exc}\n")
    except KeyboardInterrupt:
        pass
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
