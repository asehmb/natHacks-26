"""Live demo: python -m armvision --source 0   (keys: r=capture empty mat, c=add calibration view, k=solve+save, q=quit)"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from .config import ConfigError, load_config
from .intrinsics import DEFAULT_PATH, calibrate, load_intrinsics, view_points
from .pipeline import VisionSystem


def open_camera(source):
    """Camera index or URL/path; AVFoundation on macOS (same pattern as Kinematics/k2_hours_0_12/Vision/camera.py)."""
    source = int(source) if str(source).isdecimal() else source
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and isinstance(source, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera source {source!r}. Check permissions or try another index.")
    return cap


def draw_overlay(vs: VisionSystem, frame, state):
    out = frame.copy()
    pose = vs.pose
    for mid, c in vs.last_markers.items():
        cv2.polylines(out, [c.astype(np.int32)], True, (0, 255, 0), 2)
    if pose is None:
        cv2.putText(out, "mat not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return out
    w, h = vs.cfg.mat.width_mm, vs.cfg.mat.height_mm
    outline = pose.mat_to_img([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)
    cv2.polylines(out, [outline], True, (255, 200, 0), 2)
    for o in state.objects:
        gm = vs.frames.base_to_mat(o.grasp_mm[:2])
        cm = vs.frames.base_to_mat(o.center_mm[:2])
        gp, cp = pose.mat_to_img(gm)[0], pose.mat_to_img(cm)[0]
        col = (0, 165, 255) if o.occluded else (0, 0, 255)
        cv2.circle(out, tuple(int(x) for x in cp), 4, (255, 255, 255), -1)
        cv2.circle(out, tuple(int(x) for x in gp), 6, col, -1)
        color_tag = f" {o.color}" if o.color and o.color != "unknown" else ""
        cv2.putText(out, f"#{o.id}{color_tag} ({o.grasp_mm[0]:.0f},{o.grasp_mm[1]:.0f})",
                    (int(cp[0]) + 8, int(cp[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if state.arm.visible:
        x, y, z = state.arm.gripper_mm
        g = vs.last_markers[vs.cfg.arm.gripper_marker_id].mean(0)
        cv2.putText(out, f"gripper ({x:.0f},{y:.0f},{z:.0f})", (int(g[0]) + 10, int(g[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if vs.reference is None:
        cv2.putText(out, "press r with an empty mat", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0", help="camera index, stream URL or video path")
    ap.add_argument("--config")
    ap.add_argument("--intrinsics", default=str(DEFAULT_PATH))
    ap.add_argument("--reference", help="PNG of the empty rectified mat: loaded if it exists, saved on 'r'")
    ap.add_argument("--no-gui", action="store_true", help="headless: reference is taken from the first frame with the mat")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        ap.exit(1, f"config error: {exc}\n")
    vs = VisionSystem(cfg, load_intrinsics(args.intrinsics))
    if args.reference and Path(args.reference).exists():
        vs.set_reference(cv2.imread(args.reference))
    warnings.simplefilter("once")
    views = []
    calib_size = None
    try:
        cap = open_camera(args.source)
    except RuntimeError as exc:
        ap.exit(1, f"{exc}\n")
    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                break
            frame = raw_frame.copy()
            if args.no_gui and vs.reference is None:
                vs.capture_reference(frame)
            state = vs.process(frame)
            print(json.dumps(state.to_dict()), flush=True)
            if args.no_gui:
                continue
            cv2.imshow("armvision", draw_overlay(vs, frame, state))
            if vs.last_rect is not None:
                cv2.imshow("rectified", vs.last_rect)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                print("reference captured" if vs.capture_reference(raw_frame) else "mat not visible", file=sys.stderr)
                if args.reference and vs.reference is not None:
                    cv2.imwrite(args.reference, vs.reference)
            elif key == ord("c"):
                raw_markers = vs.last_raw_markers if vs.last_raw_markers else vs.detector.detect(raw_frame)
                vp = view_points(raw_markers, cfg)
                if vp is None:
                    print("not enough mat markers visible for a calibration view", file=sys.stderr)
                else:
                    h, w = raw_frame.shape[:2]
                    current_size = (w, h)
                    if calib_size is not None and current_size != calib_size:
                        print(f"resolution changed from {calib_size} to {current_size}; resetting views", file=sys.stderr)
                        views.clear()
                    calib_size = current_size
                    views.append(vp)
                    print(f"calibration views: {len(views)} (move the camera/mat; press k with >= 10)", file=sys.stderr)
            elif key == ord("k"):
                try:
                    h, w = raw_frame.shape[:2]
                    if calib_size is not None and (w, h) != calib_size:
                        raise ValueError(f"frame size {(w, h)} does not match calibration views size {calib_size}")
                    intr = calibrate(views, (w, h))
                    intr.save(args.intrinsics)
                    vs.intrinsics = intr
                    views.clear()
                    calib_size = None
                    print(f"saved {args.intrinsics} (rms {intr.rms:.2f} px); reset mat reference", file=sys.stderr)
                except (ValueError, cv2.error) as exc:
                    print(f"calibration failed: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
