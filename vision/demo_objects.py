"""Object-only demo (no arm): labels each object on the mat and prints its position in mm.

  python demo_objects.py --source 0                    # webcam; press r on an empty mat to capture the reference
  python demo_objects.py --source 0 --mode yolo        # YOLO finds the objects itself, no reference needed
  python demo_objects.py --source photo.jpg --mode yolo --save out.png
  python demo_objects.py --source 0 --vocab "red cup,screwdriver,tape roll"

Positions are in the arm base frame (mat_config.yaml: arm.base_xy_mm / base_yaw_deg): x forward, y left, mm.
Keys: r = capture empty-mat reference, s = save screenshot, q = quit.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from armvision.config import ConfigError, load_config
from armvision.intrinsics import load_intrinsics
from armvision.__main__ import open_camera
from armvision.overlay import draw_object
from armvision.pipeline import FrameSizeError, VisionSystem
from armvision.yolo import DEFAULT_VOCAB, YoloWorld, find_models

COLORS = {"red": (60, 60, 255), "orange": (0, 165, 255), "yellow": (0, 255, 255), "green": (80, 200, 80),
          "blue": (255, 140, 40), "purple": (200, 80, 160), "pink": (200, 130, 255)}


def draw(vs: VisionSystem, frame, state, fps):
    out = (vs.last_frame if vs.last_frame is not None else frame).copy()   # pose is in undistorted pixels
    pose = vs.pose
    if pose is None:
        cv2.putText(out, "mat not found (need 2+ corner markers in view)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    w, h = vs.cfg.mat.width_mm, vs.cfg.mat.height_mm
    cv2.polylines(out, [pose.mat_to_img([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)], True, (255, 200, 0), 2)
    for b in vs.last_boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)
    for o in state.objects:
        draw_object(out, vs, o)
        gx, gy = pose.mat_to_img(vs.frames.base_to_mat(o.grasp_mm[:2]))[0].astype(int)
        cv2.drawMarker(out, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        x, y = int(o.center_mm[0]), int(o.center_mm[1])
        cx, cy = pose.mat_to_img(vs.frames.base_to_mat(o.center_mm[:2]))[0].astype(int)
        t = f"x={x} y={y} mm"
        cv2.putText(out, t, (cx + 10, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, t, (cx + 10, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if vs.mode == "diff" and vs.reference is None:
        cv2.putText(out, "press r with an empty mat", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(out, f"{vs.mode}  {fps:.0f} fps", (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def print_table(state, last=[None]):
    rows = [f"#{o.id:<3}{o.label:<14}{o.color:<8}x={o.center_mm[0]:7.1f}  y={o.center_mm[1]:7.1f}  "
            f"size={o.size_mm[0]:.0f}x{o.size_mm[1]:.0f}mm" for o in state.objects]
    if rows != last[0]:       # only when something changed, so the terminal stays readable
        last[0] = rows
        print("\n" + (f"{len(rows)} object(s):\n  " + "\n  ".join(rows) if rows else "no objects"), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, stream URL, video or image path")
    ap.add_argument("--mode", choices=["diff", "yolo"], default=None,
                    help="diff: segment against an empty-mat reference, YOLO names them (default if no models: diff "
                         "unlabeled). yolo: YOLO detects them directly")
    ap.add_argument("--vocab", help="comma-separated things to look for")
    ap.add_argument("--models", help="dir with yolov8s-worldv2.onnx, onnx/text_model.onnx, tokenizer.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--yolo-every", type=int, default=5, help="run YOLO every N frames")
    ap.add_argument("--config")
    ap.add_argument("--intrinsics", default="camera_intrinsics.yaml")
    ap.add_argument("--reference", help="PNG reference frame: loaded if it exists, saved on 'r'")
    ap.add_argument("--save", help="write the annotated frame here (single image sources / on 's')")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        ap.exit(1, f"config error: {exc}\n")
    labeler = None
    if find_models(args.models):
        try:
            labeler = YoloWorld.load(args.models, conf=args.conf)
        except RuntimeError as exc:
            print(f"warning: {exc}", file=sys.stderr)
    else:
        print("warning: YOLO models not found (pass --models, or run `python -m armvision.download`). "
              "Objects will be unlabeled.", file=sys.stderr)
    mode = args.mode or "diff"
    if mode == "yolo" and labeler is None:
        ap.exit(1, "--mode yolo needs the YOLO models and `pip install -e '.[yolo]'`\n")
    vocab = [v.strip() for v in args.vocab.split(",") if v.strip()] if args.vocab else DEFAULT_VOCAB
    vs = VisionSystem(cfg, load_intrinsics(args.intrinsics), labeler, vocab, mode, track_arm=False,
                      yolo_every=args.yolo_every)
    warnings.simplefilter("once")
    if args.reference and Path(args.reference).exists():
        vs.set_reference(cv2.imread(args.reference))

    still = Path(args.source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    cap = None
    if not still:
        try:
            cap = open_camera(args.source)
        except RuntimeError as exc:
            ap.exit(1, f"{exc}\n")
    try:
        t0, fps = time.time(), 0.0
        while True:
            if still:
                frame = cv2.imread(args.source)
                if frame is None:
                    ap.exit(1, f"cannot read image {args.source}\n")
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
            if mode == "diff" and vs.reference is None and still:
                print("a still image in diff mode has no empty-mat reference; use --mode yolo", file=sys.stderr)
            try:
                state = vs.process(frame)
            except FrameSizeError as exc:
                print(f"skipping frame: {exc}", file=sys.stderr)
                if still:
                    break
                continue
            print_table(state)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t0, 1e-6) if fps else 1 / max(now - t0, 1e-6)
            t0 = now
            shown = draw(vs, frame, state, fps)
            if still and args.save:
                cv2.imwrite(args.save, shown)
                print(f"wrote {args.save}")
                break
            cv2.imshow("armvision objects", shown)
            key = cv2.waitKey(0 if still else 1) & 0xFF
            if key == ord("q") or still:
                break
            if key == ord("r"):
                print("reference captured" if vs.capture_reference(frame) else "mat not visible", file=sys.stderr)
                if args.reference and vs.reference is not None:
                    cv2.imwrite(args.reference, vs.reference)
            elif key == ord("s"):
                cv2.imwrite(args.save or "objects.png", shown)
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
