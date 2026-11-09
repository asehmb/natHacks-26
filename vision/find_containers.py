"""Marker-free container finder: where is "the red container" on the white sheet?

  python find_containers.py --calibrate --source 0          # once: click the 2 far corners + 2 pen ticks, press s
  python find_containers.py --source 0 --target red         # live overlay, positions in mm
  python find_containers.py --source photo.png --no-gui --json --target red
  python find_containers.py --source photo.png --save out.png
  python find_containers.py --source 0 --target red --port /dev/cu.usbserial-10   # g = hover above it (real arm)
  python find_containers.py --source 0 --target red --no-gui --go                 # move once, simulated arm

Positions are in the arm base frame (container_config.yaml): x forward, y left, z up, mm.
Live keys: g = move the arm --clearance-mm above the --target container (MockArm simulation without --port),
t = toggle top-down view, s = save screenshot, q = quit. --debug lists/draws every colour blob.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from armvision.__main__ import open_camera
from armvision.config import ConfigError
from armvision.container_move import DEFAULT_CLEARANCE_MM, make_arm, move_above_container
from armvision.containers import ContainerFinder, ContainerState, load_container_config
from armvision.groundcal import GroundCalibration, reference_points
from armvision.pipeline import FrameSizeError

DEFAULT_CALIBRATION = "container_calibration.yaml"
PROMPTS = ["1/4  far (camera-end) corner of the sheet, arm's LEFT (right side of the screen)",
           "2/4  far (camera-end) corner of the sheet, arm's RIGHT (left side of the screen)",
           "3/4  pen tick on the arm's LEFT long edge (right side of the screen)",
           "4/4  pen tick on the arm's RIGHT long edge (left side of the screen)"]
COLORS = {"red": (60, 60, 255), "orange": (0, 165, 255), "yellow": (0, 255, 255), "green": (80, 200, 80),
          "blue": (255, 140, 40), "purple": (200, 80, 160), "pink": (200, 130, 255)}


def text(img, s, org, color=(255, 255, 255), scale=0.5):
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:     # outline at the same thickness: glyph spacing matches
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def poly(img, pts, color, thick=2):
    cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, color, thick)


def zone_polygon(cfg, n=24):
    """Zone outline in the arm frame (box clipped by the reach circle and the angle limit)."""
    z = cfg.zone
    xs, ys = np.linspace(z.x_mm[0], z.x_mm[1], n), np.linspace(z.y_mm[0], z.y_mm[1], n)
    edge = np.vstack([np.c_[xs, np.full(n, ys[0])], np.c_[np.full(n, xs[-1]), ys], np.c_[xs[::-1], np.full(n, ys[-1])],
                      np.c_[np.full(n, xs[0]), ys[::-1]]])
    r = np.hypot(edge[:, 0], edge[:, 1])
    edge = edge * np.minimum(1.0, z.max_reach_mm / np.maximum(r, 1e-9))[:, None]
    lim = np.radians(z.max_angle_deg)
    ang = np.clip(np.arctan2(edge[:, 1], np.maximum(edge[:, 0], 1e-9)), -lim, lim)
    return np.c_[np.hypot(edge[:, 0], edge[:, 1]) * np.cos(ang), np.hypot(edge[:, 0], edge[:, 1]) * np.sin(ang)]


def draw_ground_overlay(img, calib, cfg):
    g = lambda p: calib.project(np.c_[np.asarray(p, float).reshape(-1, 2), np.zeros(len(np.asarray(p).reshape(-1, 2)))])
    for x in range(-200, 601, 50):          # 50 mm grid over the sheet
        for y in range(-250, 251, 50):
            if cfg.zone.x_mm[0] - 150 <= x <= cfg.zone.x_mm[1] + 150:
                cv2.circle(img, tuple(np.round(g([x, y])[0]).astype(int)), 2, (0, 200, 0), -1)
    ref = reference_points(cfg)
    poly(img, g(ref[[0, 1, 3, 2]]), (255, 200, 0))
    poly(img, g(zone_polygon(cfg)), (0, 255, 255), 1)
    cv2.drawMarker(img, tuple(np.round(g([0, 0])[0]).astype(int)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    text(img, "arm base", tuple(np.round(g([0, 0])[0]).astype(int) + [10, -8]), (0, 0, 255))


def calibrate(grab, cfg, out_path, fov_deg):
    """Click UI: live video until SPACE freezes a frame, then 4 clicks. Returns the saved calibration or None."""
    ref = reference_points(cfg)
    frame = grab()
    if frame is None:
        return None
    h, w = frame.shape[:2]
    clicks, mouse = [], [0, 0]
    frozen = False
    win = "calibrate (space freeze/unfreeze, u undo, s save, q quit)"

    def on_mouse(event, x, y, flags, param):
        mouse[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN and frozen and len(clicks) < 4:
            clicks.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    calib = None
    while True:
        if not frozen:
            new = grab()
            if new is None:
                break
            if new.shape[:2] != (h, w):
                continue
            frame = new
        view = frame.copy()
        if not frozen:
            text(view, "LIVE: aim the camera so the whole sheet is in view, then press SPACE to freeze a frame",
                 (20, 40), (0, 255, 255), 0.6)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                frozen, clicks[:] = True, []
            continue
        for i, c in enumerate(clicks):
            cv2.drawMarker(view, c, (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
            text(view, str(i + 1), (c[0] + 8, c[1] - 8), (0, 0, 255))
        if len(clicks) == 4:
            try:
                calib = GroundCalibration.from_points(clicks, ref, (w, h), fov_deg)
            except ValueError as exc:
                calib = None
                text(view, str(exc), (20, 70), (0, 0, 255), 0.6)
            if calib is not None:
                draw_ground_overlay(view, calib, cfg)
                d = float(np.linalg.norm(calib.camera_center - np.array([0, 0, 0])))
                text(view, f"reproj {calib.reproj_rms:.1f}px  HFOV {calib.hfov_deg:.0f} deg ({calib.fov_source})  "
                           f"cam height {calib.cam_height:.0f} mm  cam {d:.0f} mm from the arm base", (20, 40), (0, 255, 0))
                text(view, "check the numbers with a tape measure and that the grid sits on the sheet; s = save", (20, 70))
        else:
            text(view, "click " + PROMPTS[len(clicks)], (20, 40), (0, 255, 255), 0.6)
            half = 20                                   # 4x zoom inset around the cursor, bottom right
            x0, y0 = max(0, min(w - 2 * half, mouse[0] - half)), max(0, min(h - 2 * half, mouse[1] - half))
            patch = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if patch.shape[0] == patch.shape[1] == 2 * half:
                zoom = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(zoom, (zoom.shape[1] // 2, zoom.shape[0] // 2), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                view[h - zoom.shape[0] - 10:h - 10, w - zoom.shape[1] - 10:w - 10] = zoom
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            frozen, calib = False, None
        if key == ord("u") and clicks:
            clicks.pop()
        if key == ord("s") and calib is not None:
            calib.save(out_path)
            print(f"saved {out_path}")
            break
    cv2.destroyWindow(win)
    return calib


def draw(frame, finder, state, calib, target, top=False):
    out = frame.copy()
    cfg = finder.cfg
    poly(out, calib.project(np.c_[zone_polygon(cfg), np.zeros(len(zone_polygon(cfg)))]), (0, 255, 255), 1)
    best = state.find(target) if target else None
    for c in state.containers:
        col = COLORS.get(c.color, (200, 200, 200))
        th = 3 if (best is not None and c.id == best.id) else 1
        ring = c.outline_mm
        poly(out, calib.project(np.c_[ring, np.zeros(len(ring))]), col, th)               # base on the mat
        poly(out, calib.project(np.c_[ring, np.full(len(ring), c.height_mm)]), col, th)  # rim
        tx, ty = np.round(calib.project(c.top_mm[None])[0]).astype(int)
        cv2.drawMarker(out, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        label = (f"#{c.id} {c.color} x={c.center_mm[0]:.0f} y={c.center_mm[1]:.0f} h={c.height_mm:.0f}"
                 f"{'' if c.height_source != 'default' else '?'}{'' if c.in_zone else ' (out of zone)'}")
        text(out, label, (int(tx) + 10, int(ty) + 16), col)
    if target:
        text(out, f"target {target}: " + (f"#{best.id} at x={best.center_mm[0]:.0f} y={best.center_mm[1]:.0f} mm"
                                          if best else "not found"), (20, 30), (0, 255, 0) if best else (0, 0, 255), 0.7)
    if top:
        cv2.imshow("top-down", cv2.resize(finder.top_view(frame), None, fx=0.5, fy=0.5))
    return out


def debug_view(finder, frame):
    """Top-down view with every colour blob outlined: green = became a container, red = dropped (with the reason)."""
    view = finder.top_view(frame)
    s = 2.0
    for b in finder.last_blobs:
        px = np.c_[(b["poly"][:, 0] - finder.x0) * s, (b["poly"][:, 1] - finder.y0) * s]
        ok = b["status"] == "kept"
        poly(view, px, (0, 255, 0) if ok else (0, 0, 255), 2)
        c = px.mean(0)
        text(view, f"{b['color']} {b['area']:.0f}mm2" + ("" if ok else f" - {b['status']}"),
             (int(c[0]) - 60, int(c[1])), (0, 255, 0) if ok else (80, 80, 255))
    zp = zone_polygon(finder.cfg)
    poly(view, np.c_[(zp[:, 0] - finder.x0) * s, (zp[:, 1] - finder.y0) * s], (0, 255, 255), 1)
    return cv2.resize(view, None, fx=0.5, fy=0.5)


def print_blobs(finder, last=[None]):
    rows = [f"{b['color']:<7}{b['area']:7.0f} mm2  at x={b['centroid'][0]:6.0f} y={b['centroid'][1]:6.0f}  {b['status']}"
            for b in finder.last_blobs]
    if rows != last[0]:
        last[0] = rows
        print("\nblobs:\n  " + ("\n  ".join(rows) if rows else "none"), flush=True)


def print_table(state, last=[None]):
    rows = [f"#{c.id:<3}{c.color:<8}{c.shape:<6}x={c.center_mm[0]:7.1f}  y={c.center_mm[1]:7.1f}  "
            f"size={c.size_mm[0]:.0f}x{c.size_mm[1]:.0f}  h={c.height_mm:.0f} ({c.height_source})"
            f"{'' if c.in_zone else '  out of zone'}" for c in state.containers]
    if rows != last[0]:
        last[0] = rows
        print("\n" + (f"{len(rows)} container(s):\n  " + "\n  ".join(rows) if rows else "no containers"), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, stream URL, video or image path")
    ap.add_argument("--calibrate", action="store_true", help="click the 4 reference points and save the calibration")
    ap.add_argument("--config", help="container_config.yaml")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--target", help="colour to look for, e.g. red")
    ap.add_argument("--json", action="store_true", help="print one JSON line per frame")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--debug", action="store_true", help="list every colour blob and why it was dropped; "
                                                       "shows a top-down debug window")
    ap.add_argument("--save", help="write the annotated frame here (still image sources / on 's')")
    ap.add_argument("--go", action="store_true", help="move above --target once it is stable, then exit")
    ap.add_argument("--port", help="arm serial port (e.g. /dev/cu.usbserial-10); omit to simulate on the MockArm")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--gripper-deg", type=float, default=90.0, help="gripper servo angle held during the move")
    ap.add_argument("--clearance-mm", type=float, default=DEFAULT_CLEARANCE_MM, help="height above the rim")
    args = ap.parse_args(argv)
    if args.go and not args.target:
        ap.error("--go needs --target")

    try:
        cfg = load_container_config(args.config)
    except ConfigError as exc:
        ap.exit(1, f"config error: {exc}\n")
    still = Path(args.source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    cap = None
    if not still:
        try:
            cap = open_camera(args.source)
        except RuntimeError as exc:
            ap.exit(1, f"{exc}\n")

    def grab():
        if still:
            return cv2.imread(args.source)
        ok, frame = cap.read()
        return frame if ok else None

    arm = []                                    # built on the first move, so viewing never opens the serial port
    status = [None]                             # last move result, drawn on the live view

    def go(state):
        try:
            if not arm:
                arm.append(make_arm(args.port, args.baud, args.gripper_deg))
            c, xyz = move_above_container(arm[0], state, args.target, args.clearance_mm)
        except Exception as exc:
            print(f"move failed: {exc}", file=sys.stderr, flush=True)
            status[0] = (f"move failed: {exc}", (0, 0, 255))
            return {"ok": False, "error": str(exc)}
        mm = " ".join(f"{v * 1000:.0f}" for v in xyz)
        print(f"moved above #{c.id} {c.color}: tip at {mm} mm", flush=True)
        status[0] = (f"moved above #{c.id} {c.color}: {mm} mm", (0, 255, 0))
        return {"ok": True, "id": c.id, "xyz_m": [round(v, 4) for v in xyz]}

    try:
        if args.calibrate:
            if args.no_gui:
                ap.exit(1, "--calibrate needs the GUI\n")
            if still:                           # a still image is its own frozen frame
                frame = grab()
                if frame is None:
                    ap.exit(1, f"cannot read image {args.source}\n")
                grab_cal = lambda: frame
            else:
                for _ in range(10):             # let auto-exposure settle
                    if grab() is None:
                        ap.exit(1, f"cannot read a frame from {args.source}\n")
                grab_cal = grab
            return 0 if calibrate(grab_cal, cfg, args.calibration, cfg.camera.fov_deg) else 1

        try:
            calib = GroundCalibration.load(args.calibration)
        except ValueError as exc:
            if args.json:
                print(json.dumps({"calibrated": False, "containers": []}))
            ap.exit(1, f"{exc}\nrun with --calibrate first\n")
        finder = ContainerFinder(cfg, calib)
        t0, fps = time.time(), 0.0
        top = False
        while True:
            frame = grab()
            if frame is None:
                if still:
                    ap.exit(1, f"cannot read image {args.source}\n")
                break
            try:
                state = finder.process(frame)
            except FrameSizeError as exc:
                print(f"skipping frame: {exc}", file=sys.stderr)
                if still:
                    return 1
                continue
            if still:                           # a single image never reaches min_age on its own
                for _ in range(max(cfg.tracking.min_age, 1) - 1):
                    state = finder.process(frame)
            if args.debug:
                print_blobs(finder)
            move = None
            if args.go and (still or state.find(args.target) is not None):
                move = go(state)
            if args.json:
                out = state.to_dict(args.target)
                if move is not None:
                    out["move"] = move
                print(json.dumps(out), flush=True)
            elif args.no_gui:
                print_table(state)
            if move is not None:
                return 0 if move["ok"] else 1
            if args.no_gui and not args.save:
                if still:
                    break
                continue
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t0, 1e-6) if fps else 1 / max(now - t0, 1e-6)
            t0 = now
            shown = draw(frame, finder, state, finder.calib_for(frame), args.target, top and not args.no_gui)
            text(shown, f"{fps:.0f} fps", (20, shown.shape[0] - 20), (0, 255, 0), 0.6)
            if status[0]:
                text(shown, status[0][0][:110], (20, 60), status[0][1], 0.6)
            if still and args.save:
                cv2.imwrite(args.save, shown)
                print(f"wrote {args.save}", file=sys.stderr)
                break
            if args.no_gui:
                continue
            cv2.imshow("containers", shown)
            if args.debug:
                cv2.imshow("debug: top-down blobs", debug_view(finder, frame))
            key = cv2.waitKey(0 if still else 1) & 0xFF
            if key == ord("q") or still:
                break
            if key == ord("g"):
                if args.target:
                    go(state)
                else:
                    status[0] = ("g needs --target", (0, 0, 255))
            elif key == ord("t"):
                top = not top
                if not top:
                    cv2.destroyWindow("top-down")
            elif key == ord("s"):
                cv2.imwrite(args.save or "containers.png", shown)
    except KeyboardInterrupt:
        pass
    finally:
        if arm:
            arm[0].close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
