"""VisionSystem.process(frame) -> WorldState, all coordinates in the arm base frame (mm)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import arm as armmod
from . import objects as objmod
from .config import Config
from .grasp import compute_grasp
from .intrinsics import Intrinsics, fallback_intrinsics
from .mat import Frames, MarkerDetector, MatPose, marker_corners_mat, solve_mat
from .tracker import Observation, Tracker
from .yolo import DEFAULT_VOCAB, Box


class FrameSizeError(ValueError):
    """Frame shape doesn't match the calibration and can't be rescaled (e.g. a phone flipped to portrait)."""


def _r(v, n=1):
    return [round(float(x), n) for x in v]


@dataclass
class ArmState:
    visible: bool = False
    gripper_mm: np.ndarray | None = None

    def to_dict(self):
        return {"gripper_mm": None if self.gripper_mm is None else _r(self.gripper_mm), "visible": self.visible}


@dataclass
class TrackedObject:
    id: int
    color: str
    label: str
    center_mm: np.ndarray
    grasp_mm: np.ndarray
    grasp_yaw_deg: float
    size_mm: tuple[float, float]
    occluded: bool
    age: int
    seen: bool = True
    missed: int = 0
    confidence: float = 0.0   # 0..1: how sure we are this is a real, correctly labelled object

    def to_dict(self):
        return {"id": self.id, "color": self.color, "label": self.label, "center_mm": _r(self.center_mm),
                "grasp_mm": _r(self.grasp_mm), "grasp_yaw_deg": round(float(self.grasp_yaw_deg), 1),
                "size_mm": _r(self.size_mm), "occluded": self.occluded, "age": self.age,
                "seen": self.seen, "missed": self.missed,
                "confidence": round(float(self.confidence), 2)}


@dataclass
class WorldState:
    mat_visible: bool = False
    pose_age: int = 0
    arm: ArmState = field(default_factory=ArmState)
    objects: list[TrackedObject] = field(default_factory=list)

    @property
    def pose_fresh(self) -> bool:
        """True if the mat calibration was refreshed in this exact frame."""
        return self.mat_visible and self.pose_age == 0

    def to_dict(self):
        return {"mat_visible": self.mat_visible, "pose_age": self.pose_age,
                "pose_fresh": self.pose_fresh, "arm": self.arm.to_dict(),
                "objects": [o.to_dict() for o in self.objects]}

    def find(self, color: str | None = None, position: str | None = None,
             allow_stale: bool = False, allow_occluded: bool = False,
             label: str | None = None) -> list[TrackedObject]:
        """Objects matching a colour, a label (substring: "cup" finds "red cup") and/or a position word, best
        match first (default: nearest the base).
        Positions are in the base frame: left/right (y), front/far/back/near (x), center.
        Excludes stale (missed in current frame) and occluded objects by default."""
        objs = [o for o in self.objects
                if (color is None or o.color == color)
                and (label is None or label.lower() in o.label.lower())
                and (allow_stale or o.seen)
                and (allow_occluded or not o.occluded)]
        keys = {
            "left": lambda o: -o.center_mm[1], "right": lambda o: o.center_mm[1],
            "front": lambda o: -o.center_mm[0], "far": lambda o: -o.center_mm[0],
            "back": lambda o: o.center_mm[0], "near": lambda o: o.center_mm[0],
        }
        if position in ("center", "middle") and objs:
            mean = np.mean([o.center_mm[:2] for o in objs], 0)
            return sorted(objs, key=lambda o: float(np.linalg.norm(o.center_mm[:2] - mean)))
        if position is not None and position not in keys:
            raise ValueError(f"unknown position {position!r}")
        return sorted(objs, key=keys.get(position) or (lambda o: float(np.hypot(*o.center_mm[:2]))))


class VisionSystem:
    def __init__(self, cfg: Config, intrinsics: Intrinsics | None = None, labeler=None, vocab=DEFAULT_VOCAB,
                 mode: str = "diff", track_arm: bool = True, yolo_every: int = 5):
        """mode "diff": segment against an empty-mat reference, the optional labeler names the objects.
        mode "yolo": the labeler detects objects directly (no reference). track_arm: locate the gripper marker."""
        if mode not in ("diff", "yolo"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "yolo" and labeler is None:
            raise ValueError("mode 'yolo' needs a labeler")
        self.cfg = cfg
        self._intrinsics = intrinsics
        self.labeler, self.vocab, self.mode = labeler, list(vocab), mode
        self.track_arm, self.yolo_every = track_arm, max(1, yolo_every)
        self.last_boxes: list[Box] = []
        self.labels: dict[int, str] = {}
        self.label_scores: dict[int, float] = {}   # latest YOLO score behind each track's label, decays when unconfirmed
        self.votes: dict[int, dict[str, float]] = {}   # per track: decayed YOLO score summed per candidate label
        self._fresh_boxes = False
        self._frame: np.ndarray | None = None
        self._n = 0
        self.detector = MarkerDetector()
        self.frames = Frames(cfg.arm)
        self.tracker = Tracker(cfg.vision.gate_mm, cfg.vision.max_missed, cfg.vision.ema)
        self.reference: np.ndarray | None = None
        self.pose: MatPose | None = None
        self.pose_age: int = 0
        self.last_frame: np.ndarray | None = None   # undistorted frame the pose/markers refer to
        self.last_markers: dict[int, np.ndarray] = {}
        self.last_raw_markers: dict[int, np.ndarray] = {}
        self.last_rect: np.ndarray | None = None
        self.last_fg: np.ndarray | None = None
        self.last_arm_mask: np.ndarray | None = None

    @property
    def intrinsics(self) -> Intrinsics | None:
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, val: Intrinsics | None):
        self._intrinsics = val
        self.reference = None
        self.pose = None

    # -- setup -------------------------------------------------------------
    def _K(self, frame):
        h, w = frame.shape[:2]
        if self._intrinsics is None:
            self._intrinsics = fallback_intrinsics(w, h, self.cfg.vision.fov_deg)
        elif self._intrinsics.image_size != (w, h):
            w0, h0 = self._intrinsics.image_size
            if abs(w / h - w0 / h0) > 0.01 * (w0 / h0):
                raise FrameSizeError(
                    f"Frame {(w, h)} has a different aspect ratio than the calibration {(w0, h0)} (camera rotated "
                    f"or cropped?). Fix the camera orientation or recalibrate.")
            # same shape, different resolution: rescale K. The reference is resolution-independent, keep it.
            self._intrinsics = self._intrinsics.scaled((w, h))
            self.pose = None
        return self._intrinsics

    def adjust_intrinsics_for_frame(self, frame: np.ndarray) -> None:
        """Explicitly scale the current calibration intrinsics to match the given frame's resolution."""
        h, w = frame.shape[:2]
        if self._intrinsics is not None and self._intrinsics.image_size != (w, h):
            self.intrinsics = self._intrinsics.scaled((w, h))

    def _undistort(self, frame):
        intr = self._K(frame)
        return cv2.undistort(frame, intr.K, intr.dist) if intr.calibrated and np.any(intr.dist) else frame

    def update_pose(self, frame: np.ndarray):
        """Detect markers and refresh the mat pose. Returns (frame, detected_this_frame)."""
        intr = self._K(frame)
        self.last_raw_markers = self.detector.detect(frame)
        if intr.calibrated and np.any(intr.dist):
            undistorted = cv2.undistort(frame, intr.K, intr.dist)
            self.last_markers = self.detector.detect(undistorted)
        else:
            undistorted = frame
            self.last_markers = self.last_raw_markers

        self.last_frame = undistorted
        pose = solve_mat(self.last_markers, self.cfg, self.intrinsics.K)
        if pose is not None:
            self.pose = pose
            self.pose_age = 0
        else:
            self.pose_age += 1
            max_age = getattr(self.cfg.vision, "max_pose_age", 5)
            if self.pose_age > max_age:
                self.pose = None
        return undistorted, pose is not None

    def capture_reference(self, frame: np.ndarray) -> bool:
        """Store the current rectified view as the empty-mat reference."""
        frame, ok = self.update_pose(frame)
        if not ok or self.pose is None:
            return False
        self.set_reference(self.pose.rectify(frame))
        self.tracker.reset()
        return True

    def set_reference(self, rect: np.ndarray):
        self.reference = rect.copy()

    # -- per frame ---------------------------------------------------------
    def process(self, frame: np.ndarray) -> WorldState:
        cfg, v = self.cfg, self.cfg.vision
        frame, detected = self.update_pose(frame)
        state = WorldState(mat_visible=detected, pose_age=self.pose_age)
        pose = self.pose
        if pose is None:
            return state
        rect = self.last_rect = pose.rectify(frame)
        s = v.px_per_mm

        # gripper: 3D from solvePnP, since it floats above the mat plane
        marker_px = None
        g = self.last_markers.get(cfg.arm.gripper_marker_id) if self.track_arm else None
        if g is not None:
            res = armmod.marker_pose_cam(g, cfg.arm.gripper_marker_size_mm, self.intrinsics.K)
            if res is not None:
                p_mat = pose.cam_to_mat(res[1])
                xy = self.frames.mat_to_base(p_mat[:2])
                state.arm = ArmState(True, np.array([xy[0], xy[1], p_mat[2] + cfg.arm.mat_z_mm]))
                marker_px = pose.img_to_rect(g.mean(0))[0]

        if self.labeler is not None:
            self._frame = frame
            self._fresh_boxes = self._n % self.yolo_every == 0
            if self._fresh_boxes:
                self.last_boxes = self.labeler.detect(frame, self.vocab)
            self._n += 1

        if self.mode == "yolo":
            return self._process_yolo(frame, pose, state)
        if self.reference is None or self.reference.shape != rect.shape:
            return state
        fg = self.last_fg = objmod.foreground_mask(rect, self.reference, v, self._marker_mask(pose, rect.shape))
        base_px = pose.mat_to_rect(cfg.arm.base_xy_mm)[0]
        amask = self.last_arm_mask = armmod.arm_mask(
            fg, base_px, cfg.arm.base_zone_radius_mm * s, marker_px, 0.5 * cfg.arm.gripper_marker_size_mm * s)
        obj_mask = cv2.bitwise_and(fg, cv2.bitwise_not(amask))

        base = np.array(cfg.arm.base_xy_mm)
        obs = []
        for d in objmod.extract_detections(rect, obj_mask, v, cfg.mat.height_mm):
            poly = objmod.near_half(d.polygon_mm, pose.cam_center[:2]) if v.use_near_half else d.polygon_mm
            gr = compute_grasp(poly, base, v.grasp_inset_mm, v.round_aspect)
            obs.append(Observation(gr.center, gr.point, gr.yaw_deg, (gr.length, gr.width), d.color))

        def in_arm(p_mat):
            c, r = pose.mat_to_rect(p_mat)[0]
            r_i, c_i = int(round(r)), int(round(c))
            h, w = amask.shape
            return 0 <= r_i < h and 0 <= c_i < w and amask[r_i, c_i] > 0

        return self._finish(state, pose, obs, in_arm, self._label_by_containment)

    def _process_yolo(self, frame, pose, state):
        cfg = self.cfg
        obs, boxes = [], []
        for b in self.last_boxes:
            x1, y1, x2, y2 = b.xyxy
            left, right, bottom = pose.img_to_mat([(x1, y2), (x2, y2), b.bottom_center])
            if not (0 <= bottom[0] <= cfg.mat.width_mm and 0 <= bottom[1] <= cfg.mat.height_mm):
                continue
            w = float(np.linalg.norm(right - left))
            crop = frame[int(y1):int(y2), int(x1):int(x2)].reshape(-1, 3)
            obs.append(Observation(bottom, bottom.copy(), 0.0, (w, w), objmod.color_name(crop)))
            boxes.append(b)

        def label_by_footprint(tracks):
            for t in tracks:
                d = [float(np.linalg.norm(t.obs.center - o.center)) for o in obs]
                if d and min(d) <= cfg.vision.gate_mm:
                    b = boxes[int(np.argmin(d))]
                    self._vote(t.id, b.label, b.score)
                else:
                    self._vote(t.id, None, 0.0)
        return self._finish(state, pose, obs, lambda p: False, label_by_footprint)

    def _marker_mask(self, pose, shape) -> np.ndarray:
        """Rectified-view mask of the printed mat markers (known layout), grown by marker_margin_mm."""
        mask = np.zeros(shape[:2], np.uint8)
        for centre in self.cfg.mat.markers.values():
            quad = marker_corners_mat(centre, self.cfg.mat.marker_size_mm)[:, :2]
            cv2.fillConvexPoly(mask, pose.mat_to_rect(quad).astype(np.int32), 255)
        grow = round(2 * self.cfg.vision.marker_margin_mm * self.cfg.vision.px_per_mm) | 1
        return cv2.dilate(mask, np.ones((grow, grow), np.uint8))

    def _confidence(self, t) -> float:
        """0..1. Track stability (age, recent misses) plus the YOLO label score; unlabelled objects cap at 0.4."""
        stab = min(t.age, 10) / 10
        if not t.seen:
            stab *= max(0.0, 1 - t.missed / max(1, self.cfg.vision.max_missed))
        if t.occluded:
            stab *= 0.5
        return round(0.4 * stab + 0.6 * self.label_scores.get(t.id, 0.0) * (1.0 if t.seen else 0.5), 3)

    def _vote(self, tid: int, label: str | None, score: float):
        """Accumulate evidence for a track's label over YOLO runs, so one frame (a shadow) can't flip it.
        The shown label is the vote leader; its score is the latest score scaled by the leader's share of all
        votes, so a track torn between "scissors" and "screwdriver" reads as low confidence."""
        if not (self._fresh_boxes or tid not in self.labels):
            return                                    # no new YOLO output since the last vote
        v = {k: x * 0.85 for k, x in self.votes.get(tid, {}).items()}
        if label is not None:
            v[label] = v.get(label, 0.0) + score
        self.votes[tid] = v
        if not v:
            return
        best = max(v, key=v.get)
        total = sum(v.values())
        if total <= 1e-6:
            return
        self.labels[tid] = best
        self.label_scores[tid] = (score if label == best else self.label_scores.get(tid, 0.0) * 0.8) * v[best] / total

    def _label_by_containment(self, tracks, pose):
        for t in tracks:
            x, y = pose.mat_to_img(t.obs.center)[0]
            hits = [b for b in self.last_boxes if b.xyxy[0] <= x <= b.xyxy[2] and b.xyxy[1] <= y <= b.xyxy[3]]
            if hits:
                b = max(hits, key=lambda b: b.score)
                self._vote(t.id, b.label, b.score)
            else:
                # no detector box on this object (it missed, or the box landed elsewhere): ask CLIP about its footprint
                guess = self._clip_guess(t, pose) if self._fresh_boxes and t.seen else None
                self._vote(t.id, *(guess if guess else (None, 0.0)))

    def _clip_guess(self, t, pose):
        classify = getattr(self.labeler, "classify_crop", None)
        if classify is None or self._frame is None:
            return None
        o, yaw = t.obs, np.radians(t.obs.yaw_deg - 90.0)
        d, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        L, W = o.size
        quad = [o.center + d * L / 2 + n * W / 2, o.center + d * L / 2 - n * W / 2,
                o.center - d * L / 2 - n * W / 2, o.center - d * L / 2 + n * W / 2]
        px = pose.mat_to_img(np.array(quad))
        res = classify(self._frame, (*px.min(0), *px.max(0)), self.vocab)
        return (res[0], res[1] * 0.8) if res and res[1] >= getattr(self.labeler, "clip_min", 0.35) else None

    def _finish(self, state, pose, obs, in_arm, labeler):
        cfg, v = self.cfg, self.cfg.vision
        tracks = self.tracker.update(obs, in_arm)
        if self.labeler is not None:
            if labeler == self._label_by_containment:
                labeler(tracks, pose)
            else:
                labeler(tracks)
        live = {t.id for t in tracks}
        self.labels = {i: l for i, l in self.labels.items() if i in live}
        self.label_scores = {i: x for i, x in self.label_scores.items() if i in live}
        self.votes = {i: x for i, x in self.votes.items() if i in live}
        for t in tracks:
            if t.age < v.min_age:      # flicker (shadows, glare, a hand passing) never lasts
                continue
            o = t.obs
            cb = self.frames.mat_to_base(o.center)
            gb = self.frames.mat_to_base(o.grasp)
            state.objects.append(TrackedObject(
                t.id, o.color, self.labels.get(t.id, "object"), np.array([cb[0], cb[1], cfg.arm.mat_z_mm]),
                np.array([gb[0], gb[1], v.grasp_z_mm]), self.frames.yaw_to_base(o.yaw_deg),
                o.size, t.occluded, t.age, seen=t.seen, missed=t.missed, confidence=self._confidence(t)))
        return state
