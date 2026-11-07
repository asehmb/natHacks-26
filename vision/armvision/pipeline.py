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
from .mat import Frames, MarkerDetector, MatPose, solve_mat
from .tracker import Observation, Tracker


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

    def to_dict(self):
        return {"id": self.id, "color": self.color, "label": self.label, "center_mm": _r(self.center_mm),
                "grasp_mm": _r(self.grasp_mm), "grasp_yaw_deg": round(float(self.grasp_yaw_deg), 1),
                "size_mm": _r(self.size_mm), "occluded": self.occluded, "age": self.age}


@dataclass
class WorldState:
    mat_visible: bool = False
    arm: ArmState = field(default_factory=ArmState)
    objects: list[TrackedObject] = field(default_factory=list)

    def to_dict(self):
        return {"mat_visible": self.mat_visible, "arm": self.arm.to_dict(),
                "objects": [o.to_dict() for o in self.objects]}

    def find(self, color: str | None = None, position: str | None = None) -> list[TrackedObject]:
        """Objects matching a colour and/or a position word, best match first (default: nearest the base).
        Positions are in the base frame: left/right (y), front/far/back/near (x), center."""
        objs = [o for o in self.objects if color is None or o.color == color]
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
    def __init__(self, cfg: Config, intrinsics: Intrinsics | None = None):
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.detector = MarkerDetector()
        self.frames = Frames(cfg.arm)
        self.tracker = Tracker(cfg.vision.gate_mm, cfg.vision.max_missed, cfg.vision.ema)
        self.reference: np.ndarray | None = None
        self.pose: MatPose | None = None
        self.last_markers: dict[int, np.ndarray] = {}
        self.last_rect: np.ndarray | None = None
        self.last_fg: np.ndarray | None = None
        self.last_arm_mask: np.ndarray | None = None

    # -- setup -------------------------------------------------------------
    def _K(self, frame):
        if self.intrinsics is None:
            h, w = frame.shape[:2]
            self.intrinsics = fallback_intrinsics(w, h, self.cfg.vision.fov_deg)
        return self.intrinsics

    def _undistort(self, frame):
        intr = self._K(frame)
        return cv2.undistort(frame, intr.K, intr.dist) if intr.calibrated and np.any(intr.dist) else frame

    def update_pose(self, frame: np.ndarray):
        """Detect markers and refresh the mat pose. Returns (frame, detected_this_frame)."""
        frame = self._undistort(frame)
        self.last_markers = self.detector.detect(frame)
        pose = solve_mat(self.last_markers, self.cfg, self.intrinsics.K)
        if pose is not None:
            self.pose = pose
        return frame, pose is not None

    def capture_reference(self, frame: np.ndarray) -> bool:
        """Store the current rectified view as the empty-mat reference."""
        frame, ok = self.update_pose(frame)
        if self.pose is None:
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
        state = WorldState(mat_visible=detected)
        pose = self.pose
        if pose is None:
            return state
        rect = self.last_rect = pose.rectify(frame)
        s = v.px_per_mm

        # gripper: 3D from solvePnP, since it floats above the mat plane
        marker_px = None
        g = self.last_markers.get(cfg.arm.gripper_marker_id)
        if g is not None:
            res = armmod.marker_pose_cam(g, cfg.arm.gripper_marker_size_mm, self.intrinsics.K)
            if res is not None:
                p_mat = pose.cam_to_mat(res[1])
                xy = self.frames.mat_to_base(p_mat[:2])
                state.arm = ArmState(True, np.array([xy[0], xy[1], p_mat[2] + cfg.arm.mat_z_mm]))
                marker_px = pose.img_to_rect(g.mean(0))[0]

        if self.reference is None or self.reference.shape != rect.shape:
            return state
        fg = self.last_fg = objmod.foreground_mask(rect, self.reference, v)
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

        for t in self.tracker.update(obs, in_arm):
            o = t.obs
            cb = self.frames.mat_to_base(o.center)
            gb = self.frames.mat_to_base(o.grasp)
            state.objects.append(TrackedObject(
                t.id, o.color, "object", np.array([cb[0], cb[1], cfg.arm.mat_z_mm]),
                np.array([gb[0], gb[1], v.grasp_z_mm]), self.frames.yaw_to_base(o.yaw_deg),
                o.size, t.occluded, t.age))
        return state
