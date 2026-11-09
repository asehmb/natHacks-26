"""Marker-free finder for coloured containers on the white sheet: centre, footprint and height in the arm base frame.

A tall container seen from a tilted camera leans away from the camera, so the centroid of its blob is far from its
centre. Here the image is warped to a top-down view of the ground plane (needs `groundcal.GroundCalibration`), the
blob outline is split into rim points (top edge, pushed outward from the point N under the camera by
h_cam / (h_cam - H)) and base points (bottom edge, on the mat), and the footprint is fitted after pulling the rim back.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import ConfigError, _pair, _positive, _section
from .grasp import min_area_rect
from .groundcal import GroundCalibration
from .objects import CHROMATIC
from .planner_bridge import normalize_color
from .tracker import Observation, Tracker

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "container_config.yaml"

PX_PER_MM = 2.0
VIEW_PAD_MM = 200.0         # top-down view extends this far beyond the zone (room for leaning tops)
RIM_DOT, BASE_DOT, BASE_ARC_DOT = 0.25, -0.5, -0.8
ROUND_RMS = 0.05            # rim circle rms / radius below this -> round
ROUND_MIN_ARC = 0.6         # ... and the rim points must span an arc (chord / 2r >= this, ~74 deg), not a straight edge
MIN_AUTO_H = 25.0
MAX_AUTO_H_FRAC = 0.8       # of the camera height
HEIGHT_REFRESH_FRAMES = 15  # an automatic height is re-estimated from scratch this often
BOX_FIT_RMS_MM = 5.0        # worst acceptable silhouette mismatch for an automatic box height


# ----------------------------------------------------------------------------------------------- config
@dataclass
class SheetConfig:
    width_mm: float = 430.0
    length_mm: float = 400.0             # arm-end edge to the camera-end edge
    mark_distance_mm: float = 250.0      # the 2 pen ticks on the long edges, from the arm-end edge
    points: list | None = None


@dataclass
class ArmPlacement:
    base_from_edge_mm: float = 100.0     # base axis to the sheet's arm-end edge, along the sheet
    base_offset_mm: float = 0.0          # base axis to the left of the sheet centre line
    base_yaw_deg: float = 0.0            # sheet's long axis vs arm forward, counter-clockwise


@dataclass
class ZoneConfig:
    x_mm: tuple = (150.0, 330.0)
    y_mm: tuple = (-160.0, 160.0)
    max_reach_mm: float = 330.0
    max_angle_deg: float = 30.0


@dataclass
class ContainersConfig:
    heights_mm: dict = field(default_factory=dict)
    default_height_mm: float = 80.0
    max_height_mm: float = 200.0
    min_footprint_mm2: float = 2500.0
    colors: list | None = None           # only look for these colours (None = every colour in objects.CHROMATIC)
    min_saturation: float = 0.6          # container plastic is vividly coloured; shadows and tinted glow are not
    min_value: float = 0.25
    hue_ranges: dict = field(default_factory=dict)   # colour -> [[lo, hi], ...] degrees, overrides objects.CHROMATIC
    sheet_margin_mm: float | None = 60.0  # ignore colour outside the sheet outline + this margin (table, hands); None = off


@dataclass
class CameraConfig:
    fov_deg: float | None = None


@dataclass
class TrackingConfig:
    gate_mm: float = 30.0
    ema: float = 0.3
    max_missed: int = 10
    min_age: int = 3


@dataclass
class ContainerConfig:
    sheet: SheetConfig = field(default_factory=SheetConfig)
    arm: ArmPlacement = field(default_factory=ArmPlacement)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    containers: ContainersConfig = field(default_factory=ContainersConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name}: must be a finite number, got {value!r}")


def _range(value, name):
    lo, hi = _pair(value, name)
    if lo >= hi:
        raise ConfigError(f"{name}: expected [min, max] with min < max, got {value!r}")
    return lo, hi


def validate_container_config(c: ContainerConfig) -> ContainerConfig:
    s, a, z, k, cam, t = c.sheet, c.arm, c.zone, c.containers, c.camera, c.tracking
    _positive(s.width_mm, "sheet.width_mm")
    _positive(s.length_mm, "sheet.length_mm")
    _positive(s.mark_distance_mm, "sheet.mark_distance_mm")
    if s.mark_distance_mm >= s.length_mm:
        raise ConfigError("sheet.mark_distance_mm: the ticks must be closer to the arm than the camera-end edge "
                          "(smaller than sheet.length_mm)")
    if s.points is not None:
        if not isinstance(s.points, (list, tuple)) or len(s.points) != 4:
            raise ConfigError("sheet.points: expected 4 [x, y] points (far-left corner, far-right corner, left tick, right tick)")
        s.points = [list(_pair(p, f"sheet.points[{i}]")) for i, p in enumerate(s.points)]
    for v, n in [(a.base_from_edge_mm, "arm.base_from_edge_mm"), (a.base_offset_mm, "arm.base_offset_mm"),
                 (a.base_yaw_deg, "arm.base_yaw_deg")]:
        _finite(v, n)
    z.x_mm, z.y_mm = _range(z.x_mm, "zone.x_mm"), _range(z.y_mm, "zone.y_mm")
    _positive(z.max_reach_mm, "zone.max_reach_mm")
    _finite(z.max_angle_deg, "zone.max_angle_deg")
    if not (0 < z.max_angle_deg <= 180):
        raise ConfigError("zone.max_angle_deg: must be in (0, 180]")
    if not isinstance(k.heights_mm, dict):
        raise ConfigError("containers.heights_mm: expected a mapping colour -> mm")
    heights = {}
    for col, h in k.heights_mm.items():
        _positive(h, f"containers.heights_mm[{col}]")
        heights[normalize_color(str(col))] = float(h)
    k.heights_mm = heights
    _positive(k.default_height_mm, "containers.default_height_mm")
    _positive(k.max_height_mm, "containers.max_height_mm")
    _positive(k.min_footprint_mm2, "containers.min_footprint_mm2")
    if k.colors is not None:
        if not isinstance(k.colors, (list, tuple)) or not k.colors:
            raise ConfigError("containers.colors: expected a non-empty list of colour names, or null")
        names = [normalize_color(str(n)) for n in k.colors]
        bad = [n for n in names if n not in CHROMATIC]
        if bad:
            raise ConfigError(f"containers.colors: unknown colours {bad}; known: {sorted(CHROMATIC)}")
        k.colors = names
    for n, v in (("min_saturation", k.min_saturation), ("min_value", k.min_value)):
        _finite(v, f"containers.{n}")
        if not 0 <= v <= 1:
            raise ConfigError(f"containers.{n}: must be between 0 and 1, got {v!r}")
    if not isinstance(k.hue_ranges, dict):
        raise ConfigError("containers.hue_ranges: expected a mapping colour -> [[lo, hi], ...] in degrees")
    ranges = {}
    for col, rs in k.hue_ranges.items():
        name = normalize_color(str(col))
        if name not in CHROMATIC:
            raise ConfigError(f"containers.hue_ranges: unknown colour {col!r}; known: {sorted(CHROMATIC)}")
        if not isinstance(rs, (list, tuple)) or not rs:
            raise ConfigError(f"containers.hue_ranges[{col}]: expected [[lo, hi], ...]")
        ranges[name] = [tuple(_pair(r, f"containers.hue_ranges[{col}]")) for r in rs]
        if any(not (0 <= lo <= hi <= 360) for lo, hi in ranges[name]):
            raise ConfigError(f"containers.hue_ranges[{col}]: need 0 <= lo <= hi <= 360 degrees")
    k.hue_ranges = ranges
    if k.sheet_margin_mm is not None:
        _finite(k.sheet_margin_mm, "containers.sheet_margin_mm")
        if k.sheet_margin_mm < 0:
            raise ConfigError("containers.sheet_margin_mm: must be non-negative (or null)")
    if k.default_height_mm > k.max_height_mm:
        raise ConfigError("containers.default_height_mm must not exceed containers.max_height_mm")
    if cam.fov_deg is not None:
        _finite(cam.fov_deg, "camera.fov_deg")
        if not (0 < cam.fov_deg < 180):
            raise ConfigError("camera.fov_deg: must be between 0 and 180 degrees")
    _positive(t.gate_mm, "tracking.gate_mm")
    _finite(t.ema, "tracking.ema")
    if not (0 < t.ema <= 1):
        raise ConfigError("tracking.ema: must be in (0, 1]")
    for v, n in [(t.max_missed, "tracking.max_missed"), (t.min_age, "tracking.min_age")]:
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ConfigError(f"{n}: must be a non-negative integer, got {v!r}")
    return c


def load_container_config(path: str | Path | None = None) -> ContainerConfig:
    path = Path(path) if path else DEFAULT_PATH
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parts = {f.name: f.type for f in fields(ContainerConfig)}
    unknown = set(raw) - set(parts)
    if unknown:
        raise ConfigError(f"unknown top-level keys {sorted(unknown)}")
    classes = {"sheet": SheetConfig, "arm": ArmPlacement, "zone": ZoneConfig, "containers": ContainersConfig,
               "camera": CameraConfig, "tracking": TrackingConfig}
    return validate_container_config(ContainerConfig(**{n: _section(c, raw.get(n), n) for n, c in classes.items()}))


# ----------------------------------------------------------------------------------------------- geometry
def circle_fit(pts: np.ndarray, iters: int = 10):
    """Least-squares circle: algebraic fit refined by Gauss-Newton. Returns (centre, radius, rms residual)."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) < 3:
        raise ValueError("need at least 3 points")
    m = p.mean(0)
    q = p - m
    A = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    c, r = sol[:2].copy(), math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-12))
    for _ in range(iters):
        d = q - c
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        J = np.c_[-d[:, 0] / dist, -d[:, 1] / dist, -np.ones(len(q))]
        step, *_ = np.linalg.lstsq(J, -(dist - r), rcond=None)
        c, r = c + step[:2], r + step[2]
        if np.abs(step).max() < 1e-6:
            break
    res = np.linalg.norm(q - c, axis=1) - r
    return c + m, float(abs(r)), float(np.sqrt(np.mean(res ** 2)))


def resample_outline(poly: np.ndarray, step_mm: float = 2.0, smooth: int = 4) -> np.ndarray:
    """Closed outline resampled to uniform spacing and lightly smoothed (removes pixel stair-steps)."""
    p = np.asarray(poly, np.float64)
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 3 * step_mm:
        return p
    n = max(int(s[-1] / step_mm), 8)
    t = np.linspace(0, s[-1], n, endpoint=False)
    out = np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]
    if smooth > 0:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        out = np.stack([np.convolve(np.r_[out[-smooth:, i], out[:, i], out[:smooth, i]], k, "valid")
                        for i in range(2)], 1)
    return out


def split_outline(poly: np.ndarray, nadir: np.ndarray):
    """Returns (points, dots): dots = outward contour normal . direction away from the camera nadir."""
    pts = resample_outline(poly)
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    tan = np.roll(pts, -2, axis=0) - np.roll(pts, 2, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    normal = np.c_[tan[:, 1], -tan[:, 0]]      # outward for a counter-clockwise polygon
    if area < 0:
        normal = -normal
    away = pts - nadir
    away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
    return pts, np.sum(normal * away, 1)


def _pull(pts, nadir, H, h_cam):
    """Ground position of points on the rim (height H) seen at pts: move them back towards the nadir."""
    return nadir + (pts - nadir) * (1.0 - H / h_cam)


def _rect_corners(center, a, b, axis_deg):
    t = math.radians(axis_deg)
    u, v = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    return np.array([center + sx * a / 2 * u + sy * b / 2 * v for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def _box_fit(base, rim, nadir, H, h_cam, step_deg=0.5):
    pts = np.vstack([base, _pull(rim, nadir, H, h_cam)])
    return min_area_rect(pts, step_deg)


def _silhouette_error(pts, rect, nadir, H, h_cam) -> float:
    """rms mismatch between the observed outline and the silhouette a box (centre, a, b, axis) of height H would
    cast (convex hull of its footprint and its rim pushed outward)."""
    c, a, b, ax = rect
    corners = _rect_corners(c, a, b, ax)
    hull = cv2.convexHull(np.vstack([corners, nadir + (corners - nadir) / (1.0 - H / h_cam)]).astype(np.float32))
    hull = hull.reshape(-1, 2).astype(np.float64)
    edge = np.vstack([hull, hull[:1]])
    samples = np.vstack([np.linspace(edge[i], edge[i + 1], 12, endpoint=False) for i in range(len(hull))])
    d1 = np.sqrt(((pts[:, None] - samples[None]) ** 2).sum(2)).min(1)
    d2 = np.sqrt(((samples[:, None] - pts[None]) ** 2).sum(2)).min(1)
    return float(math.sqrt((np.mean(d1 ** 2) + np.mean(d2 ** 2)) / 2))


@dataclass
class Fit:
    center: np.ndarray            # footprint centre, arm frame mm
    size: tuple[float, float]
    shape: str                    # "round" | "box"
    yaw_deg: float
    height: float
    height_source: str            # "config" | "auto" | "default"
    outline: np.ndarray           # footprint polygon (closed ring), arm frame mm


def fit_container(poly: np.ndarray, nadir, h_cam: float, height_mm: float | None = None,
                  default_height: float = 80.0, max_height: float = 200.0,
                  height_source: str = "config") -> Fit | None:
    """Footprint, size and height of one container from its blob outline (arm frame mm)."""
    nadir = np.asarray(nadir, np.float64)
    pts, dots = split_outline(poly, nadir)
    rim, base = pts[dots > RIM_DOT], pts[dots < BASE_DOT]
    if len(rim) < 6 or len(base) < 6:
        return None
    h_max = min(max_height, MAX_AUTO_H_FRAC * h_cam)
    if height_mm is not None and height_mm >= 0.95 * h_cam:
        return None                      # camera is not above the rim: the configured height is impossible

    crim, rrim, rms = circle_fit(rim)
    chord = float(np.ptp(rim @ np.linalg.svd(rim - rim.mean(0), full_matrices=False)[2][0]))
    is_round = rms / rrim < ROUND_RMS and chord / (2 * rrim) >= ROUND_MIN_ARC
    source, H = height_source, height_mm
    if is_round:
        if H is None:
            H, source = None, "default"
            arc = pts[dots < BASE_ARC_DOT]
            if len(arc) >= 8:
                cb, rb, _ = circle_fit(arc)
                inl = np.mean(np.abs(np.linalg.norm(arc - cb, axis=1) - rb) < max(2.5, 0.06 * rb))
                dn = cb - nadir
                k = float((crim - nadir) @ dn / max(dn @ dn, 1e-9))
                if inl >= 0.8 and k > 1.0:
                    est = h_cam * (1.0 - 1.0 / k)
                    if MIN_AUTO_H <= est <= h_max:
                        H, source = est, "auto"
            if H is None:
                H = default_height
        c = _pull(crim, nadir, H, h_cam)
        r = rrim * (1.0 - H / h_cam)
        ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        ring = c + r * np.c_[np.cos(ang), np.sin(ang)]
        return Fit(c, (2 * r, 2 * r), "round", 0.0, float(H), source, ring)

    if H is None:
        source = "default"
        grid = np.arange(MIN_AUTO_H, h_max + 1e-9, 5.0)
        if len(grid) >= 3:
            sub = pts[::max(1, len(pts) // 80)]          # the search only needs a sparse outline
            err = lambda h: _silhouette_error(sub, _box_fit(base[::2], rim[::2], nadir, h, h_cam, 1.0), nadir, h, h_cam)
            cost = [err(h) for h in grid]
            h0 = float(grid[int(np.argmin(cost))])
            fine = np.arange(max(MIN_AUTO_H, h0 - 5), min(h_max, h0 + 5) + 1e-9, 1.0)
            fc = [err(h) for h in fine]
            best = int(np.argmin(fc))
            if fc[best] < BOX_FIT_RMS_MM and MIN_AUTO_H < fine[best] < h_max:
                H, source = float(fine[best]), "auto"
        if H is None:
            H = default_height
    c, a, b, ax = _box_fit(base, rim, nadir, H, h_cam)
    return Fit(np.asarray(c), (a, b), "box", float(ax), float(H), source, _rect_corners(c, a, b, ax))


# ----------------------------------------------------------------------------------------------- finder
@dataclass
class Container:
    id: int
    color: str
    center_mm: np.ndarray
    height_mm: float
    size_mm: tuple[float, float]
    shape: str
    yaw_deg: float
    height_source: str
    in_zone: bool
    age: int
    seen: bool = True
    outline_mm: np.ndarray | None = None

    @property
    def top_mm(self) -> np.ndarray:
        return np.array([self.center_mm[0], self.center_mm[1], self.height_mm])

    def to_dict(self):
        r = lambda v, n=1: [round(float(x), n) for x in v]
        return {"id": self.id, "color": self.color, "center_mm": r(self.center_mm), "top_mm": r(self.top_mm),
                "size_mm": r(self.size_mm), "shape": self.shape, "yaw_deg": round(float(self.yaw_deg), 1),
                "height_mm": round(float(self.height_mm), 1), "height_source": self.height_source,
                "in_zone": self.in_zone, "age": self.age, "seen": self.seen}


@dataclass
class ContainerState:
    containers: list[Container] = field(default_factory=list)
    calibrated: bool = True

    def find(self, color: str | None = None) -> Container | None:
        """Best container of that colour: in the zone, seen this frame and stable; nearest the arm first."""
        want = normalize_color(color)
        ok = [c for c in self.containers if c.in_zone and c.seen and (want is None or c.color == want)]
        return min(ok, key=lambda c: float(np.hypot(*c.center_mm)), default=None)

    def to_dict(self, target: str | None = None):
        out = {"calibrated": self.calibrated, "containers": [c.to_dict() for c in self.containers]}
        if target is not None:
            t = self.find(target)
            out["target"] = None if t is None else t.to_dict()
        return out


class ContainerFinder:
    def __init__(self, cfg: ContainerConfig, calib: GroundCalibration):
        self.cfg = cfg
        self.calib = calib
        self.tracker = Tracker(cfg.tracking.gate_mm, cfg.tracking.max_missed, cfg.tracking.ema)
        self._h: dict[int, float] = {}
        self._meta: dict[int, Fit] = {}
        self._view_cache: dict[tuple, tuple] = {}
        self._calibs: dict[tuple, GroundCalibration] = {}
        self._blob_h: list[tuple] = []                 # (colour, blob centroid, height) of the last frame
        self.last_blobs: list[dict] = []               # every blob seen last frame and why it was kept or dropped
        z = cfg.zone
        self.x0, self.x1 = z.x_mm[0] - VIEW_PAD_MM, z.x_mm[1] + VIEW_PAD_MM
        self.y0, self.y1 = z.y_mm[0] - VIEW_PAD_MM, z.y_mm[1] + VIEW_PAD_MM
        self.view_size = (round((self.x1 - self.x0) * PX_PER_MM), round((self.y1 - self.y0) * PX_PER_MM))
        self._allowed = self._sheet_mask()

    def _sheet_mask(self):
        """View mask (255 = may hold a container): the sheet outline plus a margin. None if disabled."""
        m = self.cfg.containers.sheet_margin_mm
        if m is None or self.cfg.sheet.points is not None:
            return None
        s, a = self.cfg.sheet, self.cfg.arm
        far_x = -a.base_from_edge_mm + s.length_mm                  # rectangle in the sheet's own axes, then rotated
        x0 = -a.base_from_edge_mm - m
        hw, yc = s.width_mm / 2 + m, -a.base_offset_mm
        quad = np.array([[x0, yc - hw], [far_x + m, yc - hw], [far_x + m, yc + hw], [x0, yc + hw]])
        t = math.radians(a.base_yaw_deg)
        quad = quad @ np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]).T
        px = np.c_[(quad[:, 0] - self.x0) * PX_PER_MM, (quad[:, 1] - self.y0) * PX_PER_MM]
        mask = np.zeros((self.view_size[1], self.view_size[0]), np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 255)
        return mask

    def _view(self, calib: GroundCalibration):
        """Image -> top-down view matrix and a mask of the view pixels that are unusable (outside the picture,
        behind the camera, or on the view border)."""
        key = calib.size
        if key in self._view_cache:
            return self._view_cache[key]
        s = PX_PER_MM
        A = np.array([[s, 0, -s * self.x0], [0, s, -s * self.y0], [0, 0, 1.0]])
        M = A @ np.linalg.inv(calib.H_base2img)
        w, h = self.view_size
        gx, gy = np.meshgrid(self.x0 + (np.arange(w) + 0.5) / s, self.y0 + (np.arange(h) + 0.5) / s, indexing="ij")
        depth = calib.R[2, 0] * gx + calib.R[2, 1] * gy + calib.t[2]
        ok = (depth > 1.0).T.astype(np.uint8) * 255
        inside = cv2.warpPerspective(np.full((calib.size[1], calib.size[0]), 255, np.uint8), M, (w, h))
        ok = cv2.bitwise_and(ok, (inside > 250).astype(np.uint8) * 255)
        bad = cv2.bitwise_not(ok)
        bad[:2], bad[-2:], bad[:, :2], bad[:, -2:] = 255, 255, 255, 255
        self._view_cache[key] = (M, bad, cv2.dilate(bad, np.ones((5, 5), np.uint8)))
        return self._view_cache[key]

    def calib_for(self, frame: np.ndarray) -> GroundCalibration:
        """The calibration rescaled to this frame's size (cached; FrameSizeError if the aspect ratio differs)."""
        size = (frame.shape[1], frame.shape[0])
        if size not in self._calibs:
            self._calibs[size] = self.calib.rescaled(size)
        return self._calibs[size]

    def top_view(self, frame: np.ndarray):
        calib = self.calib_for(frame)
        M, bad, _ = self._view(calib)
        view = cv2.warpPerspective(frame, M, self.view_size)
        view[bad > 0] = 0
        return view

    def in_zone(self, xy) -> bool:
        z, (x, y) = self.cfg.zone, xy
        return bool(z.x_mm[0] <= x <= z.x_mm[1] and z.y_mm[0] <= y <= z.y_mm[1]
                    and math.hypot(x, y) <= z.max_reach_mm
                    and x > 0 and abs(math.degrees(math.atan2(y, x))) <= z.max_angle_deg)

    def detect(self, frame: np.ndarray):
        """Per-frame detections: list of (colour, Fit). `last_blobs` records every blob and why it was dropped."""
        calib = self.calib_for(frame)
        M, bad, bad_wide = self._view(calib)
        cc = self.cfg.containers
        view = cv2.warpPerspective(frame, M, self.view_size)
        hsv = cv2.cvtColor(cv2.GaussianBlur(view, (5, 5), 0), cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0].astype(np.float32) * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        usable = (sat >= cc.min_saturation) & (val >= cc.min_value) & (bad == 0)
        if self._allowed is not None:
            usable &= self._allowed > 0
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out, blobs, new_h = [], [], []
        for color, ranges in CHROMATIC.items():
            if cc.colors is not None and color not in cc.colors:
                continue
            in_hue = np.zeros(hue.shape, bool)
            for lo, hi in cc.hue_ranges.get(color, ranges):
                in_hue |= (hue >= lo) & (hue <= hi)
            mask = ((in_hue & usable).astype(np.uint8)) * 255
            mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern), cv2.MORPH_CLOSE, kern)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                area = cv2.contourArea(c) / PX_PER_MM ** 2
                p = cv2.convexHull(c).reshape(-1, 2).astype(np.float64)   # a container's silhouette is convex
                poly = np.c_[self.x0 + p[:, 0] / PX_PER_MM, self.y0 + p[:, 1] / PX_PER_MM]
                rec = {"color": color, "poly": poly, "area": area, "centroid": poly.mean(0), "status": "kept"}
                if area < 0.2 * cc.min_footprint_mm2:
                    continue                                  # speckle: not worth listing
                blobs.append(rec)
                if area < cc.min_footprint_mm2:
                    rec["status"] = "too small"
                    continue
                blob = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(blob, [c], -1, 255, -1)
                if cv2.countNonZero(cv2.bitwise_and(blob, bad_wide)):
                    rec["status"] = "cut off by the picture edge"
                    continue
                h_cfg, src = cc.heights_mm.get(color), "config"
                age = 0
                if h_cfg is None:                              # reuse this blob's height from the last frames, so a
                    prev = [(h, n) for col, bc, h, n in self._blob_h        # noisy estimate can't make it jump around
                            if col == color and np.linalg.norm(bc - rec["centroid"]) < 40]
                    if prev and prev[0][1] < HEIGHT_REFRESH_FRAMES:
                        (h_cfg, age), src = prev[0], "auto"
                        age += 1
                fit = fit_container(poly, calib.nadir_xy, calib.cam_height, h_cfg, cc.default_height_mm,
                                    cc.max_height_mm, src)
                if fit is None:
                    rec["status"] = "outline not usable"
                    continue
                out.append((color, fit))
                rec["fit"] = fit
                if fit.height_source == "auto":
                    new_h.append((color, rec["centroid"], fit.height, age))
        self._blob_h, self.last_blobs = new_h, blobs
        return out

    def process(self, frame: np.ndarray) -> ContainerState:
        dets = self.detect(frame)
        obs = [Observation(f.center.copy(), f.center.copy(), f.yaw_deg, f.size, c) for c, f in dets]
        tracks = self.tracker.update(obs)
        live = {t.id for t in tracks}
        self._h = {i: h for i, h in self._h.items() if i in live}
        self._meta = {i: m for i, m in self._meta.items() if i in live}
        w = self.cfg.tracking.ema
        state = ContainerState(calibrated=True)
        for t in tracks:
            if t.seen and dets:
                j = min(range(len(obs)), key=lambda i: float(np.linalg.norm(obs[i].center - t.obs.center)))
                fit = dets[j][1]
                self._h[t.id] = fit.height if t.id not in self._h else (1 - w) * self._h[t.id] + w * fit.height
                self._meta[t.id] = fit
            if not t.seen or t.age < self.cfg.tracking.min_age or t.id not in self._meta:
                continue         # a container missed this frame is not reported (no ghosts)
            m, o = self._meta[t.id], t.obs
            state.containers.append(Container(
                t.id, o.color, o.center.copy(), self._h[t.id], tuple(o.size), m.shape, m.yaw_deg, m.height_source,
                self.in_zone(o.center), t.age, t.seen, m.outline + (o.center - m.center)))
        return state
