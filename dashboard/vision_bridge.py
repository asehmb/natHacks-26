"""Vision subsystem bridge with synthetic bench fallback and target tracking."""

from __future__ import annotations

import io
import math
import random
import time
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from dashboard.events import emit

# Try importing PIL for frame synthesis
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class DetectedObject:
    id: str
    label: str
    color: str
    confidence: float
    bbox: List[int]        # [x1, y1, x2, y2] in 640x480 frame
    table_xyz: List[float] # [x, y, z] in meters relative to arm base
    selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "color": self.color,
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox,
            "table_xyz": [round(c, 3) for c in self.table_xyz],
            "selected": self.selected,
        }


class VisionBridge:
    def __init__(self):
        self._lock = Lock()
        self.selected_target_id: Optional[str] = "obj_1"  # Default select first
        self.camera_source_mode: str = "webcam"           # "webcam" or "simulated"
        self._last_frame_time = time.time()
        
        # Pre-populate bench objects (EEZYbotARM workspace: ~0.15 - 0.28m forward)
        self.objects: List[DetectedObject] = [
            DetectedObject(
                id="obj_1",
                label="red cube",
                color="red",
                confidence=0.96,
                bbox=[140, 160, 240, 260],
                table_xyz=[0.220, -0.060, 0.025],
                selected=True,
            ),
            DetectedObject(
                id="obj_2",
                label="blue bin",
                color="blue",
                confidence=0.94,
                bbox=[380, 120, 520, 260],
                table_xyz=[0.240, 0.080, 0.040],
                selected=False,
            ),
            DetectedObject(
                id="obj_3",
                label="green cylinder",
                color="green",
                confidence=0.89,
                bbox=[260, 280, 340, 360],
                table_xyz=[0.180, 0.010, 0.020],
                selected=False,
            ),
            DetectedObject(
                id="obj_4",
                label="yellow bolt",
                color="yellow",
                confidence=0.82,
                bbox=[100, 320, 160, 380],
                table_xyz=[0.160, -0.100, 0.015],
                selected=False,
            ),
        ]

    def get_detections(self) -> List[Dict[str, Any]]:
        with self._lock:
            # Add slight realistic jitter to confidence and position
            for obj in self.objects:
                obj.confidence = max(0.70, min(0.99, obj.confidence + random.uniform(-0.01, 0.01)))
            return [o.to_dict() for o in self.objects]

    def get_selected_target(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            for obj in self.objects:
                if obj.selected:
                    return obj.to_dict()
        return None

    def select_target(self, target_id_or_label: str) -> Optional[Dict[str, Any]]:
        """Select an object by id, label, or color."""
        target_id_or_label = target_id_or_label.lower().strip()
        matched: Optional[DetectedObject] = None

        with self._lock:
            for obj in self.objects:
                if obj.id == target_id_or_label or target_id_or_label in obj.label.lower() or obj.color == target_id_or_label:
                    obj.selected = True
                    matched = obj
                else:
                    obj.selected = False

        if matched:
            emit("vision:target_selected", {
                "target": matched.to_dict(),
                "selection_mode": "manual",
            })
            return matched.to_dict()
        return None

    def find_best_match(self, noun: Optional[str] = None, color: Optional[str] = None, position: Optional[str] = None) -> Optional[DetectedObject]:
        """Find detection matching NLP parsed attributes."""
        with self._lock:
            candidates = list(self.objects)

        if color:
            candidates = [c for c in candidates if c.color.lower() == color.lower()]
        if noun:
            candidates = [c for c in candidates if noun.lower() in c.label.lower()]
        if position:
            if "left" in position.lower():
                candidates = sorted(candidates, key=lambda c: c.bbox[0])[:1]
            elif "right" in position.lower():
                candidates = sorted(candidates, key=lambda c: c.bbox[0], reverse=True)[:1]

        return candidates[0] if candidates else None

    def render_bench_image(self) -> bytes:
        """Render a synthetic 640x480 workbench JPEG image with detected bounding boxes."""
        if not PIL_AVAILABLE:
            return b""

        # Create dark workbench canvas
        width, height = 640, 480
        img = Image.new("RGB", (width, height), color=(26, 32, 44))
        draw = ImageDraw.Draw(img)

        # Draw grid lines to simulate inspection workbench
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, height)], fill=(36, 45, 62), width=1)
        for y in range(0, height, 40):
            draw.line([(0, y), (width, y)], fill=(36, 45, 62), width=1)

        # Draw arm base origin marker at bottom center
        base_x, base_y = 320, 460
        draw.ellipse([(base_x - 18, base_y - 18), (base_x + 18, base_y + 18)], outline=(100, 116, 139), width=2)
        draw.line([(base_x - 24, base_y), (base_x + 24, base_y)], fill=(100, 116, 139), width=2)
        draw.line([(base_x, base_y - 24), (base_x, base_y + 24)], fill=(100, 116, 139), width=2)
        draw.text((base_x - 30, base_y - 35), "ARM BASE", fill=(148, 163, 184))

        # Color mapping
        color_rgb = {
            "red": (239, 68, 68),
            "blue": (59, 130, 246),
            "green": (34, 197, 94),
            "yellow": (234, 179, 8),
        }

        with self._lock:
            objs = list(self.objects)

        for obj in objs:
            x1, y1, x2, y2 = obj.bbox
            col = color_rgb.get(obj.color, (200, 200, 200))
            
            # Fill object silhouette
            draw.rectangle([x1 + 4, y1 + 4, x2 - 4, y2 - 4], fill=col)
            
            # Draw bounding box
            box_col = (0, 255, 204) if obj.selected else (255, 255, 255)
            draw.rectangle([x1, y1, x2, y2], outline=box_col, width=3 if obj.selected else 2)

            # Draw target reticle if selected
            if obj.selected:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                draw.line([(cx - 15, cy), (cx + 15, cy)], fill=(0, 255, 204), width=2)
                draw.line([(cx, cy - 15), (cx, cy + 15)], fill=(0, 255, 204), width=2)
                draw.text((x1, y1 - 22), f">> TARGET: {obj.label.upper()} <<", fill=(0, 255, 204))
            else:
                label_text = f"{obj.label} ({obj.confidence:.2f})"
                draw.text((x1, y1 - 18), label_text, fill=(255, 255, 255))

            # XYZ tag
            xyz_text = f"X:{obj.table_xyz[0]:.2f} Y:{obj.table_xyz[1]:.2f} Z:{obj.table_xyz[2]:.2f}m"
            draw.text((x1, y2 + 4), xyz_text, fill=(148, 163, 184))

        # HUD Overlay
        draw.text((15, 15), "VISION FEED: BENCH INSPECTION CAM (SIMULATED)", fill=(56, 189, 248))
        draw.text((15, 32), f"OBJECTS DETECTED: {len(objs)} | CONFIDENCE THRESHOLD: 0.75", fill=(148, 163, 184))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


# Global singleton vision bridge
vision = VisionBridge()
