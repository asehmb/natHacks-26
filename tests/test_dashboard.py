"""Comprehensive unit and integration tests for the natHacks dashboard."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

# Add repo to sys.path
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dashboard.arm_bridge import ArmBridge
from dashboard.events import EventBus, emit, get_recent_events, subscribe
from dashboard.logger import JSONLLogger
from dashboard.metrics import ProductionMetrics
from dashboard.nlp_bridge import process_command
from dashboard.safety import SafetyManager
from dashboard.vision_bridge import VisionBridge
from fastapi.testclient import TestClient
from dashboard.server import app


class TestJSONLLogger(unittest.TestCase):
    def test_logger_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JSONLLogger(log_dir=tmpdir)
            entry = {"event": "test:event", "payload": {"val": 42}, "seq": 1}
            logger.append(entry)
            
            lines = logger.get_lines(limit=10)
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["event"], "test:event")
            self.assertEqual(lines[0]["payload"]["val"], 42)


class TestEventBus(unittest.TestCase):
    def test_emit_and_subscribe(self):
        bus = EventBus()
        received = []

        def on_event(ev):
            received.append(ev)

        bus.subscribe("safety:*", on_event)
        
        bus.emit("arm:moved", {"pos": [1, 2, 3]})
        self.assertEqual(len(received), 0)  # should not match

        bus.emit("safety:rejected", {"reason": "collision"})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event, "safety:rejected")
        self.assertEqual(received[0].payload["reason"], "collision")
        self.assertGreater(received[0].seq, 0)

        recent = bus.get_recent(limit=5)
        self.assertEqual(len(recent), 2)


class TestSafetyManager(unittest.TestCase):
    def setUp(self):
        self.safety = SafetyManager()

    def test_cartesian_boundaries(self):
        # Normal safe reachable target
        res = self.safety.validate_cartesian_target(0.22, 0.0, 0.15)
        self.assertTrue(res.allowed)
        self.assertEqual(res.status, "accepted")

        # Table collision (z < 0)
        res_table = self.safety.validate_cartesian_target(0.22, 0.0, -0.02)
        self.assertFalse(res_table.allowed)
        self.assertEqual(res_table.status, "rejected")
        self.assertIn("Table collision", res_table.reason)

        # Out of reach (radius > 0.30m)
        res_reach = self.safety.validate_cartesian_target(0.40, 0.20, 0.10)
        self.assertFalse(res_reach.allowed)
        self.assertEqual(res_reach.status, "rejected")
        self.assertIn("Out of workspace reach", res_reach.reason)

    def test_joint_limits(self):
        # Base within [-1.0, 1.0] rad
        self.assertTrue(self.safety.validate_joint_target("base", 0.5).allowed)
        
        # Base outside bounds
        res = self.safety.validate_joint_target("base", 1.8)
        self.assertFalse(res.allowed)
        self.assertIn("Joint safety limit violation", res.reason)

    def test_estop_latching(self):
        self.safety.trigger_estop("Test Stop")
        self.assertTrue(self.safety.is_stopped)

        res = self.safety.validate_cartesian_target(0.22, 0.0, 0.15)
        self.assertFalse(res.allowed)
        self.assertIn("E-STOP is latched active", res.reason)

        self.safety.reset_estop()
        self.assertFalse(self.safety.is_stopped)
        self.assertTrue(self.safety.validate_cartesian_target(0.22, 0.0, 0.15).allowed)


class TestNLPBridge(unittest.TestCase):
    def test_process_commands(self):
        # Test Home command
        res = process_command("Home the arm")
        self.assertEqual(res.status, "accepted")

        # Test Pick Up Red Cube
        res_pick = process_command("Pick up the red cube and put it in the blue bin")
        self.assertEqual(res_pick.status, "accepted")

        # Test Clarification for empty or nonsense
        res_empty = process_command("")
        self.assertEqual(res_empty.status, "needs_clarification")

        # Test Unsafe reach command rejection
        res_unsafe = process_command("Move to 0.45 0.20 0.10")
        # Should be rejected or clarified
        self.assertIn(res_unsafe.status, ("rejected", "needs_clarification"))


class TestProductionMetrics(unittest.TestCase):
    def test_metrics_calculations(self):
        m = ProductionMetrics()
        m.reset_stats()
        
        m.set_active_mode("Camera-Guided")
        m.record_success(cycle_time=4.0)
        m.record_success(cycle_time=4.2)
        m.record_reject(reason="Out of reach")

        summary = m.get_summary()
        self.assertEqual(summary["parts_sorted"], 2)
        self.assertEqual(summary["rejects"], 1)
        self.assertAlmostEqual(summary["pick_success_rate"], 66.7, places=1)
        self.assertAlmostEqual(summary["avg_cycle_time_s"], 4.10, places=1)

        # INA219 power calculation
        reading = m.update_ina219(arm_state="move", moving=True)
        self.assertGreater(reading.current_ma, 100)
        self.assertGreater(reading.power_mw, 500)


class TestVisionBridge(unittest.TestCase):
    def test_vision_detections_and_selection(self):
        v = VisionBridge()
        dets = v.get_detections()
        self.assertGreaterEqual(len(dets), 3)

        sel = v.select_target("obj_2")
        self.assertIsNotNone(sel)
        self.assertEqual(sel["id"], "obj_2")

        matched = v.find_best_match(noun="cube", color="red")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.color, "red")

        # Test image generation
        img_bytes = v.render_bench_image()
        self.assertTrue(len(img_bytes) > 0)


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_api_status(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("arm", data)
        self.assertIn("stats", data)
        self.assertIn("vision", data)

    def test_api_command(self):
        r = self.client.post("/api/command", json={"command": "Home the arm"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("status", data)

    def test_api_stop_and_reset(self):
        r_stop = self.client.post("/api/stop")
        self.assertEqual(r_stop.status_code, 200)
        self.assertTrue(r_stop.json()["is_stopped"])

        r_reset = self.client.post("/api/reset")
        self.assertEqual(r_reset.status_code, 200)
        self.assertFalse(r_reset.json()["is_stopped"])

    def test_api_emit(self):
        r = self.client.post("/api/emit", json={
            "event": "sensors:ina219_read",
            "payload": {"current_ma": 420.0, "voltage_v": 5.0}
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "emitted")

    def test_api_vision_frame(self):
        r = self.client.get("/api/vision/frame.jpg")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        self.assertGreater(len(r.content), 1000)

    def test_api_logs_download(self):
        r = self.client.get("/api/logs/download")
        self.assertEqual(r.status_code, 200)

    def test_frontend_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"NATHACKS", r.content)


if __name__ == "__main__":
    unittest.main()
