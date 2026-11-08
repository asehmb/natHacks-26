"""FastAPI server providing real-time WebSockets, REST APIs, and static assets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from dashboard.arm_bridge import arm
from dashboard.events import emit, get_recent_events, set_broadcast_callback
from dashboard.logger import get_logger
from dashboard.metrics import metrics
from dashboard.nlp_bridge import process_command
from dashboard.safety import safety
from dashboard.vision_bridge import vision

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="natHacks '26 Robot Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
active_connections: List[WebSocket] = []


def broadcast_sync(event_dict: Dict[str, Any]) -> None:
    """Callback from event bus: schedules broadcast to all connected WebSockets."""
    msg = json.dumps({"type": "event", "data": event_dict})
    
    # Try to push via running event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            for ws in list(active_connections):
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)
    except RuntimeError:
        pass


set_broadcast_callback(broadcast_sync)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send initial full state snapshot
        initial_payload = {
            "type": "init",
            "arm": arm.get_status(),
            "stats": metrics.get_summary(),
            "safety": {
                "is_stopped": safety.is_stopped,
                "last_rejection": safety.last_rejection,
            },
            "vision": {
                "detections": vision.get_detections(),
                "selected_target": vision.get_selected_target(),
            },
            "recent_events": get_recent_events(limit=40),
        }
        await websocket.send_text(json.dumps(initial_payload))

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "command":
                    cmd_text = msg.get("text", "")
                    res = process_command(cmd_text)
                    await websocket.send_text(json.dumps({"type": "command_result", "data": res.to_dict()}))
                elif action == "stop":
                    arm.stop("Operator clicked STOP button in UI")
                elif action == "reset":
                    arm.reset()
                elif action == "select_target":
                    target_id = msg.get("target_id", "")
                    vision.select_target(target_id)
                elif action == "set_mode":
                    mode = msg.get("mode", "Camera-Guided")
                    metrics.set_active_mode(mode)
                elif action == "heartbeat":
                    arm.heartbeat()
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ---------------------------------------------------------------- REST Endpoints

@app.post("/api/command")
def api_command(req: Dict[str, Any]):
    text = req.get("command", "")
    res = process_command(text)
    return res.to_dict()


@app.post("/api/stop")
def api_stop():
    arm.stop("Operator commanded emergency stop via API")
    return {"status": "stopped", "is_stopped": safety.is_stopped}


@app.post("/api/reset")
def api_reset():
    arm.reset()
    return {"status": "ready", "is_stopped": safety.is_stopped}


@app.get("/api/status")
def api_status():
    return {
        "arm": arm.get_status(),
        "stats": metrics.get_summary(),
        "safety": {
            "is_stopped": safety.is_stopped,
            "last_rejection": safety.last_rejection,
        },
        "vision": {
            "detections": vision.get_detections(),
            "selected_target": vision.get_selected_target(),
        },
    }


@app.get("/api/events")
def api_events(limit: int = 50, category: Optional[str] = None):
    return get_recent_events(limit=limit, category=category)


@app.post("/api/emit")
def api_emit(req: Dict[str, Any]):
    """External emit endpoint for C/C++ firmware or test scripts."""
    event_name = req.get("event")
    if not event_name:
        raise HTTPException(status_code=400, detail="Missing required field 'event'")
    payload = req.get("payload", {})
    ev = emit(event_name, payload)
    return {"status": "emitted", "event": ev.to_dict()}


@app.get("/api/vision/detections")
def api_vision_detections():
    return vision.get_detections()


@app.post("/api/vision/select")
def api_vision_select(req: Dict[str, Any]):
    target_id = req.get("target_id", "")
    target = vision.select_target(target_id)
    return {"selected": target}


@app.get("/api/vision/frame.jpg")
def api_vision_frame():
    jpeg_bytes = vision.render_bench_image()
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/logs/download")
def api_download_logs():
    content = get_logger().export_content()
    return Response(
        content=content,
        media_type="application/x-jsonlines",
        headers={"Content-Disposition": "attachment; filename=run_log.jsonl"},
    )


@app.post("/api/mode")
def api_set_mode(req: Dict[str, Any]):
    mode = req.get("mode", "Camera-Guided")
    metrics.set_active_mode(mode)
    return {"mode": metrics.active_mode}


@app.post("/api/stats/reset")
def api_reset_stats():
    metrics.reset_stats()
    return {"status": "stats_reset"}


# Mount static frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def get_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Dashboard loading...</h1>")
