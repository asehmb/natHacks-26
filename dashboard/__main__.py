"""Command line entry point for running the natHacks control dashboard."""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="natHacks '26 Robotic Sorting Arm Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  NATHACKS '26 ROBOTIC SORTING ARM DASHBOARD")
    print(f"=======================================================")
    print(f"  Control URL: http://localhost:{args.port}")
    print(f"  WebSocket:   ws://localhost:{args.port}/ws")
    print(f"  Event Bus:   from dashboard.events import emit")
    print(f"  HTTP Emit:   POST http://localhost:{args.port}/api/emit")
    print(f"  JSONL Log:   logs/latest_run.jsonl")
    print(f"=======================================================\n")

    uvicorn.run("dashboard.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
