"""Type a command, the arm does it: NLP -> container vision -> planar path -> ArmLink firmware.

  python talk_to_arm.py --source 0                                   # dry run: prints every MOVE, no hardware
  python talk_to_arm.py --source 0 --port /dev/cu.usbserial-10       # real arm
  python talk_to_arm.py --source 0 --port /dev/cu.usbserial-10 --mic   # push to talk: Enter, speak, Enter
  python talk_to_arm.py --no-gui --container blue:250:36.5 \\
      --say "grab this item, put it in the blue container, then come back to me"   # no camera, no arm

  > grab this item                     close the gripper on what you hand it (at the hand-off pose)
  > put it in the blue container       over the container, down, open, lift
  > come back to me                    back to the hand-off pose, gripper open for the next item

The base servo is broken (disabled in firmware), so the arm only moves forward/back and up/down: containers have to
sit on the arm's centre line. --container COLOR:X_MM[:H_MM] puts a fixed container there instead of (or without) the
camera. The camera needs a calibration from find_containers.py --calibrate. Quit with q (window), Ctrl-D or "quit".
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np

from armvision.container_move import make_arm
from armvision.containers import Container, ContainerState
from armvision.errand import Errand, ErrandConfig

NLP_DIR = Path(__file__).resolve().parent.parent / "nlp"
if str(NLP_DIR) not in sys.path:
    sys.path.append(str(NLP_DIR))   # appended: nlp/ also holds a `vision` package, which must not shadow anything


def parse_container(spec: str) -> Container:
    """COLOR:X_MM[:H_MM] -> a container on the arm's line, e.g. blue:250:36.5."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"expected COLOR:X_MM[:H_MM], got {spec!r}")
    try:
        x, h = float(parts[1]), float(parts[2]) if len(parts) == 3 else 36.5
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected numbers in {spec!r}") from None
    return Container(-1, parts[0].lower(), np.array([x, 0.0]), h, (80.0, 80.0), "round", 0.0, "fixed", True, 99)


def build_dispatcher(errand: Errand, say):
    from nlp import (CloseHand, Dispatcher, Give, Home, OpenHand, PickUp, PutIn, Reply, ReturnToUser, Shake)

    def put_in(action):
        drop = errand.put_in(action)
        say(f"released over the container at x={drop[0] * 1000:.0f} mm, z={drop[2] * 1000:.0f} mm")

    disp = Dispatcher()
    for cls, fn in [(PickUp, errand.grab), (CloseHand, errand.grab), (OpenHand, errand.release), (PutIn, put_in),
                    (ReturnToUser, errand.come_back), (Home, errand.come_back), (Give, errand.come_back),
                    (Reply, lambda a: say(f"arm says: {a.text}")),
                    (Shake, lambda a: say(f"(can't shake {a.kind}: the base is disabled)"))]:
        disp.register(cls, fn)
    return disp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="camera index, stream URL or video; omit to run without vision")
    ap.add_argument("--config", help="container_config.yaml")
    ap.add_argument("--calibration", default="container_calibration.yaml")
    ap.add_argument("--container", action="append", type=parse_container, default=[], metavar="COLOR:X_MM[:H_MM]",
                    help="a fixed container on the arm's line (overrides vision for that colour); repeatable")
    ap.add_argument("--port", help="arm serial port; omit for a dry run (validated and printed, nothing sent)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--link-debug", action="store_true", help="print every ArmLink packet and ACK (heartbeats too)")
    d = ErrandConfig()
    ap.add_argument("--handoff-mm", nargs=2, type=float, default=d.handoff_mm, metavar=("X", "Z"),
                    help="tip pose where you hand items over (default %(default)s)")
    ap.add_argument("--gripper-open", type=float, default=d.gripper_open)
    ap.add_argument("--gripper-closed", type=float, default=d.gripper_closed)
    ap.add_argument("--clearance-mm", type=float, default=d.clearance_mm, help="tip height above the rim on release")
    ap.add_argument("--tip-offset-mm", type=float, default=d.tip_offset_mm,
                    help="how far forward of the commanded point the real gripper lands; it is subtracted from "
                         "every target (default %(default)s; 0 = trust the kinematics)")
    ap.add_argument("--y-tol-mm", type=float, default=d.y_tol_mm, help="max container distance off the arm's line")
    ap.add_argument("--ollama", metavar="MODEL", help="local LLM fallback for sentences the rules don't parse")
    ap.add_argument("--mic", action="store_true", help="push to talk: Enter starts recording, Enter stops "
                    "(pip install sounddevice faster-whisper); you can still type a command instead")
    ap.add_argument("--mic-model", default="base.en", help="faster-whisper model (tiny.en is faster, small.en better)")
    ap.add_argument("--mic-device", help="input device index or name (default: the system microphone)")
    ap.add_argument("--say", action="append", help="run these commands in order, then exit (instead of a prompt)")
    ap.add_argument("--yes", action="store_true", help="don't ask before the first (unplanned) move to the hand-off pose")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args(argv)

    from nlp import OllamaFallback, Processor
    proc = Processor(OllamaFallback(args.ollama) if args.ollama else None)
    mic = None
    if args.mic and not args.say:
        try:
            from armvision.mic import PushToTalk
        except ImportError as exc:
            ap.exit(1, f"--mic needs: pip install sounddevice faster-whisper ({exc})\n")
        print(f"loading speech model {args.mic_model} (downloads on first use)...", flush=True)
        device = int(args.mic_device) if args.mic_device and args.mic_device.isdigit() else args.mic_device
        mic = PushToTalk(args.mic_model, device)

    lock, latest = threading.Lock(), [ContainerState([])]
    fixed = {c.color: c for c in args.container}

    def get_state() -> ContainerState:
        with lock:
            s = latest[0]
        if not fixed:
            return s
        return ContainerState([c for c in s.containers if c.color not in fixed] + list(fixed.values()), True)

    finder = cap = None
    if args.source is not None:
        from armvision.__main__ import open_camera
        from armvision.containers import ContainerFinder, load_container_config
        from armvision.groundcal import GroundCalibration
        try:
            finder = ContainerFinder(load_container_config(args.config), GroundCalibration.load(args.calibration))
            still = Path(args.source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            cap = None if still else open_camera(args.source)
        except (ValueError, RuntimeError) as exc:
            ap.exit(1, f"{exc}\n(calibrate with find_containers.py --calibrate)\n")

    status = ["starting"]

    def say(msg: str) -> None:
        status[0] = msg
        print(msg, flush=True)

    cfg = ErrandConfig(tuple(args.handoff_mm), args.clearance_mm, args.y_tol_mm, args.gripper_open, args.gripper_closed,
                       tip_offset_mm=args.tip_offset_mm)
    arm = errand = None
    failures = [0]
    done, quit_ = threading.Event(), threading.Event()

    def handle(text: str) -> None:
        result = proc.process(text)
        print(f"[{result.source}] " + "  ".join(json.dumps(a.to_dict()) for a in result.actions), flush=True)
        try:
            disp.dispatch(result.actions)
        except KeyError as exc:
            failures[0] += 1
            say(f"FAILED: can't do that yet ({exc.args[0]}); nothing moved")
        except Exception as exc:          # one bad command must not kill the session; the rest of the chain is skipped
            failures[0] += 1
            say(f"FAILED, arm stopped where it is: {exc}")

    def listen() -> str:
        from armvision.mic import MicError
        say("listening... press Enter when you're done")
        try:
            heard = mic.listen(input)
        except MicError as exc:
            say(f"mic: {exc}")
            return ""
        say(f'heard: "{heard}"' if heard else "heard nothing, try again")
        return heard

    def commands():
        try:
            if args.say:
                for text in args.say:
                    if quit_.is_set():
                        break
                    print(f"> {text}", flush=True)
                    handle(text)
                return
            while not quit_.is_set():
                try:
                    text = input("[Enter] talk, or type (q quits) > " if mic else "> ").strip()
                except EOFError:
                    break
                if text.lower() in ("q", "quit", "exit"):
                    break
                if not text and mic:
                    text = listen()
                if text:
                    handle(text)
        finally:
            done.set()

    try:
        if args.port and not args.yes:
            input(f"Enter = move to the hand-off pose x={cfg.handoff_mm[0]:.0f} z={cfg.handoff_mm[1]:.0f} mm "
                  "(one jump from wherever the arm is; Ctrl-C to abort) ")
        arm = make_arm(args.port, args.baud, cfg.gripper_open, dry=True, verbose=args.link_debug)
        errand = Errand(arm, get_state, cfg, log=say)
        errand.ready()
        say("ready: hand me the item, then e.g. 'grab this item, put it in the blue container, then come back to me'")
        disp = build_dispatcher(errand, say)
        threading.Thread(target=commands, name="commands", daemon=True).start()

        if finder is None:
            done.wait()
        else:
            import cv2
            from find_containers import draw, text as draw_text
            while not done.is_set():
                ok, frame = (True, cv2.imread(args.source)) if cap is None else cap.read()
                if not ok or frame is None:
                    say("camera stopped: vision is frozen at the last frame")
                    done.wait()           # let the commands finish; q / quit / Ctrl-D still ends the session
                    break
                try:
                    state = finder.process(frame)
                except Exception as exc:  # e.g. a frame size that doesn't match the calibration
                    print(f"skipping frame: {exc}", file=sys.stderr)
                    continue
                with lock:
                    latest[0] = state
                if args.no_gui:
                    continue
                shown = draw(frame, finder, get_state(), finder.calib_for(frame), None)
                draw_text(shown, status[0][:110], (20, 30), (0, 255, 255), 0.6)
                cv2.imshow("talk to the arm (q quits)", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            quit_.set()
            if not args.no_gui:
                cv2.destroyAllWindows()
    except KeyboardInterrupt:
        quit_.set()
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if arm is not None:
            arm.close()                   # real arm: STOP, which detaches the servos
        if cap is not None:
            cap.release()
    return 1 if failures[0] else 0


if __name__ == "__main__":
    sys.exit(main())
