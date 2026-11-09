# armvision — mat-calibrated vision for the EEZYbot arm

Finds objects on a fixed-size mat, tracks them and the arm, and outputs grasp coordinates in the **arm base frame
(mm; x forward, y left, z up)**, ready for `EEZYbotARM_Mk2.inverseKinematics(x, y, z)`.

## Setup
```
cd vision && python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## 1. Print and place markers
Edit `mat_config.yaml` (mat size, marker centres, arm base position on the mat), then:
```
python -m armvision.markers --out markers/
```
Print at 100 % (300 DPI, no "fit to page"), measure one marker's black square with a ruler, and tape markers 0–3 at the
configured mat positions. Marker 10 (small) goes flat on top of the gripper.

## 2. (Optional, recommended) calibrate the camera
`python -m armvision --source 0`, then press `c` on ~15 views of the mat from varied angles/distances and `k` to solve
and write `camera_intrinsics.yaml`. Without it a 60° FOV guess is used (warning printed): object positions are still
correct (homography), only the gripper's 3D position is approximate.

## 3. Run
```
python -m armvision --source 0 [--reference empty_mat.png]
```
Press `r` with an empty mat to capture the reference, add objects, `q` quits. One JSON line per frame on stdout
(`--no-gui` for headless; it takes the reference from the first frame with the mat visible).

```json
{"mat_visible": true,
 "arm": {"gripper_mm": [182.4, -12.0, 64.1], "visible": true},
 "objects": [{"id": 3, "color": "red", "label": "object", "center_mm": [210.0, 40.2, 0],
              "grasp_mm": [196.5, 35.1, 15.0], "grasp_yaw_deg": 72.0, "size_mm": [58.0, 21.0],
              "occluded": false, "age": 41}]}
```

From Python: `VisionSystem(cfg).process(frame)` returns a `WorldState`; `state.find(color="red", position="left")`
returns matching objects, best first.

## Notes
- Grasp = end of the long axis nearest the arm base, inset `grasp_inset_mm`; round/square objects use the centroid.
- `use_near_half` (off by default) trims tall objects' lean away from the camera but also shortens flat elongated
  objects, so enable it only for tall objects.
- Blobs touching the base-zone circle or the gripper marker are the arm and are never reported as objects; tracked
  objects under the arm are kept and flagged `occluded`.
- If the camera is calibrated, calibrate again from scratch (delete the yaml first) so views are not pre-undistorted.

## Container finder (no markers)
`find_containers.py` finds coloured containers on a white sheet ("the red container") with **no printed markers**.
It reports each container's centre, footprint size and height in the arm base frame (mm; x forward, y left, z up),
correcting for the lean of a tall container seen from a tilted camera (a blob centroid is 5–18 cm off). It does not
touch the mat pipeline above and has its own config, `container_config.yaml`.

One-time setup:
1. Draw a pen tick on each long edge, `sheet.mark_distance_mm` (250) from the arm-end edge, beyond any corner
   cutouts (the arm-end corners are never clicked, so they may be cut out). Measure the sheet width and length and
   where the base axis sits (`arm.base_from_edge_mm`, `arm.base_offset_mm`) and edit the config. Optionally add a
   measured height per colour under `containers.heights_mm`.
2. `python find_containers.py --calibrate --source N`: click the 2 far (camera-end) corners, then the 2 ticks (the
   arm's left is the right side of the screen; the zoom inset helps aim, `u` undoes). Check the printed
   reprojection error, FOV and camera height against a tape measure and that the grid sits on the sheet, then press
   `s`. Redo only if the camera is bumped. If the estimated FOV is off, set `camera.fov_deg` (a known FOV makes the
   pose far more robust to click noise). The further apart the ticks and far corners, the better.
3. `python find_containers.py --source N --target red` (`--json` prints one line per frame, `--no-gui` for headless,
   a still image as `--source` with `--save out.png` for a check). `t` shows the top-down view. `--debug` lists every
   colour blob with its area and why it was dropped and opens a top-down window with the blobs outlined (green =
   became a container, red = dropped). `containers.colors` limits the colours searched (e.g. `[orange, blue]`) and
   `containers.sheet_margin_mm` ignores colour off the sheet (table, hands).

The overlay draws each container's base and rim outlines: if the height is wrong they visibly stop matching the real
container. Python: `ContainerFinder(load_container_config(), GroundCalibration.load(path)).process(frame).find("red")`.

```json
{"calibrated": true,
 "containers": [{"id": 1, "color": "red", "center_mm": [252.3, 58.1], "top_mm": [252.3, 58.1, 81.0],
                 "size_mm": [119.6, 119.6], "shape": "round", "yaw_deg": 0.0, "height_mm": 81.0,
                 "height_source": "auto", "in_zone": true, "age": 14, "seen": true}],
 "target": {"id": 1, "color": "red", "...": "..."}}
```
`top_mm` is the centre of the opening at rim height. `find()` returns only containers inside `zone`, seen in the
current frame and older than `tracking.min_age`. `height_source` is `config`, `auto` (estimated from the outline) or
`default` (`containers.default_height_mm`, when the estimate fails its sanity checks; `?` in the overlay).

Known limits:
- Only coloured containers are detected (no white, black or clear ones).
- Touching containers of the same colour merge into one.
- If the arm or the table near the sheet edge is the same colour as a container they can merge: keep containers
  ≥ 5 cm from the side edges if the table is coloured.
- A very wide bowl hides its base behind its rim, so the automatic height is rejected or biased low (a 20 mm height
  error costs 13–50 mm of centre error): set `containers.heights_mm` for those.
- No lens-distortion model; the zone sits inside the 4 clicked points, so the error stays at a few mm.
- A container whose leaning top leaves the picture or the top-down view is dropped rather than mis-reported.
- Turn off camera auto-framing (e.g. Center Stage on an iPhone used as a webcam).

## Move the arm above a container

Press `g` in the live view (or pass `--go`) and the arm moves `--clearance-mm` (default 20) above the rim of the
`--target` container. If you give a serial port it drives the real arm through `firmware/comms/send_ik.py`. Without
one it runs the Kinematics planner on the MockArm. Before anything moves it checks that the container is in the
zone, has been stable for a few frames, and that the hover point passes IK.

    pip install -e ".[arm]"                                                          # Kinematics + pyserial deps
    python find_containers.py --source 0 --target red                                # g = simulated move
    python find_containers.py --source 0 --target red --port /dev/cu.usbserial-10    # g = real arm
    python find_containers.py --source 0 --target red --no-gui --json --go           # move once, then exit

From the NLP side ("put it in the red container"):

    from armvision.container_move import make_arm, make_put_in_handler
    dispatcher.register(PutIn, make_put_in_handler(lambda: latest_container_state, make_arm(port)))

The current IK model reaches only about 235–265 mm from the base at a hover height of 60 mm, and 312 mm at most. A
container outside that range is rejected with an "unreachable" message and the arm does not move.

## Talk to the arm (end to end, base locked)

`talk_to_arm.py` wires the whole chain together: typed text → `nlp` parser → container vision → a planar path →
ArmLink MOVEs. The base servo is broken and disabled in the firmware (`servo_bridge.h`, `enabled[0] = false`;
reflash `combined_only`), so the arm only moves forward/back and up/down along its centre line (y = 0).

    python talk_to_arm.py --source 0                                  # dry run: every MOVE validated and printed
    python talk_to_arm.py --source 0 --port /dev/cu.usbserial-10      # real arm (asks before the first move)
    python talk_to_arm.py --no-gui --container blue:250:36.5 \
        --say "grab this item, put it in the blue container, then come back to me"   # no camera, no arm

    > grab this item                    gripper closes on what you hand it, at the hand-off pose
    > put it in the blue container      over the container, down to --clearance-mm (40) above the rim, open, lift
    > come back to me                   back to the hand-off pose, gripper still open

- The hand-off pose ("me") is the arm reaching out past the containers: `--handoff-mm 320 120` (x, z). It goes there
  at start-up with the gripper open.
- The gripper angles are placeholders: tune `--gripper-open` (120) / `--gripper-closed` (60) with `grab` and
  `open your hand` first.
- A container must be within `--y-tol-mm` (40) of the arm's line, or the arm refuses and tells you which way to slide
  it. `--container COLOR:X_MM[:H_MM]` puts a fixed container on the line instead of using vision.
- Paths are 10 mm straight-line steps. Every waypoint passes IK and the firmware limits before the first MOVE is
  sent. If one command fails, the rest of the sentence is skipped and the arm stays where it is. Commands with no
  handler (rotate joint, move left…) do nothing.
- Quitting sends STOP, which detaches the servos, so support the arm.

Push to talk: `pip install -e ".[voice]"`, then add `--mic`. At the prompt press Enter, say the command, press Enter
again. It prints what it heard and runs it; you can still type instead. Speech is transcribed offline by faster-whisper
(`--mic-model base.en`, about 140 MB downloaded on first use; `tiny.en` is faster). The first time, macOS asks to let
your terminal use the microphone. If it only hears silence, allow it in System Settings > Privacy & Security >
Microphone and restart the terminal.
