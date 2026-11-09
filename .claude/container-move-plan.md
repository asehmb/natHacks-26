# Vision → kinematics: move the hand 20 mm above "the red container"

## Context
`find_containers.py` / `armvision.containers` already finds containers and reports each one's centre and height in the
arm base frame (mm). Nothing passes that to the arm yet. When told to, vision should take the tracked container's
position, build a target 20 mm above its rim, IK-check it, and move the arm there. The command can come from a key in
the live view or from the NLP `PutIn` action ("put it in the red container"). It drives the real arm over USB when a
port is given and the Kinematics MockArm simulation otherwise.

## New module `vision/armvision/container_move.py`
- `hover_point_m(c: Container, clearance_mm=20.0) -> (x, y, z)` in meters:
  `(c.center_mm[0], c.center_mm[1], c.height_mm + clearance_mm) / 1000`. Vision is already in the base frame, so the
  only conversion is mm → m (same rule as `planner_bridge.target_to_planner_coords`).
- `select_container(state: ContainerState, color) -> Container`: uses `state.find(color)` (in zone, seen, stable,
  nearest). Raises `planner_bridge.TargetValidationError` if the state is not calibrated, nothing matches (the message
  lists the colours that are visible but out of zone or unstable), or the coordinates are not finite.
- `move_above_container(arm, state, color, clearance_mm=20.0) -> (Container, xyz_m)`: select → hover point →
  `planner_bridge.check_reachability(xyz)` (IK; raises with the xyz if unreachable) → `arm.move_to_xyz(xyz)`. Nothing is
  sent unless every check passes.
- Arm backends, both with `move_to_xyz(xyz_m)` and `close()`:
  - `SimArm`: `simulate.build_arm()` + `planner.Planner`. It calls `go_home()` once, then `go_to(xyz)` (straight line
    and validated, from Kinematics). It prints the final tip position from `fk`. Imports go through
    `planner_bridge._get_kinematics()`'s sys.path setup.
  - `UsbArm(port, baud=115200, gripper_deg=90)`: lazily imports `firmware/comms/send_ik.py` (adds that dir to
    sys.path). On creation it opens pyserial, `ArmLinkSession(ser).open()` and `.arm()`. `move_to_xyz` runs
    `angles_from_xyz(xyz, gripper_deg)`, then `payload_for_angles` (this applies the temporary 60–120° firmware limits),
    then `session.move(...)`. The session stays armed between commands and the heartbeat thread keeps it alive.
    `close()` sends `stop()`, then `session.close()` and `ser.close()`. A `SessionError` is turned into a clear error
    (you need to reconnect).
  - `make_arm(port=None, ...)` returns `UsbArm` when a port is given, `SimArm` otherwise.
- `make_put_in_handler(get_state, arm, clearance_mm=20.0)` returns `handler(action)`. The handler reads
  `action.container.color` (duck-typed, so armvision does not import nlp) and calls `move_above_container`. Register
  it with `Dispatcher.register(PutIn, handler)`. Only the hover move is in scope; releasing the gripper is not.

## `vision/find_containers.py`
- New flags:
  - `--port` (real arm; omit for the sim)
  - `--baud`
  - `--gripper-deg` (default 90; holds the gripper during the move)
  - `--clearance-mm` (default 20)
  - `--go` (headless: move once the target is stable, then exit)
- Live view: key `g` runs `move_above_container(arm, state, args.target, clearance)`. It needs `--target`. It prints
  the container id, the xyz in mm and the result. Errors are printed and drawn on screen, and the loop keeps running.
  Update the docstring key list.
- The arm is built lazily on the first `g` or `--go`, so plain viewing never touches the serial port. It is closed in
  the existing `finally`.
- `--json` with `--go` adds `"move": {"xyz_m": [...], "ok": true}` or `{"ok": false, "error": "..."}`.

## Tests: `vision/tests/test_container_move.py`
- Fake arm that records `move_to_xyz` calls. A `ContainerState` built by hand from `Container(...)`, following
  `tests/test_containers.py`.
- Cases:
  - the hover z is `height + 20` and the result is in meters
  - no matching or out-of-zone container raises and the arm is not called
  - an unreachable point (for example 0.6 m out) raises and the arm is not called
  - the `PutIn`-style handler with a `SimpleNamespace(container=SimpleNamespace(color="red"))`
  - colour aliases go through `normalize_color`
- `SimArm` smoke test: a reachable point such as (0.22, 0, 0.06) and the `fk` of the final q is within 5 mm. Skip if
  Kinematics does not import.

## Docs
- Add a short "Move to a container" section to `vision/README.md`: the commands, the `g` key, and the PutIn
  registration snippet.
- Save this plan as `.claude/container-move-plan.md`.

## Known limits (shown to the user, not fixed here)
- The configured zone is x 300–400 mm. The current IK model reaches only about 220–325 mm from the base, so far
  containers will fail IK with a clear message, not move.
- The real arm gets one MOVE (a joint jump), the same as `send_ik.py`, with no streamed line. The servo limits in
  `send_ik.LIMITS` are temporary and may reject poses.

## Verification
1. `cd vision && python -m pytest tests/test_container_move.py tests/test_containers.py -q`
2. Sim on a still image: `python find_containers.py --source containers.png --no-gui --target orange --go`. Expect the
   container xyz and the sim's final tip pose, or a clear IK-unreachable message.
3. Live sim: `python find_containers.py --source 0 --target red`, then press `g`.
4. Hardware: `python find_containers.py --source 0 --target red --port /dev/cu.usbserial-10`, press `g`, and check
   with a ruler that the gripper ends up about 20 mm above the rim.
