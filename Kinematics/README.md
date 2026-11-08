# K2: hours 0-12

Copy the CONTENTS of this folder into your GitHub repository root. Keep
`config.py` at the root beside `arm_config.yaml`. If the repo already has a README,
requirements, or gitignore, merge the relevant entries instead of overwriting
your teammates' work. No repository URL was supplied, so this is a local starter
package, not a commit or push to your team's repository.

## Files

| Repo-relative path | Purpose |
| --- | --- |
| config.py | Shared config loader, validation, getters, comment-preserving saves |
| arm_config.yaml | Shared physical config plus starting HSV color thresholds |
| requirements.txt | Python dependencies, including GUI-enabled OpenCV |
| pytest.ini | Run the tests with repo root on the import path |
| .gitignore | Exclude virtual environment, caches, secrets, recordings |
| Kinematics/__init__.py | Python package marker |
| Kinematics/mock_arm.py | Deterministic fake arm and F2 codec adapter interface |
| Vision/__init__.py | Python package marker |
| Vision/camera.py | Camera index or existing video URL/file -> OpenCV BGR frames |
| Vision/detection.py | HSV color mask, bounding boxes, centers, six-key object messages |
| tests/test_config.py | Shared config validation and writeback tests |
| tests/test_camera.py | Camera handling with a fake capture device |
| tests/test_detection.py | Synthetic-image color detection tests |
| tests/test_mock_arm.py | Simulated limits, velocity caps, state, packet adapter tests |
| docs/MEASUREMENTS.md | Geometry worksheet and a second-person check column |

K1 can add FK, IK, and planning files to Kinematics later. These files contain
only K2's hours 0-12 scope. There is no servo hardware output.

## 1. Install and test

Run from the REPOSITORY ROOT, using Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m Kinematics.mock_arm
```

Use `python -m Vision.camera` rather than running `Vision/camera.py` directly;
this keeps shared root imports working. Keep folder capitalization exact.

## 2. Get iPhone video onto the Mac

First try Apple's Continuity Camera. It supports iPhone XR and later on iOS 16+
and Macs on macOS Ventura 13+. Both devices must use the same Apple Account with
two-factor authentication; enable Continuity Camera on the phone, with Wi-Fi and
Bluetooth enabled. For USB operation, trust the Mac on the phone. Your iPhone 14
Pro uses a Lightning data cable. Mount the phone securely, lock it, and keep its
rear cameras aimed at the tabletop.

Check that the iPhone appears in FaceTime's Video menu, then close FaceTime before
running Python. Grant camera access to the terminal/IDE that launches Python if
macOS prompts. Use stable lighting and keep camera effects such as Center Stage
off so your view does not move.

```bash
python -m Vision.camera --source 0
```

Press q to quit. Index 0 may be your Mac's built-in camera. Try `--source 1`,
then `--source 2`, closing each preview before trying the next. Index order is
not guaranteed. Confirm visually that you selected the fixed iPhone camera.
OpenCV cannot guarantee discovery of every Continuity Camera device on every
macOS/Python build; test this on your Mac early. If it is not exposed to Python,
use an existing iPhone streaming app's actual RGB URL:

```bash
python -m Vision.camera --source 'http://PHONE_IP:PORT/ACTUAL_VIDEO_PATH'
```

That URL is a template, not an installed service. No phone streaming app or server
is included here. A local webcam is useful to test detection while troubleshooting.
The preview prints ACTUAL width/height. Record those in camera.resolution; a
requested resolution can be ignored by the device. No frames are resized or mirrored.

Apple setup reference: https://support.apple.com/en-us/102546

## 3. Detect one colored block

Put one red block against a plain contrasting background and run:

```bash
python -m Vision.detection --source 1 --color red
```

Replace 1 with the index that worked. You will see the annotated frame and a
black/white HSV mask. JSON prints twice per second; segmentation runs every frame.
Tune `vision.hsv_ranges.red` and `vision.min_area_px` in arm_config.yaml, then
restart the detector. Red has two HSV bands because hue wraps at the boundary.
Green and blue starting thresholds are also included. These are lighting-dependent
starting values, not measured thresholds. This is color-blob segmentation: it
assumes those colored blobs are blocks in your controlled bench scene, and does
not distinguish a red sleeve from a red block.

Test a saved image without opening any camera or GUI windows:

```bash
python -m Vision.detection --image block.jpg --color red --output annotated.jpg
```

One object message looks like:

```json
{"class":"block","color":"red","pixel_xy":[120.0,85.0],"table_xy":null,"angle":null,"confidence":0.98}
```

The exact six keys are retained. Pixel coordinates are (u right, v down) in the
original captured image. `table_xy` and `angle` are unknown, so remain null in this
phase. Planner consumers must skip unlocalized detections: do not plan a physical
pick when table_xy is null. Confidence is a rectangularity score (contour area /
minimum bounding rectangle area), not a calibrated probability of being a block.
The identity camera matrices in the example are never used by this detector.
Table localization and angle-in-base-frame estimation belong to later calibration.

OpenCV HSV reference: https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html

## 4. Mock arm for team integration

```python
from Kinematics.mock_arm import MockArm

arm = MockArm()
arm.move([0.1, 0.0, 0.0, 0.0, 0.5])  # target model joint coordinates, radians
state = arm.advance(0.1)                 # simulate 0.1 seconds without sleeping
print(state)
```

Order is base, shoulder, elbow, wrist, gripper. The mock starts at each joint's
configured interval midpoint, not an assumed physical home. It rejects bad
targets and simulates capped per-joint speeds. No FK, collision model, actuator
dynamics, watchdog, or signing is implemented in the mock itself.

F2's packet specification was not provided. To use the REAL protocol, F2 supplies
two adapter callbacks, rather than you inventing a competing wire format:

```python
response_bytes = arm.exchange(
    request_bytes,
    decode_targets=f2_decode_to_model_q,
    encode_state=f2_encode_mock_state,
)
```

Those two callback names are examples, not supplied functions. The decoder must
validate length, version, HMAC, and counter, then translate the wire's fixed-point
servo-space targets into five MODEL joint radians. The encoder translates the
mock's `q`, `target_q`, and `moving` fields into the actual protocol response.
A bad decode must raise an exception. No target changes when decoding fails.
Call advance(dt) explicitly between requests to progress simulated motion.
Once F2 provides those callbacks, this seam can speak the team's protocol;
until then only the direct Python interface and adapter fixture tests work.

## 5. Config and measurements

All physical values marked MEASURE ME are invented placeholders. Use the actual
assembled arm and docs/MEASUREMENTS.md to measure geometry, with a teammate checking.
Do not copy assumed internet dimensions into the physical configuration.
The five requested actuator slots include wrist and gripper; verify what the
built mechanism actually provides with K1. No Mk2 linkage or extra wrist is assumed.

```python
from config import load_config, get_joint_limits, get_T_base_cam, save_config

cfg = load_config()
limits = get_joint_limits(cfg)  # new float ndarray (5, 2), radians
T = get_T_base_cam(cfg)          # new float ndarray (4, 4), camera -> base
# After measuring: cfg["links"]["lengths"]["upper_arm"] = measured_length_m
# save_config(cfg)
```

Default YAML path is relative to config.py, independent of the shell's directory.
For custom paths use load_config(path) and save_config(cfg, path). Edit the
returned ruamel CommentedMap directly to retain comments; converting to a dict
or replacing whole subtrees may lose associated comments. Saves validate first,
then atomically replace the file. Only K2 writes; no concurrent-edit merge exists.

Locked base frame: right-handed, origin at base rotation axis at table height,
x forward, y left, z up; meters and radians. Joint/servo mapping contract:

```python
servo_angle_rad = joint["sign"] * q_rad + joint["zero_offset"]
c0, c1 = servo["calibration"]["coefficients"]
pulse_us = c0 + c1 * servo_angle_rad
```

K1 owns the actual joint_to_servo implementation and any nonlinear linkage mapping.
Coefficients are [intercept_us, slope_us_per_rad]. They must map angle_at_min/max
to pulse_min/max within 1e-6 us. After an affine fit, set endpoint angles to
(pulse_endpoint - c0) / c1. Descending shaft angles at ascending pulses are allowed.
Mapped joint limits and gripper endpoints must fit the configured safe ranges.
Effective speed cap is min(joint.max_velocity, safety.max_speed), in rad/s.

T_base_cam uses p_base = T_base_cam @ p_cam, with OpenCV camera axes x right,
y down, z forward and translation in meters. Table homography is pixel -> base
table meters at z=0; it and the transform remain placeholders. Intrinsics null
means uncalibrated. Structural validation does not prove physical safety, camera
calibration, mechanical reachability, or table clearance. Firmware still enforces
limits independently; the LLM never talks directly to servos.

## Hours 0-12 finish checklist

- Config tests pass and available geometry is measured/checked.
- Python shows the fixed iPhone RGB feed.
- One colored block has a stable bounding box and pixel center.
- Teammates can exercise the mock Python API; integrate F2's actual codec as soon
  as it is provided (this remains a protocol integration dependency).

Validation uses synthetic images and fake camera objects, plus a local mock.
The real iPhone feed and real hardware cannot be tested in this build environment.
Result: 277 tests passed on Python 3.12.14. The mock demo and saved-image detector
also ran successfully. Python 3.11 syntax was checked; a Python 3.11 runtime was
not available here.

## Optional after this phase

Servo least-squares fitting, table homography and pixel_to_table, depth, object
orientation, and closed-loop grasp correction are intentionally later work.
