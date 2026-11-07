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
