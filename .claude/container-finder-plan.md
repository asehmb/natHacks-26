# Marker-free "red container" finder for the white sheet

> Status (2026-11-08): planned, not implemented yet. The numbers below come from a throwaway in-memory simulation, not the real rig.

## Context
The arm has to drop handed-over objects into a container (e.g. "the red container") on a ~43 cm wide white sheet.
The existing `vision/armvision` pipeline doesn't fit this job:
- It needs 5 printed ArUco markers (4 cut out and taped at the mat corners, plus 1 on the gripper).
- It assumes a side view.
- It reports the ground-plane centroid of a blob, which is only right for flat objects.

The new setup is different. A fixed camera sits on the sheet, faces the arm head-on and looks down at about 45°. From
that angle a container's top "leans" away from the camera. A blob centroid is then 5–18 cm off (measured in the
simulation below), which is enough to miss the container.

Goal: find every coloured container on the sheet and estimate each one's **centre, footprint size and height** in the
**arm base frame**. Units are mm: x forward (towards the camera), y to the arm's left, z up. This is the same frame
as `Kinematics/arm_config.yaml`. Only containers inside a configurable valid zone in front of the arm count. Wiring
this into the kinematics comes later; this work provides `find("red")` and a JSON stream.

## Approach (prototyped in an in-memory simulation; numbers below come from it)
1. **Zero-print calibration, once.** Draw a pen tick on each long edge of the sheet, `mark_distance_mm` (default
   400) from the arm-end edge. In a one-time UI, click those 2 ticks plus the 2 sheet corners at the arm end (a zoom
   inset helps aim). The known sheet width, tick distance and base position give the arm-frame coordinates of the 4
   points. From those 4 points:
   - `cv2.findHomography` gives the mapping between the mat and the image.
   - The focal length is estimated from the same homography (principal point at the image centre, square pixels).
     In the simulation it was within 1–2 % even with 2 px of click noise, so no FOV guess and no checkerboard are
     needed.
   - `cv2.solvePnP(..., SOLVEPNP_IPPE)` gives the camera centre: the point on the mat straight below it (the nadir N)
     and its height h_cam.
2. **Segmentation without an empty-mat reference.** Warp to a top-down view (2 px/mm, arm-frame mm) that covers the
   zone plus room for leaning container tops. Build one HSV mask per colour from `objects.CHROMATIC`, `MIN_SAT` and
   `MIN_VAL`, the same colour words the NLP side uses. The white sheet and grey shadows have no colour, so they drop
   out on their own. Blobs that touch the view border or are too small are discarded.
3. **Height-aware footprint.** In the top-down view, vertical walls show up as lines pointing away from N, and a point
   at height h is pushed outward from N by a factor h_cam/(h_cam − h). Each outline point is classified by its
   outward normal against that outward direction:
   - **Rim** (top edge, normal pointing away from N, dot > 0.25): pulled back towards N by the factor (1 − H/h_cam).
   - **Base** (bottom edge on the mat, normal pointing towards N, dot < −0.5): kept where it is.
   - **Round** if a circle fits the rim points well (rms/r < 5 %). The centre is the rim circle's centre; the size is
     the rim diameter.
   - **Box** otherwise. `grasp.min_area_rect` over the base points plus the corrected rim points gives the centre,
     L × W and yaw.
   - **Height H** comes from `containers.heights_mm[colour]` if set. Otherwise it is estimated:
     - Round: the base-arc circle (only points with dot < −0.8) must share its centre with the rim circle. That gives
       k = ((c_rim′ − N)·(c_base − N)) / |c_base − N|² and H = h_cam (1 − 1/k).
     - Box: the H that makes base ∪ corrected rim most rectangular (coarse-to-fine 1-D search).
     - Sanity checks: base-arc inlier share ≥ 80 %, 25 mm ≤ H ≤ 0.8 h_cam. If they fail, use `default_height_mm`
       and report `height_source: "default"`.
   - Simulation, with the right H: centre error < 3 mm for a cylinder, a tapered cup, a bowl and a box rotated 30°.
     The naive centroid was 48–184 mm off.
   - Simulation, auto-H: centre within 0.2–7 mm for cylinders, cups, tubs, boxes and moderate bowls. A very wide
     bowl whose base is hidden behind its rim gives H ≈ 13 mm, which the H ≥ 25 mm check catches. Wide bowls should
     have their height set in the config.
   - A 20 mm height error costs 13–50 mm of centre error, which is why a configured height wins.
4. **Tracking and zone.**
   - Reuse `tracker.Tracker` (gate 30 mm, EMA 0.3) for stable ids and smoothing. Keep a per-track EMA of the height
     beside it, the same way `pipeline.py` keeps labels.
   - A container is `in_zone` if its centre is inside the x/y box, within `max_reach_mm` of the base axis, and within
     ±`max_angle_deg` of straight ahead. The default 30° is the q1 limit in `Kinematics/kinematic_model.py`. With the
     current model the arm reaches only about 220–325 mm from the base axis at z = 120 mm.
   - `find(color)` returns only containers that are in the zone, seen this frame and stable (age ≥ `min_age`). Colour
     names are normalised with `planner_bridge.normalize_color`.

## Files
- **New `vision/armvision/groundcal.py`**: ground-plane camera model built from clicked points.
  - `reference_points(cfg)`: the 4 default points in the arm frame, or `sheet.points` if the config overrides them.
  - `estimate_focal(H, size)` and `GroundCalibration.from_points(img_pts, base_pts, size, fov_deg=None)`, falling
    back to `mat.camera_matrix` with 60° if the estimate is implausible (HFOV outside 25–140°).
  - Fields: `K`, `R`, `t`, `H_base2img`, `nadir_xy`, `cam_height`, `reproj_rms`.
  - Methods: `project(xyz)`, `img_to_ground(pts)`, `rescaled(size)` (same aspect ratio only, otherwise
    `pipeline.FrameSizeError`), `save` and `load` (YAML).
- **New `vision/armvision/containers.py`**: the container finder.
  - Config dataclasses and `load_container_config`, reusing `config._section`, `_positive`, `_pair` and
    `ConfigError`.
  - `circle_fit`: algebraic fit plus Gauss-Newton refinement.
  - `split_outline` (rim / base / side points), `fit_container`, and `ContainerFinder.process(frame) -> ContainerState`.
  - `Container` (with `to_dict`) and `ContainerState.find(color)`.
- **New `vision/find_containers.py`**: command-line tool, following `demo_objects.py`.
  - `--calibrate`: click UI with prompts and a 4× zoom inset; `u` undoes. After the 4th click it draws a 50 mm grid,
    the sheet outline and the zone over the image, and prints the reprojection error, estimated FOV and camera height
    and distance so you can check them with a tape measure. `s` saves to `container_calibration.yaml`.
  - Live mode: camera overlay showing the projected base and rim outlines (a wrong height is visible immediately),
    the top-centre cross, a label, the zone and an optional top-down view. Accepts `--target red`, `--json` (one
    line per frame), `--no-gui`, and a still image as `--source` with `--save out.png`. Cameras open through
    `armvision.__main__.open_camera`.
- **New `vision/container_config.yaml`**:
  - `sheet`: `width_mm` 430, `mark_distance_mm` 400, optional `points`.
  - `arm`: `base_from_edge_mm`, `base_offset_mm`, `base_yaw_deg`.
  - `zone`: `x_mm` [150, 330], `y_mm` [−160, 160], `max_reach_mm` 330, `max_angle_deg` 30.
  - `containers`: `heights_mm` {}, `default_height_mm` 80, `max_height_mm` 200, `min_footprint_mm2` 1500.
  - `camera`: `fov_deg` null.
  - `tracking`: `gate_mm`, `ema`, `max_missed`, `min_age`.
- **New `vision/tests/test_containers.py`**: synthetic 3D scenes with no GUI, so it runs in the headless `armvision`
  CI job.
- **Edit `vision/README.md`** (new "Container finder (no markers)" section) and **`vision/.gitignore`** (add
  `container_calibration.yaml`). Existing pipeline modules are not changed.

## Output (arm base frame, mm)
```json
{"calibrated": true,
 "containers": [{"id": 1, "color": "red", "center_mm": [252.3, 58.1], "top_mm": [252.3, 58.1, 81.0],
                 "size_mm": [119.6, 119.6], "shape": "round", "yaw_deg": 0.0, "height_mm": 81.0,
                 "height_source": "auto", "in_zone": true, "age": 14}],
 "target": {"id": 1, "color": "red", ...}}
```
`top_mm` is the centre of the opening at rim height, so the later drop target is roughly `ik(x/1000, y/1000,
(H + clearance)/1000)`. The mm → m conversion matches what `planner_bridge.target_to_planner_coords` already does.

## Your one-time setup
1. Draw the 2 pen ticks. Measure the sheet width and where the base axis sits relative to the arm-end edge and the
   centre line, and enter them in `container_config.yaml`. Optionally add the measured height of each container.
2. Run `python find_containers.py --calibrate --source N`, click the 4 points, check the grid overlay, and press `s`.
   Redo this only if the camera gets bumped.
3. Run `python find_containers.py --source N --target red`.

## Verification
- `cd vision && .venv/bin/python -m pytest -q`: all existing tests plus the new ones pass. The new tests cover:
  - Reference points match the sheet and arm config.
  - Focal length within 1 % with no noise and within 4 % with 1 px noise; camera centre within 2 %; save/load and
    rescale to 1920×1080 give the same result.
  - With the configured H, the cylinder, tapered cup, wide bowl and 30° box give centre error < 4 mm and size error
    < 5 mm, with the right shape, and yaw within 3° for the box.
  - Auto-H for cylinder, cup and box: H within 8 mm and centre within 6 mm; the hidden-base bowl falls back to
    `"default"`.
  - With red and blue containers in view, `find("red")` returns the red one.
  - An out-of-zone container is flagged and `find` returns None.
  - An empty sheet and a grey shadow produce no detections.
  - `min_age` gating and stable ids across frames.
  - Config validation errors.
  - A still image through `find_containers.main([... "--no-gui", "--json"])` produces the expected JSON.
- On the real rig: after calibrating, put the red container at a centre position measured with a ruler from the
  sheet edges. Expect the reported centre within about 1 cm and the drawn rim outline to match the real rim; repeat
  at the zone corners. Then add a second container of another colour and check `--target red`.

## Known limits (go in the README)
- Only coloured containers are detected (no white, black or clear ones).
- Touching containers of the same colour merge into one.
- If the arm or the table near the sheet edge is the same colour as a container, the two can merge. Keep containers
  ≥ 5 cm from the side edges if the table is coloured.
- No lens-distortion model; the zone sits inside the 4 clicked points, so the error stays at a few mm.
- Turn off camera auto-framing (e.g. Center Stage on an iPhone used as a webcam).
