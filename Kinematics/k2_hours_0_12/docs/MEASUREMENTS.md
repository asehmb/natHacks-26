# K2 measurement worksheet (hours 0-12)

Use a ruler/caliper and the actual assembled arm. Have a teammate independently
check each measurement. Keep missing dimensions marked MEASURE ME in YAML.
Invented example dimensions are not specifications from the Mk2 project page.

| YAML field | Definition | Measured value | Checked by |
| --- | --- | --- | --- |
| links.lengths.upper_arm | Shoulder-to-elbow pivot distance, m | | |
| links.lengths.forearm | Elbow-to-wrist pivot distance, m | | |
| links.lengths.tool | Wrist pivot to grasp point, m | | |
| links.offsets.shoulder_xyz | Shoulder position in base frame, m | | |
| links.offsets.wrist_xyz | Additional forearm-end-frame offset, m | | |
| links.offsets.tool_xyz | Additional tool-frame offset, m | | |
| camera.resolution | Actual Python frame width/height in pixels | | |

Base frame: origin on the base rotation axis at table height; right-handed,
x forward, y left, z up. Lengths m; angles rad. Agree the model joint zero poses
and positive directions with K1. Do not count link length again as an offset.

Record each servo's label/model and its installed joint. Confirm whether the
assembled mechanism actually has a separate wrist actuator before enabling that
requested slot. Servo pulse limits and shaft ranges are measured calibration
data, not guaranteed by a generic model label.

Hours 0-12 finish: fill available geometry, confirm a colored block on a live
Python image, run the mock locally, and request F2's packet adapter. Servo fitting,
table homography, depth, and physical motion verification follow later.
