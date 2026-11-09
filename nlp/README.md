# natHacks26 – robot NLP + object grounding

Turns text like "grab that red screwdriver and put it in the box" into typed actions for a robot
arm/hand, and locates the named object in an image or video stream. See `../.claude/plan.md` for the design.

## Layout
All commands below run from this folder (`nlp/` in the repo root).
- `nlp/` (the Python package, `nlp/nlp/`) – stdlib-only parser (rules + typo correction), dialogue, pluggable fallback, dispatcher.
- `vision/` – HSV color matching, grounder, ONNX YOLO detector, stream smoothing (needs numpy/opencv/onnxruntime).
- `tests/`, `demo.py`

## Use
```python
from nlp import Processor, Dispatcher, PickUp, Home, Reply, Shake

proc, disp = Processor(), Dispatcher()
disp.register(PickUp, lambda a: arm.pick_up(a.target))   # a.target is an ObjectRef(noun, color, size, position)
disp.register(Home, lambda a: arm.home())
...
result = proc.process("pick up the red screwdriver and give it to me")
disp.dispatch(result.actions)   # raises KeyError before running anything if a handler is missing
```
Actions: `PickUp PutDown PutIn Give MoveRelative Home ReturnToUser ReturnToZero RotateJoint OpenHand CloseHand Shake Reply`.
`ReturnToUser` is "come back to me" (back to the hand-off pose); plain "come back" stays a relative move.
`Reply` carries text plus an optional `Shake` gesture. A target of `None` means "the held object".
Commands are all-or-nothing: if any clause isn't understood, nothing is executed and a `Reply` explains.

Fallback: `Processor(OllamaFallback("qwen2.5:1.5b"))` sends unparsed text to a local LLM. Its JSON output is
validated against the action schema and rejected if malformed.

## Grounding
```
pip install -e '.[vision]' && python -m vision.download      # once: ~290 MB into ./models
```
```python
from vision import Grounder
from vision.openvocab import ClipTextEncoder, YoloWorldDetector
enc = ClipTextEncoder("models/onnx/text_model.onnx", "models/tokenizer.json")
grounder = Grounder(YoloWorldDetector("models/yolov8s-worldv2.onnx", enc))
det = grounder.find(rgb_frame, action.target)   # Detection(label, bbox, score) or None
```
- **Open vocabulary:** YOLO-World v2 (ONNX) takes the noun as a runtime text prompt, encoded by CLIP's text tower
  (ONNX, cached per phrase). No PyTorch needed. ~0.3 s per frame on CPU, fine for once-per-command grounding;
  use `vision.track` + `TemporalSmoother` to follow the box cheaply afterwards. Uses CUDA/TensorRT when present,
  and falls back to CPU if an accelerator fails (CoreML currently fails on this model, so it is not used).
- **Colour** words are checked with HSV on the detected box, so "red bus" returns `None` for a blue bus.
- **Position/size** words (left, right, nearest, big, small…) choose among multiple matches.
- `YoloOnnxDetector` (closed-set COCO) and the colour-blob fallback (pure numpy, picks the largest connected region of that colour; no OpenCV needed) still exist for models without open vocab;
  the blob fallback is never used with an open-vocab detector.
- **License:** the YOLO-World weights are AGPL-3.0 (inherited from Ultralytics). Check this before distributing.

## Dev
```
python -m venv .venv
.venv/bin/pip install pytest numpy opencv-python-headless onnxruntime tokenizers   # or just pytest numpy for the core
.venv/bin/python -m pytest        # CI runs the same from this folder
```
