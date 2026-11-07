# Plan: Lightweight NLP command processor + object grounding

## Context
Robot-arm/hand project (natHacks26). Repo is empty. We need a light, portable Python NLP that turns utterances like
"grab that red screwdriver" into structured actions the Python action layer executes, and grounds the named object
(noun + attributes like color) in an image/video stream.

Decisions: parser = rules + fuzzy match + pluggable fallback. Vision and output format were not specified, so I'm using the
defaults below (easy to swap).

## Layout
Everything lives in the repo's `nlp/` project folder (the Python package is `nlp/nlp/`); the GitHub Actions
workflow is at `.github/workflows/tests.yml`.
```
nlp/                # project folder
  nlp/              # python package
  actions.py     # dataclasses: ObjectRef(noun, color, size, ...), Action types
  lexicon.py     # verb/noun/color synonyms, number words, direction phrases
  parser.py      # rule parser -> list[Action]; fuzzy match via rapidfuzz (stdlib difflib fallback)
  fallback.py    # Fallback protocol; NullFallback (default), OllamaFallback (optional, JSON output validated against action schema)
  dialogue.py    # yes/no + chit-chat ("did you have a good day") -> Reply(text, gesture)
  dispatch.py    # Action -> handler registry; user's robot code registers handlers
vision/
  grounding.py   # Grounder protocol: find(frame, ObjectRef) -> Detection(bbox, score)
  hsv.py         # color filter (red wraps hue range)
  detector.py    # ONNX Runtime open-vocab detector (YOLO-World style); closed-set YOLO as alt backend
  stream.py      # OpenCV VideoCapture loop, temporal smoothing across frames
tests/           # parser table tests, grounding color tests on synthetic images
README.md
pyproject.toml   # core deps: rapidfuzz only; extras: [vision] onnxruntime opencv-python numpy, [llm] httpx
```

## Parser design
1. Normalize (lowercase, strip fillers "please", "that", "the", "a").
2. Split into clauses on "and", "then", "and then".
3. Per clause match intent by verb synonyms, with fuzzy matching for typos/ASR errors
   (grab/pick up/take -> PICK_UP; put/place/drop in Y -> PUT_IN; give me/hand me -> GIVE;
   come closer/back up/go back a bit -> MOVE_RELATIVE(dir, amount: "a bit"/"a lot"/number);
   home/return to 0 -> HOME; rotate joint N -> ROTATE_JOINT; open/close hand; shake yes/no/maybe).
4. Extract ObjectRef: color + noun (+ size/position words like "left"); resolve pronoun "it" to the previous clause's object.
5. Compound "pick up X and give it to me" -> [PickUp(X), Give(it)]; "pick up X and put in Y" -> [PickUp(X), PutIn(Y)].
6. Yes/no questions and chit-chat -> dialogue module (canned replies plus gesture like ShakeYes/ShakeNo/ShakeMaybe).
7. Low confidence / no rule match -> Fallback. Default NullFallback returns "didn't understand". Optional local LLM fallback
   must return JSON that is validated against the Action schema, never executed unvalidated.

Maps to the main calls: pick_up, put_down, home, return_to_zero, rotate_joint(n), open_hand, close_hand, shake_yes, shake_no, shake_maybe (shrug).

## Grounding
Parser emits ObjectRef; `Grounder.find(frame, ref)` runs the open-vocab detector on the noun, filters/re-scores by HSV for the color,
and returns the best box. Stream loop smooths over frames. Returns pixel bbox plus score; mapping to arm coordinates is left to the action layer.

## Verification
- `pytest`: table of ~40 utterances -> expected action lists (all examples in your list, typos, compound commands, "it" resolution).
- Fallback test with a stub fallback; invalid JSON is rejected.
- Color grounding test on synthetic images (red vs blue rectangle, red hue wraparound).
- Demo script: type a command, print parsed actions; with `--camera`, overlay the detected box on a live stream.
- Check it runs on CPU only with core install (no vision extras) for the parser.
