"""Type commands, see the parsed actions. Optionally locate pick-up targets on a camera.

    python demo.py                          # parser only (no dependencies)
    python demo.py --ollama qwen2.5:1.5b    # add a local-LLM fallback
    python demo.py --camera 0 --openvocab           # needs the [vision] extras + python -m vision.download
"""
import argparse
import json

from nlp import OllamaFallback, PickUp, Processor


def build_grounder(model, openvocab):
    from vision import Grounder

    if openvocab:
        from vision.openvocab import ClipTextEncoder, YoloWorldDetector

        enc = ClipTextEncoder("models/onnx/text_model.onnx", "models/tokenizer.json")
        return Grounder(YoloWorldDetector("models/yolov8s-worldv2.onnx", enc))
    if model:
        from vision.detector import YoloOnnxDetector

        return Grounder(YoloOnnxDetector(model))
    return Grounder(None)


def show_camera(ref, source, grounder):
    import cv2

    from vision import track

    for frame, det in track(grounder, ref, source):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if det:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr, f"{ref.describe()} {det.score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("grounding (q to quit)", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", metavar="MODEL", help="enable local LLM fallback via Ollama")
    ap.add_argument("--camera", help="camera index or video path for grounding pick-up targets")
    ap.add_argument("--model", help="closed-set YOLOv8 ONNX model path")
    ap.add_argument("--openvocab", action="store_true", help="use YOLO-World (run python -m vision.download first)")
    args = ap.parse_args()

    grounder = build_grounder(args.model, args.openvocab) if args.camera is not None else None
    proc = Processor(OllamaFallback(args.ollama) if args.ollama else None)
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        result = proc.process(text)
        print(f"[{result.source}]")
        for a in result.actions:
            print(" ", json.dumps(a.to_dict()))
        if args.camera is not None:
            for a in result.actions:
                if isinstance(a, PickUp):
                    src = int(args.camera) if args.camera.isdigit() else args.camera
                    show_camera(a.target, src, grounder)


if __name__ == "__main__":
    main()
