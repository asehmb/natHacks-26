"""Fetch the models into vision/models (run once: python -m armvision.download [dir])."""
import sys

from .yolo import VISION_MODELS

FILES = [
    ("Instemic/yolo-world-onnx", "yolov8l-worldv2.onnx"),          # AGPL-3.0 (Ultralytics YOLOv8-World v2), 182 MB
    ("Instemic/yolo-world-onnx", "yolov8s-worldv2.onnx"),          # fast fallback, 51 MB
    ("Xenova/clip-vit-base-patch32", "onnx/text_model.onnx"),      # prompts for both stages
    ("Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx"),    # crop re-classifier
    ("Xenova/clip-vit-base-patch32", "tokenizer.json"),
]


def main(dest: str = str(VISION_MODELS)) -> None:
    from huggingface_hub import hf_hub_download

    for repo, name in FILES:
        print("->", hf_hub_download(repo, name, local_dir=dest))


if __name__ == "__main__":
    main(*sys.argv[1:2])
