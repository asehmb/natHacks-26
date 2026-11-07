"""Fetch the open-vocabulary models into ./models (run once: python -m vision.download)."""
import sys

FILES = [
    ("Instemic/yolo-world-onnx", "yolov8s-worldv2.onnx"),  # AGPL-3.0 (Ultralytics YOLOv8-World v2)
    ("Xenova/clip-vit-base-patch32", "onnx/text_model.onnx"),
    ("Xenova/clip-vit-base-patch32", "tokenizer.json"),
]


def main(dest: str = "models") -> None:
    from huggingface_hub import hf_hub_download

    for repo, name in FILES:
        print("->", hf_hub_download(repo, name, local_dir=dest))


if __name__ == "__main__":
    main(*sys.argv[1:2])
