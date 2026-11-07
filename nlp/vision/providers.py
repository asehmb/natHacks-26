"""Pick the best available ONNX Runtime execution provider."""
from __future__ import annotations

# CoreML is deliberately excluded: it loads YOLO-World but fails at inference (verified on macOS, ORT 1.x).
PREFERENCE = ("TensorrtExecutionProvider", "CUDAExecutionProvider")


def select_providers(allow_accel: bool = True) -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    chosen = [p for p in PREFERENCE if allow_accel and p in avail]
    return chosen + ["CPUExecutionProvider"]
