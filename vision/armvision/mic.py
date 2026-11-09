"""Push-to-talk speech input: Enter starts recording, Enter stops, faster-whisper transcribes offline.

    pip install sounddevice faster-whisper        # the model (~140 MB for base.en) downloads on first use

    mic = PushToTalk()                  # loads the model once
    text = mic.listen(input)            # blocks: Enter, speak, Enter -> "grab this item"
"""
from __future__ import annotations

import sys
import threading
import types
from typing import Callable, Optional

import numpy as np

# faster-whisper imports PyAV only to decode audio files; we hand it samples. PyAV bundles its own libavdevice, which
# clashes with OpenCV's copy (duplicate AVFoundation classes, camera crashes on macOS), so it is never loaded.
sys.modules.setdefault("av", types.ModuleType("av"))

RATE = 16000                            # Whisper's sample rate
# Biases recognition towards the command vocabulary ("blue" not "blew", "container" not "contain her").
PROMPT = ("Robot arm commands: grab this item, put it in the blue container, put it in the orange container, "
          "red, green, come back to me, open your hand, close your hand, go home.")


class MicError(RuntimeError):
    """No usable audio (no device, permission denied, or silence)."""


class PushToTalk:
    def __init__(self, model: str = "base.en", device: Optional[int | str] = None, max_s: float = 15.0):
        import sounddevice
        from faster_whisper import WhisperModel
        self._sd, self.device, self.max_s = sounddevice, device, max_s
        self._model = WhisperModel(model, device="cpu", compute_type="int8")

    def record(self, wait_stop: Callable[[], object]) -> np.ndarray:
        """Record until wait_stop() returns (only the first max_s seconds are kept). Float32 mono at 16 kHz."""
        chunks, stop = [], threading.Event()

        def on_audio(data, frames, time_info, status):
            if sum(len(c) for c in chunks) < self.max_s * RATE:     # keep only the first max_s seconds
                chunks.append(data[:, 0].copy())

        try:
            stream = self._sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=self.device,
                                          callback=on_audio)
        except Exception as exc:
            raise MicError(f"can't open the microphone: {exc}") from exc
        with stream:
            waiter = threading.Thread(target=lambda: (wait_stop(), stop.set()), daemon=True)
            waiter.start()
            stop.wait()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        if audio.size < RATE * 0.3:
            raise MicError("too short: hold the recording for at least half a second")
        if float(np.abs(audio).max()) < 1e-4:
            raise MicError("only silence: allow your terminal in System Settings > Privacy & Security > Microphone, "
                           "then restart it")
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language="en", beam_size=5, initial_prompt=PROMPT,
                                             vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def listen(self, wait_stop: Callable[[], object]) -> str:
        return self.transcribe(self.record(wait_stop))
