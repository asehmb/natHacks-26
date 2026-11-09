import importlib.util
import subprocess
import sys
import types

import numpy as np
import pytest

pytest.importorskip("sounddevice")
if importlib.util.find_spec("faster_whisper") is None:   # not importorskip: that would load the real PyAV here
    pytest.skip("faster-whisper not installed", allow_module_level=True)
from armvision import mic as micmod  # noqa: E402


class FakeStream:
    """Delivers `audio` to the callback in 1600-sample blocks as soon as the stream starts."""

    def __init__(self, audio, callback, **kw):
        self.audio, self.callback = audio, callback

    def __enter__(self):
        for i in range(0, len(self.audio), 1600):
            self.callback(self.audio[i:i + 1600, None], 1600, None, None)
        return self

    def __exit__(self, *exc):
        return False


def make(audio, max_s=15.0):
    m = micmod.PushToTalk.__new__(micmod.PushToTalk)
    m._sd = types.SimpleNamespace(InputStream=lambda **kw: FakeStream(audio, **kw))
    m.device, m.max_s = None, max_s
    return m


def test_records_until_stop_and_caps_length():
    tone = (0.1 * np.sin(np.arange(16000 * 3) / 5)).astype(np.float32)
    assert make(tone).record(lambda: None).size == 48000
    assert make(tone, max_s=1.0).record(lambda: None).size == 16000


@pytest.mark.parametrize("audio,match", [
    (np.zeros(16000, np.float32), "Privacy"),        # permission denied on macOS gives pure zeros
    (np.full(1600, 0.1, np.float32), "too short"),
])
def test_unusable_audio_raises(audio, match):
    with pytest.raises(micmod.MicError, match=match):
        make(audio).record(lambda: None)


def test_pyav_is_never_loaded():
    code = "import cv2, armvision.mic, faster_whisper, sys; print(hasattr(sys.modules['av'], 'open'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False" and "implemented in both" not in out.stderr
