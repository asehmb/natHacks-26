import numpy as np
import pytest

from Vision import camera


@pytest.mark.parametrize("source,expected", [("0", 0), (1, 1), (" 2 ", 2),
                                             ("http://example.test/video", "http://example.test/video"),
                                             ("video.mp4", "video.mp4")])
def test_source_parsing(source, expected):
    assert camera.parse_source(source) == expected


@pytest.mark.parametrize("source", [True, -1, "", None])
def test_bad_source(source):
    with pytest.raises(ValueError, match="source"):
        camera.parse_source(source)


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True

    def set(self, key, value):
        self.properties[key] = value


def test_mac_uses_avfoundation_for_index(monkeypatch):
    fake = FakeCapture()
    calls = []

    def factory(source, backend):
        calls.append((source, backend))
        return fake

    monkeypatch.setattr(camera.sys, "platform", "darwin")
    monkeypatch.setattr(camera.cv2, "VideoCapture", factory)
    assert camera.open_camera("1", resolution=[640, 480]) is fake
    assert calls == [(1, camera.cv2.CAP_AVFOUNDATION)]
    assert fake.properties[camera.cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert fake.properties[camera.cv2.CAP_PROP_FRAME_HEIGHT] == 480


def test_stream_uses_auto_backend_and_failed_open_releases(monkeypatch):
    fake = FakeCapture(opened=False)
    calls = []

    def factory(source, backend):
        calls.append((source, backend))
        return fake

    monkeypatch.setattr(camera.sys, "platform", "darwin")
    monkeypatch.setattr(camera.cv2, "VideoCapture", factory)
    with pytest.raises(RuntimeError, match="Cannot open"):
        camera.open_camera("http://example.test/video")
    assert fake.released
    assert calls == [("http://example.test/video", camera.cv2.CAP_ANY)]


def test_read_failures_and_success():
    class Reader:
        def __init__(self, result):
            self.result = result

        def read(self):
            return self.result

    with pytest.raises(RuntimeError, match="no frame"):
        camera.read_frame(Reader((False, None)))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert camera.read_frame(Reader((True, frame))) is frame
