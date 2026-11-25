"""
webapp/camera.py

Opening and assigning the line camera, based on powder_line_vision's
src/calibration/camera_devices.py.

* DirectShow on Windows: it opens a camera in ~1-3 s, where Media Foundation
  can take up to 25 s per index. Probe, calibration preview and inference all
  use the same backend, because the two backends number cameras differently.
* 1920x1080 MJPG is requested on every open, so the ROI is drawn on the same
  frames inference runs on. The ROI is stored as fractions of the frame, so a
  camera that delivers another 16:9 size still lines up.
* The assigned camera index lives in camera.json in the project folder.

Test mode: `python -m webapp.app --test-video FILE` makes "the camera" a video
file (played at its own frame rate), so the whole UI can be tried without the
line camera. It never shows up in the UI as a choice.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
CAMERA_FILE = PROJECT_DIR / "camera.json"

CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080
PROBE_MAX_INDEX = 6

TEST_VIDEO: str | None = None          # set by app.py --test-video


def set_test_video(path: str | None) -> None:
    global TEST_VIDEO
    TEST_VIDEO = path


class PacedCapture:
    """Wraps a video file so read() delivers frames at the file's own frame
    rate, the way a camera would, and loops at the end (test mode only)."""

    def __init__(self, path: str):
        self.path, self.cap = path, cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self._next = time.perf_counter()

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + 1 / self.fps, time.perf_counter() - 1)
        ok, frame = self.cap.read()
        if not ok:                                   # loop the test video
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            ok, frame = self.cap.read()
        return ok, frame

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.cap.release()


def open_camera(index: int):
    if TEST_VIDEO:
        return PacedCapture(TEST_VIDEO)
    cap = cv2.VideoCapture(index, CAMERA_BACKEND)
    if cap.isOpened():
        # size first, then MJPG: DirectShow ignores MJPG set before the size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame, not a backlog (ignored by some drivers)
    return cap


def frame_size(cap) -> tuple[int, int]:
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def probe_cameras(max_index: int = PROBE_MAX_INDEX) -> list[dict]:
    """Opens indices 0..max_index-1 one at a time and returns those that
    deliver a frame, each with a small thumbnail so the operator can tell the
    cameras apart. A camera already held open (by the preview or the running
    monitor) reports as unavailable."""
    if TEST_VIDEO:
        indices = [0]
    else:
        indices = range(max_index)
    found = []
    for index in indices:
        cap = open_camera(index)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            w, h = frame_size(cap)
            ok_jpg, buf = cv2.imencode(".jpg", cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA))
            found.append({"index": index, "width": w, "height": h,
                          "label": "Test video" if TEST_VIDEO else f"Camera index {index}",
                          "thumbnail_b64": base64.b64encode(buf.tobytes()).decode() if ok_jpg else None})
        finally:
            cap.release()
    return found


def load_assigned() -> int | None:
    if not CAMERA_FILE.exists():
        return None
    try:
        value = json.loads(CAMERA_FILE.read_text()).get("camera_index")
        return value if isinstance(value, int) else None
    except (ValueError, OSError):
        return None


def save_assigned(index: int) -> None:
    CAMERA_FILE.write_text(json.dumps({"camera_index": index}, indent=2))
