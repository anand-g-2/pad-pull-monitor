"""
webapp/camera_session.py

The Calibration tab's live preview: this module opens, reads, and releases
the camera. It is based on powder_line_vision's camera_session.py, with one
difference: there is only one camera.

A camera can only be opened once, so the preview and the monitor never hold it
at the same time:
  * runner.start() closes this session before it opens the camera;
  * open_session() refuses while the monitor is running (checked in
    calibration_routes.py).
A session nobody reads from (e.g. the browser tab was closed) is released by
a background sweeper after IDLE_TIMEOUT_SECONDS.
"""
from __future__ import annotations

import threading
import time

from . import camera

IDLE_TIMEOUT_SECONDS = 60.0

_lock = threading.Lock()
_session: dict | None = None        # {"index", "cap", "last_used", "width", "height"}


def _sweep_idle_locked() -> None:
    global _session
    if _session and time.monotonic() - _session["last_used"] > IDLE_TIMEOUT_SECONDS:
        _session["cap"].release()
        _session = None


def _sweeper() -> None:
    while True:
        time.sleep(15)
        with _lock:
            _sweep_idle_locked()


threading.Thread(target=_sweeper, daemon=True).start()


def open_session(index: int) -> dict:
    global _session
    with _lock:
        _sweep_idle_locked()
        if _session and _session["index"] != index:
            _session["cap"].release()
            _session = None
        if _session is None:
            cap = camera.open_camera(index)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"could not open camera index {index} - is it plugged in, "
                                   "or open in another program?")
            w, h = camera.frame_size(cap)
            _session = {"index": index, "cap": cap, "width": w, "height": h}
        _session["last_used"] = time.monotonic()
        return {"width": _session["width"], "height": _session["height"]}


def read_frame():
    with _lock:
        if _session is None:
            return None
        _session["last_used"] = time.monotonic()
        ok, frame = _session["cap"].read()
    return frame if ok else None


def is_open() -> bool:
    with _lock:
        return _session is not None


def close_session() -> None:
    global _session
    with _lock:
        if _session is not None:
            _session["cap"].release()
            _session = None
