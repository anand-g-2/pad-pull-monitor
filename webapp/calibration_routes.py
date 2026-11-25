"""
webapp/calibration_routes.py

Backend for the Calibration tab:
  * find the cameras on this PC and assign one to the line (camera.json)
  * a live MJPEG preview of the assigned camera
  * capture a still frame and save the hand-drawn ROI polygon (roi.json)

roi.json keeps the format monitor.py / draw_roi.py already read: "polygon" is a
list of [x, y] fractions of the frame (0..1). Every save also keeps:
  * a timestamped copy of the previous roi.json in roi_backups/
  * roi_reference.jpg, the frame the ROI was drawn on, so you can later check
    whether the camera has moved
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime

import cv2
from flask import Blueprint, Response, jsonify, request

from . import camera, camera_session
from .runner import PROJECT_DIR, runner

calibration_bp = Blueprint("calibration", __name__)

ROI_FILE = PROJECT_DIR / "roi.json"
ROI_BACKUP_DIR = PROJECT_DIR / "roi_backups"
ROI_REFERENCE = PROJECT_DIR / "roi_reference.jpg"
PREVIEW_JPEG_QUALITY = 80

_captured = {"frame": None}            # last captured still, saved as roi_reference.jpg
_captured_lock = threading.Lock()


def _monitor_busy():
    if runner.is_running():
        return jsonify(error="Monitoring is running and is using the camera. "
                             "Stop it on the Live Monitoring page first."), 409
    return None


def _assigned_or_error():
    index = camera.load_assigned()
    if index is None:
        return None, (jsonify(error="no camera assigned yet - detect cameras and assign one"), 400)
    return index, None


def load_roi_info() -> dict:
    if not ROI_FILE.exists():
        return {"exists": False}
    try:
        data = json.loads(ROI_FILE.read_text())
        return {"exists": True, "polygon": data["polygon"], "saved_at": data.get("saved_at"),
                "camera_index": data.get("camera_index"), "frame_size": data.get("frame_size")}
    except (ValueError, KeyError, OSError) as exc:
        return {"exists": False, "error": f"roi.json is unreadable: {exc}"}


# ----------------------------------------------------------------- camera assignment
@calibration_bp.route("/cameras")
def cameras():
    if (busy := _monitor_busy()) is not None:
        return busy
    camera_session.close_session()      # an open preview would show as "unavailable"
    return jsonify(cameras=camera.probe_cameras(), test_mode=bool(camera.TEST_VIDEO))


@calibration_bp.route("/device", methods=["GET"])
def get_device():
    return jsonify(camera_index=camera.load_assigned(), test_mode=bool(camera.TEST_VIDEO))


@calibration_bp.route("/device", methods=["POST"])
def set_device():
    if (busy := _monitor_busy()) is not None:
        return busy
    index = (request.get_json(silent=True) or {}).get("camera_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        return jsonify(error="camera_index must be a non-negative integer"), 400
    camera_session.close_session()
    camera.save_assigned(index)
    return jsonify(camera_index=index)


# ----------------------------------------------------------------- preview session
@calibration_bp.route("/session/open", methods=["POST"])
def open_session():
    if (busy := _monitor_busy()) is not None:
        return busy
    index, err = _assigned_or_error()
    if err:
        return err
    try:
        info = camera_session.open_session(index)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(camera_index=index, **info)


@calibration_bp.route("/session/close", methods=["POST"])
def close_session():
    camera_session.close_session()
    return "", 204


@calibration_bp.route("/preview")
def preview():
    # The session is NOT closed when this stream disconnects: the page drops
    # the stream every time a frame is captured for drawing, and closing then
    # would leave "back to live" with no camera. The camera is released when
    # the page is left (session/close), when monitoring starts, when cameras
    # are probed, or after camera_session.IDLE_TIMEOUT_SECONDS without reads.
    def generate():
        while True:
            frame = camera_session.read_frame()
            if frame is None:
                break
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@calibration_bp.route("/capture", methods=["POST"])
def capture():
    frame = camera_session.read_frame()
    if frame is None:
        return jsonify(error="the camera preview is not open - reopen the Calibration page"), 409
    with _captured_lock:
        _captured["frame"] = frame
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return jsonify(error="failed to encode the frame"), 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ----------------------------------------------------------------- ROI
@calibration_bp.route("/roi", methods=["GET"])
def get_roi():
    return jsonify(load_roi_info())


@calibration_bp.route("/roi", methods=["POST"])
def save_roi():
    body = request.get_json(silent=True) or {}
    poly = body.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        return jsonify(error="the ROI needs at least 3 corners"), 400
    try:
        points = [[round(float(x), 4), round(float(y), 4)] for x, y in poly]
    except (TypeError, ValueError):
        return jsonify(error="polygon must be a list of [x, y] pairs"), 400
    if not all(0.0 <= v <= 1.0 for p in points for v in p):
        return jsonify(error="polygon corners must lie inside the frame"), 400

    with _captured_lock:
        frame = _captured["frame"]
    if ROI_FILE.exists():
        ROI_BACKUP_DIR.mkdir(exist_ok=True)
        shutil.copy2(ROI_FILE, ROI_BACKUP_DIR / f"roi_{datetime.now():%Y%m%d_%H%M%S}.json")
    data = {"polygon": points, "camera_index": camera.load_assigned(),
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
            "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds")}
    ROI_FILE.write_text(json.dumps(data, indent=1))
    if frame is not None:
        cv2.imwrite(str(ROI_REFERENCE), frame)
    return jsonify(load_roi_info())
