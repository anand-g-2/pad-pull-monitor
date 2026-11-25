"""
webapp/app.py

Web interface for the disc-pad pull monitor: Overview, Live Monitoring
(annotated MJPEG stream + Start/Stop), Analytics and Calibration (assign the
camera, draw the ROI). It has the same layout and style as
powder_line_vision's webapp.

Start always runs on the camera assigned in Calibration, with the ROI saved
there and the newest trained model. The monitor runs in a background thread
of this process (webapp/runner.py). Every judged batch goes into
webapp/analytics.db, and NC batches also get a snapshot in webapp/snapshots/.

Run from the pad_pull_monitor folder:
    python -m webapp.app
Then open http://localhost:5050

No login: by default it listens on this PC only (127.0.0.1). Pass
--host 0.0.0.0 to open it to the network, but only on a trusted network.

Testing without the line camera:
    python -m webapp.app --test-video PATH_TO_VIDEO
makes "the camera" that video, played at its own speed and looped.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from datetime import date

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from . import camera, store
from .calibration_routes import calibration_bp, load_roi_info
from .runner import SNAPSHOT_DIR, StartError, model_label, newest_model, runner

WATCH_POLL_SECONDS = 0.04
MAX_TABLE_ROWS = 500

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.register_blueprint(calibration_bp, url_prefix="/api/calibration")
store.init_db()


def _gpu_name() -> str | None:
    # nvidia-smi answers in well under a second. "import torch" takes ~2 min
    # cold on this laptop and would block the server from starting.
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


GPU_NAME = _gpu_name()


@app.route("/")
def index():
    return render_template("index.html", gpu_name=GPU_NAME)


# ----------------------------------------------------------------- monitor control
@app.route("/api/config")
def config():
    """What Start will use: the assigned camera, the saved ROI, the model."""
    return jsonify(camera_index=camera.load_assigned(), test_video=camera.TEST_VIDEO,
                   roi=load_roi_info(), model=model_label(newest_model()), gpu=GPU_NAME)


@app.route("/api/monitor/start", methods=["POST"])
def start():
    try:
        return jsonify(runner.start())
    except StartError as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/monitor/stop", methods=["POST"])
def stop():
    return jsonify(runner.stop())


@app.route("/api/monitor/status")
def status():
    return jsonify(runner.status())


@app.route("/api/monitor/ack", methods=["POST"])
def acknowledge():
    batch = (request.get_json(silent=True) or {}).get("batch")
    if not isinstance(batch, int):
        return jsonify(error="expected {'batch': <int>}"), 400
    return jsonify(acknowledged=runner.acknowledge(batch))


@app.route("/api/watch")
def watch():
    def generate():
        last = -1
        while True:
            frame, seq = runner.frame, runner.seq
            if frame is not None and seq != last:
                last = seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(WATCH_POLL_SECONDS)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshots/<path:name>")
def snapshot(name):
    return send_from_directory(SNAPSHOT_DIR, name)


# ----------------------------------------------------------------- dashboards
@app.route("/api/overview")
def overview():
    return jsonify(store.overview(date.today()))


def _parse_date(value: str | None, param: str) -> date:
    if not value:
        raise ValueError(f"missing query param '{param}'")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"'{param}' must be YYYY-MM-DD, got {value!r}") from None


def _analytics_from_args() -> dict:
    range_ = request.args.get("range", "day")
    if range_ == "custom":
        return store.analytics("custom", from_date=_parse_date(request.args.get("from"), "from"),
                               to_date=_parse_date(request.args.get("to"), "to"))
    ref = _parse_date(request.args.get("date"), "date") if request.args.get("date") else date.today()
    mode = request.args.get("mode", "daily")
    if mode not in ("daily", "hourly_avg"):
        raise ValueError(f"unknown mode {mode!r}")
    return store.analytics(range_, ref_date=ref, mode=mode)


@app.route("/api/analytics")
def analytics():
    try:
        data = _analytics_from_args()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    data["rows_total"] = len(data["rows"])
    data["rows"] = data["rows"][:MAX_TABLE_ROWS]
    return jsonify(data)


@app.route("/api/analytics.csv")
def analytics_csv():
    try:
        data = _analytics_from_args()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    buf = io.StringIO()
    cols = ["ts", "source", "batch_no", "pads_in", "pulled", "leftover", "result", "reason", "snapshot"]
    w = csv.writer(buf)
    w.writerow(cols)
    for r in data["rows"]:
        w.writerow([r[c] for c in cols])
    name = f"pad_batches_{data['from']}_to_{data['to']}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--test-video", help="use this video file instead of the camera (testing only)")
    a = ap.parse_args()
    if a.test_video:
        camera.set_test_video(a.test_video)
        print(f"TEST MODE: the camera is replaced by {a.test_video}")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)
