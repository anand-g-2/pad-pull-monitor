"""
webapp/runner.py

Runs PadPullMonitor (../monitor.py) on the assigned line camera in a
background thread inside the web app. Start always uses:
  * the camera assigned on the Calibration tab   (camera.json)
  * the ROI drawn on the Calibration tab          (roi.json)
  * the newest trained YOLO model                (runs/*/weights/best.pt)

Each frame is annotated and kept as the latest JPEG for the MJPEG stream.
Every judged batch goes to the SQLite store, and NC batches also get a
snapshot. If the camera stops delivering frames, the runner tries to reopen it
for RECONNECT_WINDOW_S before giving up with an error.

Only one monitor runs at a time (one camera, one belt).
"""
from __future__ import annotations

import glob
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent           # pad_pull_monitor/
sys.path.insert(0, str(PROJECT_DIR))

import monitor  # noqa: E402  (needs the path above)

from . import camera, camera_session, store  # noqa: E402

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
ROI_FILE = PROJECT_DIR / "roi.json"
JPEG_QUALITY = 80
READ_FAILS_BEFORE_RECONNECT = 15
RECONNECT_WINDOW_S = 30.0
MIN_OK_FPS = 8


def newest_model() -> str | None:
    """Newest runs/*/weights/best.pt, or the best.engine (TensorRT) next to it
    if one was built from it (export_tensorrt.py) - an engine older than its
    best.pt is stale and ignored."""
    found = sorted(glob.glob(str(PROJECT_DIR / "runs" / "**" / "weights" / "best.pt"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    if not found:
        return None
    pt = Path(found[0])
    engine = pt.with_suffix(".engine")
    if engine.exists() and engine.stat().st_mtime >= pt.stat().st_mtime:
        return str(engine)
    return str(pt)


def model_label(path: str | None) -> str | None:
    if not path:
        return None
    name = Path(path).parent.parent.name
    return f"{name} (TensorRT)" if path.endswith(".engine") else name


class StartError(Exception):
    pass


class MonitorRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frame: bytes | None = None
        self.seq = 0
        self._status = self._idle_status()
        self._writes: queue.Queue = queue.Queue()
        threading.Thread(target=self._writer, daemon=True).start()

    @staticmethod
    def _idle_status() -> dict:
        return {"state": "idle", "camera_index": None, "model": None, "started_at": None,
                "error": None, "notice": None, "monitor_state": None, "pads_in_roi": 0,
                "puller_score": 0.0, "batches": 0, "pads": 0, "pulled": 0, "nc": 0,
                "fps": 0.0, "events": [], "last_nc": None}

    # ---------------------------------------------------------------- control
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> dict:
        with self._lock:
            if self.is_running():
                raise StartError("monitoring is already running")
            index = camera.load_assigned()
            if index is None:
                raise StartError("No camera assigned. Assign one on the Calibration page.")
            if not ROI_FILE.exists():
                raise StartError("No ROI saved. Draw and save it on the Calibration page.")
            model = newest_model()
            if model is None:
                raise StartError("No trained model found in runs/*/weights/best.pt.")

            camera_session.close_session()       # the calibration preview must let go of the camera
            self._stop.clear()
            self.frame = None
            self._status = self._idle_status()
            self._status.update(state="starting", camera_index=index, model=model_label(model),
                                started_at=datetime.now().isoformat(sep=" ", timespec="seconds"))
            self._thread = threading.Thread(target=self._run, daemon=True, args=(index, model))
            self._thread.start()
            return dict(self._status)

    def stop(self) -> dict:
        self._stop.set()
        t = self._thread
        if t:
            t.join(timeout=10)
        with self._lock:
            if self._status["state"] in ("running", "starting", "reconnecting"):
                self._status["state"] = "stopped"
            return dict(self._status)

    def status(self) -> dict:
        with self._lock:
            s = dict(self._status)
            s["events"] = list(s["events"])
            return s

    def acknowledge(self, batch_no: int) -> bool:
        with self._lock:
            nc = self._status.get("last_nc")
            if nc and nc["batch"] == batch_no:
                nc["acknowledged"] = True
                return True
            return False

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _event(self, text: str):
        with self._lock:
            self._status["events"] = ([{"t": datetime.now().strftime("%H:%M:%S"), "text": text}]
                                      + self._status["events"])[:50]

    # ---------------------------------------------------------------- worker
    def _open(self, index):
        cap = camera.open_camera(index)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    @staticmethod
    def _measure_fps(cap) -> float:
        """The monitor's timings (stable count, settle, 2 s post-pull check)
        are counted in frames, so it needs the camera's REAL frame rate. Many
        drivers report 0 or a nominal value, so time ~2 s of reads instead."""
        for _ in range(3):                     # the first frames after opening are often slow
            cap.read()
        n, t0 = 0, time.perf_counter()
        while n < 60 and time.perf_counter() - t0 < 2.0:
            ok, _ = cap.read()
            n += bool(ok)
        elapsed = time.perf_counter() - t0
        return min(max(n / elapsed, 0.5), 60.0) if n else 15.0

    def _reconnect(self, index):
        """Camera stopped delivering frames: retry until it is back, the
        window runs out, or Stop is pressed."""
        self._set(state="reconnecting", notice=f"Camera {index} stopped sending frames - reconnecting…")
        self._event(f"camera {index} lost - reconnecting")
        deadline = time.monotonic() + RECONNECT_WINDOW_S
        while time.monotonic() < deadline and not self._stop.is_set():
            cap = self._open(index)
            if cap is not None:
                ok, _ = cap.read()
                if ok:
                    self._set(state="running", notice=None)
                    self._event(f"camera {index} reconnected")
                    return cap
                cap.release()
            self._stop.wait(2.0)
        return None

    def _run(self, index, model):
        cap = None
        try:
            cfg = monitor.Cfg()
            roi = monitor.roi_mask(str(ROI_FILE))
            det = monitor.YoloDetector(model, roi, cfg)     # loading the model can take a while cold

            cap = self._open(index)
            if cap is None:
                raise RuntimeError(f"could not open camera {index} - is it plugged in, "
                                   "or open in another program?")
            fps = self._measure_fps(cap)
            if fps < MIN_OK_FPS:
                self._set(notice=f"Camera delivers only {fps:.1f} fps (need ~{MIN_OK_FPS}+). Check its "
                                 "resolution/format and USB port. Results may be unreliable.")
                self._event(f"WARNING: camera delivers only {fps:.1f} fps")
            mon = monitor.PadPullMonitor(det, roi, fps, cfg)
            run_id = uuid.uuid4().hex
            label = f"Camera {index}" if not camera.TEST_VIDEO else f"Test video ({Path(camera.TEST_VIDEO).name})"
            self._set(state="running")

            fails, t_fps, n_fps = 0, time.perf_counter(), 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails < READ_FAILS_BEFORE_RECONNECT:
                        continue
                    cap.release()
                    cap = self._reconnect(index)
                    if cap is None:
                        if self._stop.is_set():
                            break
                        raise RuntimeError(f"camera {index} did not come back within "
                                           f"{RECONNECT_WINDOW_S:.0f} s - check the cable, then press Start")
                    fails = 0
                    continue
                fails = 0

                n_before = len(mon.batches)
                events = mon.update(frame)
                vis = mon.draw(frame)
                ok_jpg, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok_jpg:
                    with self._lock:
                        self.frame, self.seq = jpg.tobytes(), self.seq + 1
                for b in mon.batches[n_before:]:
                    self._record(run_id, label, b, vis)

                n_fps += 1
                now = time.perf_counter()
                with self._lock:
                    s = self._status
                    s.update(monitor_state=mon.state, pads_in_roi=len(mon.d.pads),
                             puller_score=round(float(mon.d.puller_score), 2),
                             batches=len(mon.batches), pads=mon.total_pads,
                             pulled=mon.total_pulled, nc=mon.nc_count)
                    if now - t_fps >= 1.0:
                        rate = n_fps / (now - t_fps)
                        s["fps"], t_fps, n_fps = round(rate, 1), now, 0
                        # keep the monitor's frame-counted windows in real seconds,
                        # whether the camera or the model is the bottleneck
                        mon.fps = 0.7 * mon.fps + 0.3 * max(rate, 0.5)
                for e in events:
                    self._event(e.strip())

            if mon.state != mon.APPROACH:
                self._event(f"stopped during {mon.state} - batch {len(mon.batches) + 1} not judged")
            with self._lock:
                if self._status["state"] in ("running", "reconnecting"):
                    self._status["state"] = "stopped"
        except Exception as exc:  # surface any failure in the UI instead of dying silently
            self._set(state="error", error=f"{type(exc).__name__}: {exc}", notice=None)
        finally:
            if cap is not None:
                cap.release()

    def _record(self, run_id, label, b, vis):
        """Queue the disk work (snapshot + DB row) for the writer thread and
        raise the NC alert straight away. Disk writes can stall for seconds on
        Windows (seen once: ~16 s, likely antivirus), and the camera loop must
        not wait for them."""
        now = datetime.now()
        snap = f"nc_{now:%Y%m%d_%H%M%S}_{run_id[:6]}_b{b.no}.jpg" if b.result == "NC" else None
        pulled = max(b.pads_in - b.leftover, 0)
        self._writes.put((snap, vis.copy() if snap else None,
                          (run_id, label, b.no, b.pads_in, pulled, b.leftover, b.result, b.reason, snap, now)))
        if b.result == "NC":
            with self._lock:
                self._status["last_nc"] = {"batch": b.no, "reason": b.reason, "snapshot": snap,
                                           "pads_in": b.pads_in, "leftover": b.leftover,
                                           "time": now.strftime("%H:%M:%S"), "acknowledged": False}

    def _writer(self):
        while True:
            snap, img, row = self._writes.get()
            try:
                if snap:
                    SNAPSHOT_DIR.mkdir(exist_ok=True)
                    cv2.imwrite(str(SNAPSHOT_DIR / snap), img)
                store.record_batch(*row[:-1], ts=row[-1])
            except Exception as exc:  # never lose the thread over one bad write
                self._event(f"could not save batch {row[2]}: {exc}")
            finally:
                self._writes.task_done()


runner = MonitorRunner()
