"""
Disc-pad pull monitor
=====================
Watches the oven-exit belt, counts pads per batch, detects the pulling unit,
and keeps checking the belt AFTER the pulling unit has left.  If a pad is still
inside the ROI after a pull -> NC (non-conformance).

Two interchangeable detectors:
  * YOLO (default when --model is given): trained 2-class model, pad + puller
  * classic (fallback, no training):  dark blobs = pads, bright change vs.
    empty-belt reference = puller; dark blobs attached to the puller are ignored

The ROI is the hand-drawn polygon from draw_roi.py (roi.json).

Usage
-----
  python draw_roi.py VIDEO --out roi.json                       # once per camera position
  python monitor.py run VIDEO --model best.pt --roi roi.json --out annotated.mp4 --csv batches.csv [--show]

  # classic fallback needs an empty-belt reference instead of a model
  python monitor.py calibrate VIDEO --bg bg.png
  python monitor.py run VIDEO --bg bg.png --roi roi.json
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from draw_roi import load_roi, roi_to_pixels

WORK_W, WORK_H = 960, 540          # all boxes / ROI are handled at this resolution


# ----------------------------------------------------------------- settings
@dataclass
class Cfg:
    expected_pads: int = 5
    # YOLO
    pad_conf: float = 0.40
    puller_conf_on: float = 0.50
    puller_conf_off: float = 0.25
    imgsz: int = 960
    # classic
    blur: int = 15
    pad_dark_thr: int = 75
    pad_min_area: int = 350
    pad_max_area: int = 15000
    static_dark_thr: int = 90
    puller_on: float = 0.15
    puller_off: float = 0.05
    # state machine
    puller_on_frames: int = 2
    puller_off_frames: int = 3
    stable_frames: int = 6
    settle_s: float = 0.35
    verify_s: float = 2.0
    nc_on_count_mismatch: bool = True


@dataclass
class Detections:
    pads: list                  # [(x0, y0, x1, y1), ...] at WORK size, inside ROI
    puller_box: tuple | None    # (x0, y0, x1, y1) or None
    puller_on: bool             # strong evidence the puller is in view
    puller_off: bool            # strong evidence it is NOT in view
    puller_score: float


def open_source(src):
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def roi_mask(roi_path):
    m = np.zeros((WORK_H, WORK_W), np.uint8)
    cv2.fillPoly(m, [roi_to_pixels(load_roi(roi_path), WORK_W, WORK_H)], 1)
    return m > 0


def in_roi(mask, box):
    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    return 0 <= cx < WORK_W and 0 <= cy < WORK_H and mask[cy, cx]


# ----------------------------------------------------------------- detectors
class YoloDetector:
    def __init__(self, model_path, roi, cfg):
        from ultralytics import YOLO
        # .pt, TensorRT .engine, or an NCNN model folder (*_ncnn_model, for the Raspberry Pi)
        self.model, self.roi, self.cfg = YOLO(model_path, task="detect"), roi, cfg
        self.device, self.imgsz = None, cfg.imgsz
        meta = Path(model_path) / "metadata.yaml"
        if meta.exists():   # NCNN folder: fixed input shape (e.g. 544x960), CPU runtime
            import yaml
            self.imgsz = tuple(yaml.safe_load(meta.read_text())["imgsz"])
            self.device = "cpu"   # without this Ultralytics would use Vulkan on a GPU PC

    def __call__(self, frame):
        c = self.cfg
        r = self.model.predict(frame, imgsz=self.imgsz, conf=min(c.pad_conf, c.puller_conf_off),
                               device=self.device, verbose=False)[0]
        sx, sy = WORK_W / frame.shape[1], WORK_H / frame.shape[0]
        pads, puller, pconf = [], None, 0.0
        for (x0, y0, x1, y1), cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                               r.boxes.conf.tolist()):
            box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            if int(cls) == 0 and conf >= c.pad_conf and in_roi(self.roi, box):
                pads.append(box)
            elif int(cls) == 1 and conf > pconf:
                puller, pconf = box, conf
        return Detections(pads, puller, pconf >= c.puller_conf_on, pconf < c.puller_conf_off, pconf)


class ClassicDetector:
    def __init__(self, bg_bgr, roi, cfg):
        self.cfg = cfg
        self.bg = cv2.blur(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY), (cfg.blur, cfg.blur))
        search = roi.copy()
        search[self.bg < cfg.static_dark_thr] = False          # oven shadow, sensor gap
        self.search = cv2.erode(search.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        self.search_area = max(self.search.sum(), 1)

    def __call__(self, frame):
        c = self.cfg
        small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        g = cv2.blur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (c.blur, c.blur))
        diff = cv2.absdiff(g, self.bg)
        bright = ((diff > 45) & (g > self.bg)).astype(np.uint8)
        score = (bright.astype(bool) & self.search).sum() / self.search_area

        puller_box, near = None, None
        if score > c.puller_off:
            pm = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            ys, xs = np.nonzero(pm)
            puller_box = (xs.min(), ys.min(), xs.max(), ys.max())
            near = cv2.dilate(pm, np.ones((31, 31), np.uint8)) > 0

        dark = ((g < c.pad_dark_thr) & self.search).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
        pads = []
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if not c.pad_min_area <= area <= c.pad_max_area:
                continue
            if near is not None and near[lab == k].any():
                continue           # attached to the puller -> its black parts, not a pad
            pads.append((x, y, x + w, y + h))
        return Detections(pads, puller_box, score > c.puller_on, score < c.puller_off, score)


# ----------------------------------------------------------------- state machine
class StableCounter:
    """Returns a value only once it has been seen N frames in a row."""
    def __init__(self, n):
        self.n, self.hist = n, deque(maxlen=n)

    def push(self, v):
        self.hist.append(v)
        return v if len(self.hist) == self.n and len(set(self.hist)) == 1 else None

    def reset(self):
        self.hist.clear()


@dataclass
class Batch:
    no: int
    pads_in: int = 0
    pull_start_s: float = 0.0
    pull_end_s: float = 0.0
    leftover: int = 0
    result: str = ""
    reason: str = ""


class PadPullMonitor:
    APPROACH, PULLING, VERIFY = "WAIT/APPROACH", "PULLING", "POST-PULL CHECK"

    def __init__(self, detector, roi, fps, cfg=Cfg()):
        self.det, self.roi, self.fps, self.cfg = detector, roi, fps, cfg
        self.state = self.APPROACH
        self.on_cnt = self.off_cnt = 0
        self.stable = StableCounter(cfg.stable_frames)
        self.batch_max = self.leftover_max = 0
        self.verify_start = self.verify_end = 0
        self.frame_i = 0
        self.batches: list[Batch] = []
        self.cur: Batch | None = None
        self.total_pads = self.total_pulled = self.nc_count = 0
        self.last_nc_frame = -10**9
        self.d = Detections([], None, False, True, 0.0)

    def update(self, frame):
        c, t = self.cfg, self.frame_i / self.fps
        self.d = self.det(frame)
        n = len(self.d.pads)
        events = []
        puller_now = self._puller_hysteresis()

        if self.state == self.APPROACH:
            s = self.stable.push(n)
            if s is not None:
                self.batch_max = max(self.batch_max, s)
            if puller_now:
                self.cur = Batch(no=len(self.batches) + 1, pads_in=self.batch_max, pull_start_s=t)
                self.state = self.PULLING
                events.append(f"{t:7.2f}s  batch {self.cur.no}: puller arrived, {self.batch_max} pads staged")

        elif self.state == self.PULLING:
            if not puller_now:
                self.cur.pull_end_s = t
                self.verify_start = self.frame_i + int(c.settle_s * self.fps)
                self.verify_end = self.verify_start + int(c.verify_s * self.fps)
                self.leftover_max = 0
                self.stable.reset()
                self.state = self.VERIFY
                events.append(f"{t:7.2f}s  batch {self.cur.no}: puller left, checking belt")

        elif self.state == self.VERIFY:
            if self.frame_i >= self.verify_start:
                s = self.stable.push(n)
                if s is not None:
                    self.leftover_max = max(self.leftover_max, s)
            if self.frame_i >= self.verify_end or puller_now:
                events += self._close_batch(t)
                if puller_now:   # puller came straight back (re-pull) -> new cycle
                    self.cur = Batch(no=len(self.batches) + 1, pads_in=self.leftover_max, pull_start_s=t)
                    self.state = self.PULLING

        self.frame_i += 1
        return events

    def _puller_hysteresis(self):
        c = self.cfg
        if self.d.puller_on:
            self.on_cnt += 1; self.off_cnt = 0
        elif self.d.puller_off:
            self.off_cnt += 1; self.on_cnt = 0
        if self.state == self.PULLING:
            return self.off_cnt < c.puller_off_frames
        return self.on_cnt >= c.puller_on_frames

    def _close_batch(self, t):
        c, b = self.cfg, self.cur
        b.leftover = self.leftover_max
        reasons = []
        if b.leftover > 0:
            reasons.append(f"{b.leftover} pad(s) left on belt after pull")
        if c.nc_on_count_mismatch and b.pads_in != c.expected_pads:
            reasons.append(f"batch had {b.pads_in} pads, expected {c.expected_pads}")
        b.result = "NC" if reasons else "OK"
        b.reason = "; ".join(reasons)
        pulled = max(b.pads_in - b.leftover, 0)
        self.total_pads += b.pads_in
        self.total_pulled += pulled
        if b.result == "NC":
            self.nc_count += 1
            self.last_nc_frame = self.frame_i
        self.batches.append(b)
        self.batch_max = 0
        self.stable.reset()
        self.state = self.APPROACH
        return [f"{t:7.2f}s  batch {b.no}: {b.result}  pads_in={b.pads_in} pulled={pulled} "
                f"leftover={b.leftover}  {b.reason}"]

    def draw(self, frame, hud=True):
        """ROI outline + detection boxes. hud=False leaves out the stats box and NC
        banner (the web app shows those in its side panel instead)."""
        small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        cnts, _ = cv2.findContours(self.roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(small, cnts, -1, (0, 255, 255), 1)
        if self.d.puller_box is not None and self.state == self.PULLING:
            x0, y0, x1, y1 = map(int, self.d.puller_box)
            cv2.rectangle(small, (x0, y0), (x1, y1), (0, 140, 255), 2)
            cv2.putText(small, "PULLER", (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        else:   # pads are only meaningful when the puller is not on the belt
            col = (0, 0, 255) if self.state == self.VERIFY else (0, 255, 0)
            for x0, y0, x1, y1 in self.d.pads:
                cv2.rectangle(small, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
        if not hud:
            return small
        lines = [f"State: {self.state}",
                 f"Pads in ROI: {len(self.d.pads)}   puller: {self.d.puller_score:.2f}",
                 f"Batches: {len(self.batches)}   Pads: {self.total_pads}   Pulled: {self.total_pulled}",
                 f"NC: {self.nc_count}"]
        cv2.rectangle(small, (0, 0), (520, 118), (0, 0, 0), -1)
        for k, s in enumerate(lines):
            cv2.putText(small, s, (8, 24 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if self.frame_i - self.last_nc_frame < 3 * self.fps:
            b = self.batches[-1]
            cv2.rectangle(small, (0, WORK_H - 60), (WORK_W, WORK_H), (0, 0, 255), -1)
            cv2.putText(small, f"NC  batch {b.no}: {b.reason}", (10, WORK_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return small


# ----------------------------------------------------------------- calibration / runner
def build_reference(src, out_path, step=20, max_frames=3000):
    """Median of sampled frames = empty belt (pads/puller are only transient)."""
    cap = open_source(src)
    frames, i = [], 0
    while i < max_frames and cap.grab():
        if i % step == 0:
            _, f = cap.retrieve()
            frames.append(cv2.resize(f, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA))
        i += 1
    if len(frames) < 10:
        sys.exit("not enough frames to build reference")
    cv2.imwrite(out_path, np.median(np.stack(frames), axis=0).astype(np.uint8))
    print(f"reference written to {out_path} from {len(frames)} frames")


def run(src, roi_path, model=None, bg_path=None, out=None, csv_path=None, show=False,
        frame_hook=None, cfg=Cfg()):
    roi = roi_mask(roi_path)
    if model:
        det = YoloDetector(model, roi, cfg)
    else:
        bg = cv2.imread(bg_path) if bg_path else None
        if bg is None:
            sys.exit("classic mode needs --bg (run 'calibrate' first) - or pass --model")
        det = ClassicDetector(bg, roi, cfg)
    cap = open_source(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    mon = PadPullMonitor(det, roi, fps, cfg)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WORK_W, WORK_H)) if out else None
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_hook:
            frame = frame_hook(mon.frame_i, frame)
        for e in mon.update(frame):
            print(e, flush=True)
        if writer or show:
            vis = mon.draw(frame)
            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("pad pull monitor", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    if mon.state != mon.APPROACH:
        print(f"stream ended during {mon.state} - batch {len(mon.batches) + 1} not judged")
    if writer:
        writer.release()
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["batch", "pads_in", "pull_start_s", "pull_end_s", "leftover", "result", "reason"])
            for b in mon.batches:
                w.writerow([b.no, b.pads_in, f"{b.pull_start_s:.2f}", f"{b.pull_end_s:.2f}",
                            b.leftover, b.result, b.reason])
    print(f"\nframes={mon.frame_i}  batches={len(mon.batches)}  pads={mon.total_pads}  "
          f"pulled={mon.total_pulled}  NC={mon.nc_count}  ({mon.frame_i / (time.time() - t0):.0f} fps)")
    return mon


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "run"])
    ap.add_argument("source", help="video file or camera index")
    ap.add_argument("--roi", default="roi.json")
    ap.add_argument("--model", help="YOLO weights (best.pt); omit to use the classic detector")
    ap.add_argument("--bg", default="bg.png", help="empty-belt reference (classic detector only)")
    ap.add_argument("--out")
    ap.add_argument("--csv")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.mode == "calibrate":
        build_reference(a.source, a.bg)
    else:
        run(a.source, a.roi, a.model, a.bg, a.out, a.csv, a.show)
