"""
Convert the trained YOLO model (best.pt) to NCNN, the CPU runtime used on a
Raspberry Pi, and check it against the original.

  python export_ncnn.py                 # newest runs/*/weights/best.pt, 960 wide
  python export_ncnn.py --imgsz 640     # ~2x faster, misses a few more far pads
                                        # (then also set Cfg.imgsz = 640 in monitor.py)

Output: runs/<run>/weights/best_ncnn_model_544x960 (height x width). The input
is 16:9 like the camera, not square: a square 960x960 input would spend 40% of
the work on padding. The web app uses this folder automatically when there is
no TensorRT engine (as on a Pi) and its width matches Cfg.imgsz.

An NCNN model is not tied to the machine that built it: export on the PC,
then copy the folder to the Pi (it is committed to Git for that reason).
Exporting needs `pip install ncnn pnnx`; running it only needs ncnn.

The check runs both models on the validation images and scores detections
against the labels at the monitor's own thresholds (pad 0.40, puller 0.50,
IoU 0.5), plus CPU time per frame. Ultralytics' own `val` can't be used: it
only supports square inputs for exported models. Speed on a PC is not Pi speed.
"""
import argparse
import glob
import math
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from monitor import Cfg

NAMES = {0: "pad", 1: "puller"}


def newest_weights() -> str:
    found = sorted(glob.glob("runs/**/weights/best.pt", recursive=True), key=os.path.getmtime)
    if not found:
        raise SystemExit("no runs/*/weights/best.pt found - train a model first (train.py)")
    return found[-1]


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def _labels(img_path, w, h):
    lp = Path(img_path.replace("images", "labels")).with_suffix(".txt")
    out = []
    for line in (lp.read_text().splitlines() if lp.exists() else []):
        if line.strip():
            c, x, y, bw, bh = map(float, line.split())
            out.append((int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def score(model_path, imgsz, images, cfg=Cfg()):
    """Precision/recall per class at the monitor thresholds + CPU ms/frame."""
    conf = {0: cfg.pad_conf, 1: cfg.puller_conf_on}
    m = YOLO(model_path, task="detect")
    tp, fp, fn, times = {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, []
    for k, ip in enumerate(images):
        im = cv2.imread(ip)
        t = time.perf_counter()
        r = m.predict(im, imgsz=imgsz, conf=0.25, device="cpu", verbose=False)[0]
        if k >= 3:                                   # skip warm-up
            times.append(time.perf_counter() - t)
        dets = [(int(c), b.tolist()) for c, b, s in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)
                if s >= conf[int(c)]]
        gt, used = _labels(ip, im.shape[1], im.shape[0]), set()
        for c, b in dets:
            cand = [(_iou(b, gb), j) for j, (gc, gb) in enumerate(gt) if gc == c and j not in used]
            best = max(cand, default=(0, -1))
            if best[0] >= 0.5:
                tp[c] += 1
                used.add(best[1])
            else:
                fp[c] += 1
        for j, (gc, _) in enumerate(gt):
            if j not in used:
                fn[gc] += 1
    return tp, fp, fn, float(np.mean(times) * 1000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="best.pt to convert (default: newest in runs/)")
    ap.add_argument("--imgsz", type=int, default=960, help="input width; height = 16:9, rounded up to 32")
    ap.add_argument("--fp16", action="store_true", help="store FP16 weights (half the size)")
    ap.add_argument("--no-check", action="store_true", help="export only")
    a = ap.parse_args()

    weights = a.weights or newest_weights()
    shape = (math.ceil(a.imgsz * 9 / 16 / 32) * 32, a.imgsz)      # 960 -> 544x960, 640 -> 384x640
    print(f"exporting {weights} -> NCNN {shape[0]}x{shape[1]} ({'FP16' if a.fp16 else 'FP32'})")
    out = Path(YOLO(weights).export(format="ncnn", imgsz=shape, half=a.fp16, device="cpu"))
    target = out.with_name(f"best_ncnn_model_{shape[0]}x{shape[1]}")
    shutil.rmtree(target, ignore_errors=True)
    out = Path(shutil.move(str(out), str(target)))
    print(f"\nNCNN model written: {out}")

    images = sorted(glob.glob("dataset/images/val/*.jpg"))
    if a.no_check or not images:
        print("check skipped" + ("" if images else " (dataset/images/val not found)"))
        raise SystemExit
    print(f"\nchecking on {len(images)} validation images (CPU, {os.cpu_count()} threads)...")
    print(f"{'model':22} {'pad P':>6} {'pad R':>6} {'puller P':>9} {'puller R':>9} {'ms/frame':>9}")
    for name, path, sz in (("PyTorch .pt", weights, a.imgsz), (f"NCNN {shape[0]}x{shape[1]}", str(out), shape)):
        tp, fp, fn, ms = score(path, sz, images)
        p = {c: tp[c] / max(tp[c] + fp[c], 1) for c in NAMES}
        r = {c: tp[c] / max(tp[c] + fn[c], 1) for c in NAMES}
        print(f"{name:22} {p[0]:6.3f} {r[0]:6.3f} {p[1]:9.3f} {r[1]:9.3f} {ms:9.1f}")
