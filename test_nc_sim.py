"""
Simulated NC test: there is no real footage of a missed pull, so we cut a real
pad out of one frame and paste it back onto the belt for a stretch of frames
right after a pull, i.e. "the puller left one pad behind".

  python test_nc_sim.py VIDEO --model best.pt --roi roi.json --out nc_sim.mp4
  python test_nc_sim.py VIDEO --bg bg.png --roi roi.json          # classic detector
"""
import argparse

import cv2
import numpy as np

import monitor

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--roi", default="roi.json")
ap.add_argument("--model", help="YOLO weights; omit for the classic detector")
ap.add_argument("--bg", default="bg.png")
ap.add_argument("--donor", type=int, default=525, help="frame to cut the pad from")
ap.add_argument("--box", default="1240,710,1470,840", help="x0,y0,x1,y1 of a pad in the donor frame (full res)")
ap.add_argument("--start", type=int, default=635)
ap.add_argument("--end", type=int, default=950)
ap.add_argument("--out")
ap.add_argument("--csv")
a = ap.parse_args()

x0, y0, x1, y1 = map(int, a.box.split(","))
cap = cv2.VideoCapture(a.video)
for _ in range(a.donor + 1):
    ok, donor = cap.read()
patch = donor[y0:y1, x0:x1].copy()
g = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (21, 21), 0)
mask = cv2.GaussianBlur((g < 90).astype(np.float32), (9, 9), 0)[..., None]   # soft pad mask


def inject(i, frame):
    if a.start <= i < a.end:
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (mask * patch + (1 - mask) * roi).astype(np.uint8)
    return frame


monitor.run(a.video, a.roi, a.model, a.bg, a.out, a.csv, frame_hook=inject)
