"""
Draw the region of interest (where pads are counted / checked) by hand.

  python draw_roi.py SOURCE [--frame 525] [--out roi.json]

SOURCE = video file, camera index, or an image (e.g. bg.png)

  left click   add a corner
  right click  remove last corner
  c            clear all corners
  s / Enter    save and quit (needs >= 3 corners)
  Esc / q      quit without saving

Corners are stored as fractions of width/height (0..1), so the same roi.json
works at any resolution.
"""
import argparse
import json

import cv2
import numpy as np

DISPLAY_W, DISPLAY_H = 1280, 720


def load_frame(src, idx):
    if str(src).lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return cv2.imread(src)
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    frame = None
    for _ in range(idx + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    return frame


def load_roi(path):
    """Returns the polygon as a list of (x, y) fractions."""
    with open(path) as f:
        return [tuple(p) for p in json.load(f)["polygon"]]


def roi_to_pixels(poly, w, h):
    return np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--frame", type=int, default=525)
    ap.add_argument("--out", default="roi.json")
    a = ap.parse_args()

    frame = load_frame(a.source, a.frame)
    if frame is None:
        raise SystemExit("could not read a frame from " + a.source)
    base = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
    pts = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    win = "draw ROI - left: add, right: undo, c: clear, s: save, Esc: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = base.copy()
        if len(pts) >= 3:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 255, 255))
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, np.int32)], len(pts) >= 3, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
        cv2.putText(vis, f"corners: {len(pts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("s"), 13) and len(pts) >= 3:
            poly = [(round(x / DISPLAY_W, 4), round(y / DISPLAY_H, 4)) for x, y in pts]
            with open(a.out, "w") as f:
                json.dump({"polygon": poly}, f, indent=1)
            print(f"saved {len(poly)} corners to {a.out}")
            break
        if k == ord("c"):
            pts.clear()
        if k in (27, ord("q")):
            print("quit without saving")
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
