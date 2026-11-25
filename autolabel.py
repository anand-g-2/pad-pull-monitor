"""
Make first-draft YOLO labels (pad, puller) from the videos with the classical
pipeline, so you only have to CORRECT boxes instead of drawing all of them.

  python autolabel.py --train ../20261124162029.avi --val ../20261124162403.avi --bg bg.png --out dataset

Classes: 0 = pad, 1 = puller
Frames with the puller in view are listed in dataset/review.csv: pads touching
the puller cannot be separated from its black parts automatically, so those
frames MUST be checked with review_labels.py.
"""
import argparse
import csv
import os

import cv2
import numpy as np

WORK_W, WORK_H = 960, 540
SAVE_W, SAVE_H = 1280, 720
PAD, PULLER = 0, 1


class AutoLabeler:
    def __init__(self, bg_bgr):
        self.bg = cv2.blur(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY), (15, 15))
        m = np.zeros_like(self.bg)
        m[205:, 295:935] = 255
        m[self.bg < 90] = 0
        self.belt = cv2.erode(m, np.ones((9, 9), np.uint8)) > 0

    def label(self, frame):
        small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        g = cv2.blur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (15, 15))
        diff = cv2.absdiff(g, self.bg)

        # puller = big bright change + any changed pixels attached to it (black cap, bar end)
        bright = ((diff > 45) & (g > self.bg)).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        changed = cv2.morphologyEx((diff > 45).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(changed)
        puller_mask = np.zeros_like(changed)
        for k in range(1, n):
            comp = lab == k
            if st[k, 4] > 3000 and (bright[comp].mean() > 0.3):
                puller_mask[comp] = 1
        boxes, puller_present = [], puller_mask.sum() > 4000
        if puller_present:
            ys, xs = np.nonzero(puller_mask)
            boxes.append((PULLER, xs.min(), ys.min(), xs.max(), ys.max()))
        near_puller = cv2.dilate(puller_mask, np.ones((31, 31), np.uint8)) > 0

        dark = ((g < 75) & self.belt).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if not 350 <= area <= 15000:
                continue
            if puller_present and near_puller[lab == k].any():
                continue          # ambiguous: could be the puller's black parts
            boxes.append((PAD, x - 3, y - 3, x + w + 3, y + h + 3))
        return boxes, puller_present


def to_yolo(boxes):
    lines = []
    for c, x0, y0, x1, y1 in boxes:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, WORK_W - 1), min(y1, WORK_H - 1)
        cx, cy = (x0 + x1) / 2 / WORK_W, (y0 + y1) / 2 / WORK_H
        w, h = (x1 - x0) / WORK_W, (y1 - y0) / WORK_H
        lines.append(f"{c} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}")
    return lines


def process(video, split, tag, labeler, out, review_rows, step_idle, step_puller):
    os.makedirs(f"{out}/images/{split}", exist_ok=True)
    os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    cap = cv2.VideoCapture(video)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step_puller == 0 or i % step_idle == 0:
            boxes, puller = labeler.label(frame)
            if (puller and i % step_puller == 0) or i % step_idle == 0:
                name = f"{tag}_{i:05d}"
                cv2.imwrite(f"{out}/images/{split}/{name}.jpg",
                            cv2.resize(frame, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(f"{out}/labels/{split}/{name}.txt", "w") as f:
                    f.write("\n".join(to_yolo(boxes)))
                if puller:
                    review_rows.append((split, name, "puller in view - add pads next to/under the puller"))
                saved += 1
        i += 1
    print(f"{video}: {i} frames read, {saved} saved to {split}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--bg", default="bg.png")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--step-idle", type=int, default=10, help="keep every Nth frame without puller")
    ap.add_argument("--step-puller", type=int, default=4, help="keep every Nth frame with puller")
    a = ap.parse_args()

    labeler = AutoLabeler(cv2.imread(a.bg))
    rows = []
    for split, vids in (("train", a.train), ("val", a.val)):
        for v in vids:
            tag = os.path.splitext(os.path.basename(v))[0]
            process(v, split, tag, labeler, a.out, rows, a.step_idle, a.step_puller)
    with open(f"{a.out}/review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "reason"])
        w.writerows(rows)
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: pad\n  1: puller\n")
    print(f"{len(rows)} frames flagged for review in {a.out}/review.csv")
