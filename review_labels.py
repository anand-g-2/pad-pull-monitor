"""
Review / correct YOLO labels (pad, puller).

  python review_labels.py --data dataset --flagged          # only frames in review.csv
  python review_labels.py --data dataset --split train      # every train image

Mouse
  left drag       draw a box of the current class
  right click     delete the box under the cursor (smallest one)
Keys
  1 / 2           current class: 1 = pad (green), 2 = puller (orange)
  n, Space, ->    save, mark reviewed, next image
  p, <-           save, previous image
  v               paste the PAD boxes of the previous image (frames are similar)
  u               undo last change
  x               delete all boxes on this image
  q, Esc          save and quit (progress is kept in reviewed.txt; run again to resume)
"""
import argparse
import csv
import glob
import os

import cv2

NAMES = {0: "pad", 1: "puller"}
COLORS = {0: (0, 255, 0), 1: (0, 140, 255)}
KEY_LEFT, KEY_RIGHT = 2424832, 2555904       # cv2.waitKeyEx codes on Windows


def read_labels(path):
    boxes = []
    if os.path.exists(path):
        for line in open(path).read().splitlines():
            if line.strip():
                c, cx, cy, w, h = line.split()
                boxes.append([int(c), float(cx), float(cy), float(w), float(h)])
    return boxes


def write_labels(path, boxes):
    with open(path, "w") as f:
        f.write("\n".join(f"{c} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}" for c, cx, cy, w, h in boxes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="all", choices=["train", "val", "all"])
    ap.add_argument("--flagged", action="store_true", help="only images listed in review.csv")
    a = ap.parse_args()

    splits = ["train", "val"] if a.split == "all" else [a.split]
    items = []
    if a.flagged:
        for r in csv.DictReader(open(os.path.join(a.data, "review.csv"))):
            if r["split"] in splits:
                items.append((r["split"], r["image"]))
    else:
        for s in splits:
            items += [(s, os.path.splitext(os.path.basename(p))[0])
                      for p in sorted(glob.glob(os.path.join(a.data, "images", s, "*.jpg")))]
    if not items:
        raise SystemExit("nothing to review")

    done_path = os.path.join(a.data, "reviewed.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    idx = next((k for k, (_, n) in enumerate(items) if n not in done), 0)

    st = dict(cls=0, drag=None, cur=None, boxes=[], history=[], prev_pads=[])
    win = "review labels"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def load(k):
        split, name = items[k]
        img = cv2.imread(os.path.join(a.data, "images", split, name + ".jpg"))
        lab = os.path.join(a.data, "labels", split, name + ".txt")
        return img, lab, read_labels(lab)

    img, lab_path, st["boxes"] = load(idx)
    H, W = img.shape[:2]
    cv2.resizeWindow(win, W, H)

    def push():
        st["history"].append([b[:] for b in st["boxes"]])

    def on_mouse(event, x, y, flags, _):
        # dragging past the window edge reports coordinates outside the image;
        # clamp them, or YOLO rejects the whole image as corrupt
        x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = (x, y); st["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["drag"]:
            (x0, y0), (x1, y1) = st["drag"], (x, y)
            st["drag"] = None
            if abs(x1 - x0) > 4 and abs(y1 - y0) > 4:
                push()
                st["boxes"].append([st["cls"], (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                                    abs(x1 - x0) / W, abs(y1 - y0) / H])
        elif event == cv2.EVENT_RBUTTONDOWN:
            hits = [(b[3] * b[4], k) for k, b in enumerate(st["boxes"])
                    if abs(x / W - b[1]) <= b[3] / 2 and abs(y / H - b[2]) <= b[4] / 2]
            if hits:
                push()
                st["boxes"].pop(min(hits)[1])

    cv2.setMouseCallback(win, on_mouse)

    def save_and_go(new_idx, mark):
        nonlocal img, lab_path, idx
        write_labels(lab_path, st["boxes"])
        if mark:
            done.add(items[idx][1])
            with open(done_path, "w") as f:
                f.write("\n".join(sorted(done)))
        st["prev_pads"] = [b[:] for b in st["boxes"] if b[0] == 0]
        idx = max(0, min(new_idx, len(items) - 1))
        img, lab_path, st["boxes"] = load(idx)
        st["history"].clear()

    while True:
        vis = img.copy()
        for c, cx, cy, w, h in st["boxes"]:
            p0 = (int((cx - w / 2) * W), int((cy - h / 2) * H))
            p1 = (int((cx + w / 2) * W), int((cy + h / 2) * H))
            cv2.rectangle(vis, p0, p1, COLORS[c], 2)
            cv2.putText(vis, NAMES[c], (p0[0], p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1)
        if st["drag"]:
            cv2.rectangle(vis, st["drag"], st["cur"], COLORS[st["cls"]], 1)
        split, name = items[idx]
        n_pad = sum(b[0] == 0 for b in st["boxes"])
        status = (f"[{idx + 1}/{len(items)}] {split}/{name}  {'REVIEWED' if name in done else ''}  "
                  f"class: {NAMES[st['cls']]}  pads: {n_pad}  reviewed: {len(done)}")
        cv2.rectangle(vis, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(vis, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, vis)

        k = cv2.waitKeyEx(20)
        if k == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                write_labels(lab_path, st["boxes"])
                break
            continue
        ch = chr(k & 0xFF).lower() if k < 256 else ""
        if ch == "1":
            st["cls"] = 0
        elif ch == "2":
            st["cls"] = 1
        elif ch in ("n", " ") or k == KEY_RIGHT:
            if idx == len(items) - 1:
                save_and_go(idx, True)
                print("last image reached")
            else:
                save_and_go(idx + 1, True)
        elif ch == "p" or k == KEY_LEFT:
            save_and_go(idx - 1, False)
        elif ch == "v" and st["prev_pads"]:
            push()
            st["boxes"] += [b[:] for b in st["prev_pads"]]
        elif ch == "u" and st["history"]:
            st["boxes"] = st["history"].pop()
        elif ch == "x":
            push()
            st["boxes"] = []
        elif ch == "q" or k == 27:
            write_labels(lab_path, st["boxes"])
            break
    cv2.destroyAllWindows()
    print(f"{len(done)} images marked reviewed (saved in {done_path})")


if __name__ == "__main__":
    main()
