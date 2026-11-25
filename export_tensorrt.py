"""
Convert the trained YOLO model (best.pt) to a TensorRT engine (best.engine)
for faster inference on an NVIDIA GPU.

  python export_tensorrt.py                                   # newest runs/*/weights/best.pt
  python export_tensorrt.py --weights runs/pad_puller_v1/weights/best.pt
  python export_tensorrt.py --fp32                            # if FP16 gives worse results

IMPORTANT: a TensorRT engine only works on the GPU model + TensorRT version it
was built with. Do NOT copy best.engine between PCs: run this script once on
every PC (e.g. the client PC) after installing the requirements. The web app
automatically uses best.engine if it sits next to the newest best.pt and is
newer than it; otherwise it falls back to best.pt.

After exporting, the script checks the engine on the validation set and prints
its accuracy next to the .pt model's, so you can see that nothing was lost.
"""
import argparse
import glob
import os
import time
from pathlib import Path

from ultralytics import YOLO


def newest_weights() -> str:
    found = sorted(glob.glob("runs/**/weights/best.pt", recursive=True), key=os.path.getmtime)
    if not found:
        raise SystemExit("no runs/*/weights/best.pt found - train a model first (train.py)")
    return found[-1]


def speed(model, img, n=50):
    for _ in range(5):                      # warm-up
        model.predict(img, imgsz=960, verbose=False)
    t = time.perf_counter()
    for _ in range(n):
        model.predict(img, imgsz=960, verbose=False)
    return (time.perf_counter() - t) / n * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="best.pt to convert (default: newest in runs/)")
    ap.add_argument("--imgsz", type=int, default=960, help="must match training / monitor (960)")
    ap.add_argument("--fp32", action="store_true", help="build in FP32 instead of FP16")
    ap.add_argument("--data", default="dataset/data.yaml", help="for the accuracy check (skipped if missing)")
    a = ap.parse_args()

    weights = a.weights or newest_weights()
    print(f"exporting {weights} -> TensorRT ({'FP32' if a.fp32 else 'FP16'}, imgsz {a.imgsz})")
    engine = YOLO(weights).export(format="engine", imgsz=a.imgsz, half=not a.fp32, device=0,
                                  batch=1, workspace=2, simplify=True)
    print(f"\nengine written: {engine}")

    # ---- check: accuracy and speed, .pt vs .engine
    val_imgs = sorted(glob.glob("dataset/images/val/*.jpg"))
    if Path(a.data).exists() and val_imgs:
        rows = []
        for name, path in (("PyTorch .pt", weights), ("TensorRT .engine", engine)):
            m = YOLO(path, task="detect")
            r = m.val(data=a.data, imgsz=a.imgsz, batch=1, device=0, verbose=False, plots=False)
            rows.append((name, r.box.mp, r.box.mr, r.box.map50, r.box.map, speed(m, val_imgs[0])))
        print(f"\n{'model':18} {'precision':>9} {'recall':>7} {'mAP50':>7} {'mAP50-95':>9} {'ms/frame':>9}")
        for name, p, rcl, m50, m, ms in rows:
            print(f"{name:18} {p:9.3f} {rcl:7.3f} {m50:7.3f} {m:9.3f} {ms:9.1f}")
    else:
        print("dataset not found - skipped the accuracy/speed check")
