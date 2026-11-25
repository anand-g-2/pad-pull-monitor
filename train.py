"""
Train the 2-class YOLO detector (pad, puller).

  python train.py --data dataset/data.yaml --epochs 60

Best weights end up in runs/pad_puller*/weights/best.pt (path printed at the end).
"""
import argparse
import os

from ultralytics import YOLO

if __name__ == "__main__":          # guard is required on Windows (dataloader workers)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--weights", default="yolo11n.pt", help="pretrained start point (downloaded automatically if missing)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960, help="960 keeps the far, small pads visible")
    ap.add_argument("--batch", type=int, default=4,
                    help="4 fits a 4 GB GTX 1650 at 960 (AMP is disabled on that card, 8 overflows VRAM)")
    ap.add_argument("--name", default="pad_puller")
    a = ap.parse_args()

    model = YOLO(a.weights)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=0, workers=2, project=os.path.abspath("runs"), name=a.name, patience=20,
        fliplr=0.0,          # fixed camera: the puller always comes from the left
        degrees=0.0, scale=0.3, translate=0.1,   # small shifts = tolerance to camera bumps
        hsv_v=0.4,           # brightness changes (exposure, lighting)
        plots=True, verbose=False,
    )
    print("best weights:", model.trainer.best)
