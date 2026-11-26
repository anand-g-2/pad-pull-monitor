# Disc Pad Pull Monitor

Camera-based quality check at the exit of the disc-pad baking oven.

Batches of **5 disc pads** leave the oven on a mesh belt and stop at a proximity sensor. A pneumatic **pulling unit** then reaches in and pulls the batch onto the conveyor. This application watches the belt with a camera and:

- **counts the pads**, per batch and in total;
- **counts the batches**;
- keeps watching **after the pulling unit has left the frame**. If any pad is still inside the region of interest (ROI), it raises an **NC (non-conformance)** with a snapshot;
- also raises an NC when a batch doesn't contain exactly 5 pads (for example, a leftover pad from the previous pull plus a new batch).

It runs as a **web application** with four pages: Overview, Live Monitoring, Analytics and Calibration. Detection uses a **YOLO11n model** trained on this line (classes `pad` and `puller`). On NVIDIA GPUs it can be sped up with **TensorRT**.

---

## Contents

1. [How it works](#1-how-it-works)
2. [What you need](#2-what-you-need)
3. [Setup on a new PC](#3-setup-on-a-new-pc)
4. [Speed up with TensorRT (recommended)](#4-speed-up-with-tensorrt-recommended)
   - [4.1 Raspberry Pi: the NCNN model](#41-raspberry-pi-the-ncnn-model)
5. [Run the web app](#5-run-the-web-app)
6. [Using the web app](#6-using-the-web-app)
7. [Testing without the line camera](#7-testing-without-the-line-camera)
8. [Command-line tools](#8-command-line-tools)
9. [Retraining the model](#9-retraining-the-model)
10. [Settings you can tune](#10-settings-you-can-tune)
11. [Files and folders](#11-files-and-folders)
12. [Troubleshooting](#12-troubleshooting)
13. [Validation results](#13-validation-results)

---

## 1. How it works

```
camera ──► YOLO (pad, puller) ──► pads inside the ROI polygon ──► state machine ──► OK / NC
                                                                        │
                                       web app: live stream, alerts, SQLite records, charts
```

The state machine runs once per batch:

| State | What happens |
|---|---|
| **WAIT / APPROACH** | Pads arrive from the oven. The batch size is the highest pad count that stays the same for 6 frames in a row. |
| **PULLING** | The pulling unit is in view. Nothing is counted while it is there. |
| **POST-PULL CHECK** | Starts 0.35 s after the puller leaves and lasts 2 s. Any pad still in the ROI (seen for 6 frames in a row) means **NC: pad left on belt after pull**. |
| → back to WAIT | The batch is stored as OK or NC. A batch whose size isn't 5 is also NC. |

All timings are in seconds. The app measures the camera's real frame rate, so the timings stay correct whatever the camera delivers.

---

## 2. What you need

| | Minimum | Tested with |
|---|---|---|
| OS | Windows 10/11, 64-bit | Windows 11 |
| GPU | NVIDIA GPU with 4 GB+ VRAM (needed for real-time; a CPU works but is slow) | GeForce GTX 1650, 4 GB |
| NVIDIA driver | supporting CUDA 12.4 or newer (`nvidia-smi` shows "CUDA Version: 12.4" or higher) | 610.62 |
| Python | **3.11** (64-bit) | 3.11.9 |
| Camera | USB/UVC camera, **1920×1080 (or another 16:9 size) at ≥ 10 fps**, fixed and looking down on the belt | — |
| Disk | ~6 GB free (Python packages incl. PyTorch + TensorRT) | — |
| Git | to clone the repository | 2.51 |

> **Camera position matters.** The model was trained on footage from one fixed camera position: the belt seen from the left, the oven exit at the top left, the sensor at the bottom right. A very different view needs new training data (see [section 9](#9-retraining-the-model)).

---

## 3. Setup on a new PC

All commands are for **Windows Command Prompt** or **PowerShell**, run from the project folder.

### 3.1 Install the prerequisites

1. **NVIDIA driver**: install the latest Game Ready or Studio driver from nvidia.com. Check it with:
   ```
   nvidia-smi
   ```
2. **Python 3.11**: download it from python.org. In the installer, tick **"Add python.exe to PATH"**. Check it with:
   ```
   python --version
   ```
3. **Git**: download it from git-scm.com.

### 3.2 Get the code

```
git clone https://github.com/<owner>/<repo>.git pad_pull_monitor
cd pad_pull_monitor
```

### 3.3 Create a virtual environment (recommended)

A virtual environment keeps this project's packages separate from other Python projects on the PC.

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Your prompt now starts with `(.venv)`. Run `.venv\Scripts\activate` again in every new terminal before using the project. `start_webapp.bat` does it for you.

### 3.4 Install PyTorch with CUDA

The normal `pip install torch` gives a CPU-only build, so install the CUDA build explicitly:

```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Check that PyTorch sees the GPU:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print `True` and your GPU's name.

### 3.5 Install the remaining packages

```
pip install -r requirements.txt
```

This installs Ultralytics (YOLO), OpenCV, Flask and the TensorRT/ONNX tools. `numpy` is pinned to 1.26.4 on purpose; see [Troubleshooting](#12-troubleshooting).

### 3.6 Check the installation

```
python -c "import cv2, flask, numpy, tensorrt; from ultralytics import YOLO; m = YOLO('runs/pad_puller_v1/weights/best.pt'); print('OK', numpy.__version__, tensorrt.__version__)"
```

> The **first** `import torch` / `ultralytics` on a PC can take 1–2 minutes. Later starts are much faster.

---

## 4. Speed up with TensorRT (recommended)

TensorRT converts the model into an engine optimised for **this PC's GPU**, with the same accuracy (see [section 13](#13-validation-results)). On the GTX 1650 the model itself runs about **20% faster** (32 → 25.5 ms per frame). The complete monitor, however, is limited by the CPU (decoding 1080p frames, resizing, drawing), so overall it runs at about the same rate either way: 27–30 fps, well above the camera's 15 fps. TensorRT mainly adds headroom: less GPU time per frame, which matters on a weaker GPU or when other software shares the GPU.

```
python export_tensorrt.py
```

- It takes about 5–10 minutes. It writes `runs/pad_puller_v1/weights/best.engine` and then prints an accuracy and speed comparison with the `.pt` model.
- The web app **automatically uses `best.engine`** when it exists next to the newest `best.pt` (and is newer than it). The Live page then shows the model as `pad_puller_v1 (TensorRT)`.
- **An engine only works on the GPU model and TensorRT version it was built on.** Never copy `best.engine` to another PC. Run `export_tensorrt.py` on each PC instead. That's why `*.engine` is not in Git.
- If the FP16 engine ever performs worse than the `.pt` model, rebuild it in full precision: `python export_tensorrt.py --fp32`.
- To go back to the `.pt` model, delete `best.engine`.

### 4.1 Raspberry Pi: the NCNN model

A Raspberry Pi has no NVIDIA GPU, so it runs the model on its CPU with **NCNN**, a runtime built for ARM boards. The NCNN model is **already in the repo** (`runs/pad_puller_v1/weights/best_ncnn_model_544x960/`). Unlike a TensorRT engine it works on any machine, so the Pi doesn't need to convert anything.

To rebuild it on the PC after retraining:

```
python export_ncnn.py
```

It writes the folder above, then scores the `.pt` and NCNN models on the validation images. The input is **544x960 (16:9, like the camera)**, not a square 960x960, because a square input would spend about 40% of the work on padding.

How the web app chooses: `best.engine` if present, otherwise the NCNN folder whose width matches `Cfg.imgsz` (960), otherwise `best.pt`. The Live page shows `pad_puller_v1 (NCNN 544x960)`. On this PC the engine still wins. To run NCNN on a PC, temporarily rename `best.engine`.

**Faster, slightly less accurate option:** `python export_ncnn.py --imgsz 640` makes `best_ncnn_model_384x640`, about 2x faster. To use it, set `imgsz: int = 640` in `Cfg` (`monitor.py`). See [section 13](#13-validation-results) for what it costs in accuracy.

**Setting up the Pi** (64-bit Raspberry Pi OS; a Pi 5 is recommended):

```
sudo apt install -y python3-venv git
git clone https://github.com/anand-g-2/pad-pull-monitor
cd pad-pull-monitor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
python -m webapp.app
```

Then open `http://localhost:5050` on the Pi's monitor, and use Calibration and Live Monitoring as on the PC. Copy `bg.png` only if you need the classic detector. On Linux the camera is opened through V4L2. `start_webapp.bat` is Windows-only.

---

## 5. Run the web app

**Option A: double-click** `start_webapp.bat`.

**Option B: from a terminal:**

```
.venv\Scripts\activate
python -m webapp.app
```

Then open **http://localhost:5050** in Chrome or Edge.

- Stop the app with **Ctrl + C** in its window, or close the window.
- By default the app is reachable **only from this PC**. To open it from another PC on the network:
  ```
  python -m webapp.app --host 0.0.0.0 --port 5050
  ```
  and browse to `http://<this-PC's-IP>:5050`. **The app has no login.** Only do this on a trusted network. Windows Firewall may ask to allow Python.
- Run **only one copy** of the app. Windows lets two copies listen on the same port, and then requests go to the wrong one.

To start it automatically with Windows, put a shortcut to `start_webapp.bat` in the Startup folder (press `Win + R`, type `shell:startup`).

---

## 6. Using the web app

### First time on a PC: Calibration

1. Open **Calibration** (left menu, under SETUP).
2. **Detect cameras**: thumbnails of every camera found appear after a few seconds. Click **Use this camera** under the one that sees the belt. It is saved in `camera.json`.
3. The **live preview** opens. The saved ROI, if any, is drawn in yellow.
4. Click **Capture frame to draw ROI**. Ideally capture a moment when a batch of pads is waiting at the sensor.
5. Draw the ROI polygon on the frozen frame:
   - **click** to place a corner;
   - **drag** a corner to move it;
   - **right-click** a corner to delete it;
   - **click on an edge** to insert a corner there;
   - **Undo last corner**, **Clear** and **Back to live** are available as buttons.

   **Include** the whole pad lane from the oven exit to the sensor, including the staging spot under the sensor. **Exclude** the oven shadow and the area where the pulling unit parks.
6. Click **Save ROI**. It is saved to `roi.json` as fractions of the frame, so it doesn't depend on resolution. The previous ROI is copied to `roi_backups/`, and the frame you drew on is saved as `roi_reference.jpg`.

Redo the calibration whenever the camera is moved, bumped or replaced. You can compare `roi_reference.jpg` with the live preview to tell whether it has moved.

### Daily use: Live Monitoring

- The top row shows what **Start** will use: **camera**, **ROI** and **model**. A red item means it is missing; use the link to fix it in Calibration.
- Press **Start**. The first start after booting can show "Starting…" for 1–2 minutes while the AI libraries load.
- The stream shows green pad boxes while pads arrive, red pad boxes during the post-pull check, and an orange box around the puller, inside the yellow ROI outline. The state and counts are in the panel beside the video.
- The right-hand panel shows the current state, pads in the ROI, processing fps, this run's counts and an event log.
- **Check the fps figure.** If the camera delivers less than about 8 fps, a warning appears; see [Troubleshooting](#12-troubleshooting).
- If the camera is unplugged, the app shows **Reconnecting…** and retries for 30 s.
- **Stop** ends monitoring. While monitoring runs, Calibration can't open the camera.

### NC alert

When an NC happens, a red pop-up appears on **any** page. It shows the batch, the reason and a snapshot.

1. Remove the pad left on the belt **before the next batch arrives**. Otherwise the next batch is also counted wrong (6 pads → NC).
2. The pop-up closes by itself after 5 seconds (click **OK** to close it sooner). The NC stays in the records.

### Overview and Analytics

- **Overview**: today's batches, pads counted, pads pulled and NC batches; hourly charts (batches OK/NC, pads pulled/left behind); the latest NC snapshots. It refreshes by itself.
- **Analytics**: choose **Day / Week / Month / Custom range**, plus *Day-wise* or *Avg hour-wise* for week and month. Shows the same charts and a table of every batch. **Download CSV** exports the table.

Records are kept in `webapp/analytics.db` (SQLite) and NC snapshots in `webapp/snapshots/`. Back up both if the history matters.

---

## 7. Testing without the line camera

Any recorded video of the line can stand in for the camera:

```
python -m webapp.app --test-video C:\path\to\recording.avi
```

The video plays at its own speed and loops. Calibration shows it as **"Test video"**, and the whole flow (assign, draw ROI, Start, NC alert, analytics) works as with a real camera. The video never appears as a choice in normal mode. Test runs are stored in `analytics.db` under the source name `Test video (…)`.

---

## 8. Command-line tools

These are useful for development and offline analysis, run from the project folder with the venv active.

| Command | What it does |
|---|---|
| `python monitor.py run VIDEO --model runs/pad_puller_v1/weights/best.pt --roi roi.json --out annotated.mp4 --csv batches.csv` | Runs the monitor on a video file, writes an annotated video and a CSV of batches, and prints each OK/NC. `--show` opens a preview window. `--model` also accepts a `.engine` or an NCNN folder. |
| `python monitor.py run VIDEO --bg bg.png --roi roi.json` | The same with the **classic** detector (no AI: dark blobs = pads), for comparison. It needs an empty-belt reference: `python monitor.py calibrate VIDEO --bg bg.png`. |
| `python test_nc_sim.py VIDEO --model ... --roi roi.json --out nc_sim.mp4` | **Simulated NC test:** pastes a real pad onto the belt after the 2nd pull, which must produce an NC. Use it after any change to the model or settings. |
| `python draw_roi.py VIDEO_OR_CAMERA --out roi.json` | Draws the ROI in an OpenCV window (the same file the web app writes). |
| `python export_tensorrt.py` | Builds the TensorRT engine ([section 4](#4-speed-up-with-tensorrt-recommended)). |

---

## 9. Retraining the model

Retrain when the camera position, lighting, pad type or pulling unit changes noticeably. The training dataset, the training preview images and `bg.png` are **not** in this repository: they contain client footage and are kept offline. Rebuild them from recordings:

1. **Record** a few minutes of the line from the camera position (several pull cycles).
2. **Build a reference image** of the empty belt (used only for draft labels):
   ```
   python monitor.py calibrate recording1.avi --bg bg.png
   ```
3. **Make draft labels** (pads and puller) from the recordings. One video is used for training and one for validation:
   ```
   python autolabel.py --train recording1.avi --val recording2.avi --bg bg.png --out dataset
   ```
4. **Review and correct the labels.** Frames with the puller in view are flagged, because pads next to it can't be labelled automatically:
   ```
   python review_labels.py --data dataset --flagged
   ```
   | Mouse / key | Action |
   |---|---|
   | left-drag | draw a box of the current class |
   | right-click | delete a box |
   | `1` / `2` | class: pad / puller |
   | `n`, Space, → | save and go to the next image |
   | `p`, ← | previous image |
   | `v` | paste the previous image's pad boxes |
   | `u` | undo |
   | `x` | delete all boxes |
   | `q`, Esc | save and quit (progress is kept) |
5. **Train** (about 40–60 min on a GTX 1650):
   ```
   python train.py --data dataset/data.yaml --epochs 60 --name pad_puller_v2
   ```
   On a 4 GB GPU keep `--batch 4` (the default). Batch 8 overflows the GPU memory and becomes about 10× slower.
6. **Validate** with `monitor.py run` on a recording and with `test_nc_sim.py`.
7. **Rebuild the exports**: `python export_tensorrt.py` (PC) and `python export_ncnn.py` (for the Pi). The web app always uses the newest `runs/*/weights/best.pt`, or an export built from it (TensorRT first, then NCNN).

---

## 10. Settings you can tune

These are in `monitor.py`, in the `Cfg` class.

| Setting | Default | Meaning |
|---|---|---|
| `expected_pads` | 5 | Pads per batch; any other count gives an NC |
| `nc_on_count_mismatch` | True | Set to False to judge only on leftover pads |
| `pad_conf` | 0.40 | Minimum YOLO confidence for a pad |
| `puller_conf_on` / `puller_conf_off` | 0.50 / 0.25 | Confidence for "puller arrived" / "puller left" |
| `puller_on_frames` / `puller_off_frames` | 2 / 3 | Frames needed to confirm the puller arrived / left |
| `stable_frames` | 6 | A count must hold this many frames before it is trusted |
| `settle_s` | 0.35 | Seconds ignored right after the puller leaves |
| `verify_s` | 2.0 | Length of the post-pull check in seconds. Must end before the next batch reaches the ROI (≥ 5 s on the recorded line). |
| `imgsz` | 960 | Model input size; must match training and the TensorRT engine |

Restart the web app after changing them.

---

## 11. Files and folders

```
pad_pull_monitor/
├─ webapp/                  web application (Flask)
│  ├─ app.py                routes, server start (python -m webapp.app)
│  ├─ runner.py             runs the monitor on the camera in a background thread
│  ├─ camera.py             camera open / probe / assignment (DirectShow, 1920×1080 MJPG)
│  ├─ camera_session.py     calibration preview session
│  ├─ calibration_routes.py Calibration page API (camera, preview, capture, ROI)
│  ├─ store.py              SQLite records + overview/analytics queries
│  ├─ templates/index.html  the single-page UI
│  ├─ static/               JS, CSS, logo
│  ├─ analytics.db          batch records            (created at runtime, not in Git)
│  └─ snapshots/            NC snapshots             (created at runtime, not in Git)
├─ monitor.py               detectors + OK/NC state machine (also a CLI)
├─ export_tensorrt.py       .pt → TensorRT engine (NVIDIA PCs)
├─ export_ncnn.py           .pt → NCNN model (Raspberry Pi / CPU) + accuracy check
├─ requirements-pi.txt      packages for the Raspberry Pi
├─ train.py                 YOLO training
├─ autolabel.py             draft labels from recordings
├─ review_labels.py         label review/correction tool
├─ draw_roi.py              ROI drawing tool (OpenCV window)
├─ test_nc_sim.py           simulated missed-pad test
├─ runs/pad_puller_v1/      trained model (weights/best.pt) + training plots
├─ roi.json                 ROI polygon (example from the recorded line; redraw on site)
├─ bg.png                   empty-belt reference for classic detector / autolabel (not in Git:
│                           create with  python monitor.py calibrate VIDEO --bg bg.png)
├─ camera.json              assigned camera index       (created on each PC, not in Git)
├─ roi_backups/, roi_reference.jpg                      (created on each PC, not in Git)
├─ requirements.txt
└─ start_webapp.bat
```

---

## 12. Troubleshooting

| Problem | Cause / fix |
|---|---|
| **Training or inference is extremely slow; `nvidia-smi -q -d POWER` shows "Current Power Limit: 5 W"** | The laptop firmware has capped the GPU. Plug in the **original** charger and set the Lenovo/OEM power mode to Performance (Fn+Q on Lenovo). On the development laptop only a **restart** cleared it. A healthy GPU shows ~50 W and P0 under load. |
| "Starting…" takes 1–2 minutes | The first import of PyTorch after a reboot is slow. Later starts are fast. |
| **Camera warning "delivers only N fps"** | The camera ignores the MJPG request and sends raw YUY2 at 1080p, which USB can't carry at full rate. Try another USB port (USB 3), a lower resolution with the same 16:9 shape, or another camera. The app needs about 10+ fps. |
| Detect cameras finds nothing / the camera isn't listed | The camera is in use by another program (Camera app, Teams, another copy of this app), or monitoring is running. Close the other program; stop monitoring first. |
| "Monitoring is running and is using the camera" in Calibration | Stop monitoring on the Live page first. A camera can only be opened once. |
| Page shows old buttons after an update | The browser cached the old files. Press **Ctrl + Shift + R**. |
| The page doesn't change after restarting the app | Two copies of the app are running on the same port. Close all of them (Task Manager → python.exe) and start one. |
| `numpy` errors after installing other packages ("compiled against NumPy 2", scipy errors) | Some packages (onnx → ml_dtypes) pull in numpy 2. Restore with: `pip install "numpy==1.26.4" "ml_dtypes>=0.5,<0.6"` |
| TensorRT engine fails to load ("serialization" / "version" errors) | The engine was built on another PC or with another TensorRT version. Delete it and run `python export_tensorrt.py` again. |
| `torch.cuda.is_available()` is False | The CPU build of PyTorch is installed. Reinstall with the `--index-url .../cu124` command in step 3.4. |
| Pad counts are wrong (e.g. 4 instead of 5) | The ROI cuts off part of the lane. Redraw it to include the oven exit and the staging spot at the sensor. |
| Many false NCs right after the pull | Increase `settle_s`, or check that the ROI excludes where the pulling unit parks. |

---

## 13. Validation results

Model `pad_puller_v1` (YOLO11n, 960 px, trained on recording `20261124162029`, validated on `20261124162403`; 603 hand-reviewed frames):

| Model | Precision | Recall | mAP50 | mAP50-95 | ms/frame* |
|---|---|---|---|---|---|
| PyTorch `.pt` | 0.980 | 0.980 | 0.993 | 0.901 | 32.2 |
| TensorRT FP16 `.engine` | 0.981 | 0.981 | 0.992 | 0.892 | 25.5 |

\*GTX 1650, YOLO prediction per frame including image loading and pre/post-processing. The full monitor on recording 1 (no video output) ran at 27–30 fps with `.pt` and 29 fps with TensorRT: the CPU is the limit there, not the model.

Monitor results on the full recordings (both `.pt` and TensorRT give identical decisions):

| Test | Result |
|---|---|
| Recording 1 | 7 batches, 35 pads, all OK |
| Recording 2 | 6 batches, 30 pads, all OK (7th cut off by the end of the file) |
| Simulated missed pad | NC raised for the affected batch ("1 pad left on belt after pull"), plus NC on the next batch (wrong count) |

**NCNN (for the Raspberry Pi).** Scored on the 267 validation images at the monitor's thresholds (pad 0.40, puller 0.50, IoU 0.5), on the PC's CPU (i7-10750H):

| Model | Pad precision | Pad recall | Puller precision | Puller recall | ms/frame (PC CPU) |
|---|---|---|---|---|---|
| PyTorch `.pt`, 960 | 0.958 | 0.975 | 0.991 | 0.991 | 137 |
| **NCNN 544x960** | 0.958 | 0.975 | 0.991 | 0.991 | 140 |
| PyTorch `.pt`, 640 | 0.946 | 0.957 | 0.991 | 0.982 | 81 |
| NCNN 384x640 | 0.946 | 0.957 | 0.991 | 0.982 | 66 |

NCNN gives exactly the same detections as PyTorch at both sizes. The 640 input misses 22 of the validation pads instead of 13; these are mostly the small, far pads. Ultralytics' own `val` (mAP) works for NCNN only with square inputs, which is why this table uses its own scoring. With the square 960x960 NCNN export, `val` gave mAP50 0.992 and mAP50-95 0.893.

The monitor with NCNN 544x960 on the full recordings, at the camera's 15 fps and with frames dropped to imitate a slower processor:

| Processing rate | Recording 1 | Recording 2 | Simulated missed pad |
|---|---|---|---|
| 15 fps | 7 OK | 6 OK | NC caught (+ count NC on the next batch) |
| 7.5 fps | 7 OK | 6 OK | NC caught (+ count NC on the next batch) |
| 5 fps | 7 OK | 6 OK | NC caught (+ count NC on the next batch) |
| 3 fps | 7 OK | 6 OK | NC caught |

The OK/NC logic holds down to 3 fps. **Pi speed has not been measured** (no Pi was available). A rough estimate for a Pi 5 is 120–150 ms per frame for the model alone, so about 4–6 fps for the whole monitor with the browser running on the same Pi. That is enough by the table above. A Pi 4 is roughly 3x slower and would need the 640 model. Check the fps on the Live page on the actual Pi; below about 3 fps, use the 640 model.

**Not yet validated:** a real missed pull. No footage of one exists. Before relying on the system in production, record and run at least one real or staged missed pull.
