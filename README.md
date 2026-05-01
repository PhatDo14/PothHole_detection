# MET EV Vision Test — Pothole Detection & GPS-Degraded Visual Localization

This repository contains the implementation for the **AI Engineer — MET EV Vision Test**, including:

- **Part A — Pothole Detection & Depth/Area Estimation**
- **Part B — GPS-Degraded Visual Localization**

The project is designed as a CPU-based Computer Vision prototype for electric vehicle road perception and localization.

---

## 1. Project Overview

This repository includes two main systems.

### Part A — Pothole Detection & Depth/Area Estimation

Part A detects potholes from road videos, estimates relative pothole depth and surface area, classifies severity levels, and generates an annotated output video with runtime metrics.

Main components:

- Fine-tuned YOLOv8s pothole detector
- ONNX Runtime CPU inference
- Monocular relative depth estimation using Depth Anything V2
- Pothole surface area estimation
- Rule-based severity classification
- Output video visualization
- Runtime and per-frame metrics logging

---

### Part B — GPS-Degraded Visual Localization

Part B demonstrates a prototype GPS fallback localization system. When GPS becomes degraded or lost, the system continues estimating vehicle pose using visual odometry, GPS state monitoring, lane/road-boundary fallback, landmark database logic, and EKF-inspired fusion.

Main components:

- GPS integrity monitoring
- GPS state machine: `GPS_GOOD`, `GPS_DEGRADED`, `GPS_LOST`
- Visual Odometry using Lucas-Kanade optical flow
- Lane detection / unstructured-road fallback
- Landmark database prototype
- EKF-inspired pose fusion
- U-turn detection logic
- Output video, trajectory CSV, GPS state log, and metrics report

---

## 2. Key Features

### Part A

- Fine-tuned YOLOv8s pothole detector
- ONNX Runtime CPU deployment
- Monocular relative depth estimation
- Pothole area estimation
- Severity classification:
  - `minor`
  - `moderate`
  - `severe`
- Annotated video output
- Summary metrics report
- Per-frame CSV logging
- Baseline vs fine-tuned validation comparison
- Failure analysis and improvement roadmap

### Part B

- GPS quality classification using HDOP, number of satellites, and SNR
- State-machine-based GPS handover logic
- Visual odometry fallback when GPS is degraded/lost
- Lane detection with fallback for unstructured roads
- Landmark database prototype
- EKF-inspired weighted fusion
- U-turn detection logic
- Trajectory mini-map visualization
- Runtime metrics and state logs
- End-to-end CPU processing

---

## 3. Project Structure

```text
PothHole_detection/
│
├── src/
│   ├── config.py
│   ├── yolo_onnx_detector.py
│   ├── depth_onnx.py
│   ├── pothole_measure.py
│   └── run_part_a.py
│
├── part_b/
│   ├── src/
│   │   ├── config_b.py
│   │   ├── generate_gps_log.py
│   │   ├── gps_monitor.py
│   │   ├── state_machine.py
│   │   ├── visual_odometry.py
│   │   ├── lane_detection.py
│   │   ├── landmark_database.py
│   │   ├── fusion_engine.py
│   │   ├── metrics.py
│   │   └── run_part_b.py
│   │
│   ├── input/
│   │   ├── test_video.mp4
│   │   └── gps_log.csv
│   │
│   ├── output/
│   │   ├── result_part_b.mp4
│   │   ├── trajectory.csv
│   │   ├── gps_state_log.csv
│   │   └── part_b_metrics.txt
│   │
│   ├── configs/
│   └── README_PART_B.md
│
├── models/
│   ├── pothole_detector.onnx
│   ├── depth_anything_v2.onnx
│   └── best_finetune_200epochs.pt
│
├── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── input/
│   └── test_video.mp4
│
├── output/
│   ├── result_part_a.mp4
│   ├── part_a_metrics.txt
│   └── part_a_frame_metrics.csv
│
├── Report_Part_A.pdf
├── Report_Part_B.pdf
├── download_models.py
├── export_yolo_to_onnx.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Environment Setup

### 4.1 Clone Repository

```bash
git clone https://github.com/PhatDo14/PothHole_detection.git
cd PothHole_detection
```

---

### 4.2 Create Virtual Environment

Windows CMD:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Check Python version:

```bash
python --version
```

Recommended:

```text
Python 3.10 or Python 3.11
```

---

### 4.3 Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Main packages:

| Package | Purpose |
|---|---|
| `opencv-python` | Read/write video, image processing, visualization, optical flow |
| `numpy` | Numerical processing |
| `onnxruntime` | Run ONNX models on CPU |
| `ultralytics` | YOLO training, validation, and export |
| `tqdm` | Progress bar |
| `huggingface_hub` | Download baseline models if needed |
| `roboflow` | Download Roboflow dataset if needed |
| `onnx` | ONNX model checking and utilities |

---

## 5. Model Files

The project uses two main ONNX models for Part A:

```text
models/pothole_detector.onnx
models/depth_anything_v2.onnx
```

Expected model structure:

```text
models/
├── pothole_detector.onnx
├── depth_anything_v2.onnx
└── best_finetune_200epochs.pt
```

### Model Description

| Model | Path | Description |
|---|---|---|
| Pothole Detector | `models/pothole_detector.onnx` | Fine-tuned YOLOv8s detector exported to ONNX |
| Depth Estimator | `models/depth_anything_v2.onnx` | Depth Anything V2 Small ONNX model |
| Fine-tuned Checkpoint | `models/best_finetune_200epochs.pt` | PyTorch checkpoint used for validation/training reference |

Important:

```text
The pothole_detector.onnx model was exported with input size 448.
Therefore, src/config.py must use yolo_input_size = 448.
```

If model files are missing, run:

```bash
python download_models.py
python export_yolo_to_onnx.py
```

Note:

The automatically downloaded YOLO model is the baseline model. For best performance, use the provided fine-tuned ONNX detector.

---

# Part A — Pothole Detection & Depth/Area Estimation

## 6. Part A Overview

Part A builds an end-to-end road pothole analysis pipeline.

The system:

1. Reads a road video.
2. Detects potholes using YOLOv8s ONNX.
3. Estimates monocular relative depth using Depth Anything V2.
4. Estimates pothole area from bounding boxes.
5. Classifies severity into `minor`, `moderate`, and `severe`.
6. Draws results on output video.
7. Saves runtime metrics and per-frame logs.

---

## 7. Part A Input and Output

### Input Video

Place your Part A test video at:

```text
input/test_video.mp4
```

If the `input/` folder does not exist:

```bash
mkdir input
```

If your video has a different name, either rename it to:

```text
test_video.mp4
```

or update the path in:

```text
src/config.py
```

```python
input_video_path = "input/test_video.mp4"
```

---

### Output Files

After running Part A, results are saved to:

```text
output/
```

Expected outputs:

```text
output/result_part_a.mp4
output/part_a_metrics.txt
output/part_a_frame_metrics.csv
```

| File | Description |
|---|---|
| `output/result_part_a.mp4` | Annotated video with bbox, confidence, depth, area, severity, and FPS |
| `output/part_a_metrics.txt` | Summary metrics for the whole video |
| `output/part_a_frame_metrics.csv` | Per-frame runtime and detection logs |

---

## 8. Part A Configuration

Open:

```text
src/config.py
```

Important settings:

```python
yolo_onnx_path = "models/pothole_detector.onnx"
depth_onnx_path = "models/depth_anything_v2.onnx"

input_video_path = "input/test_video.mp4"
output_video_path = "output/result_part_a.mp4"

yolo_input_size = 448
conf_thres = 0.25
iou_thres = 0.45

depth_input_size = 256
depth_scale_cm = 300.0
meter_per_pixel = 0.003

frame_skip = 2
```

Important:

```text
yolo_input_size must match the input size used when exporting pothole_detector.onnx.
For the provided model, use yolo_input_size = 448.
```

---

## 9. Run Part A Inference

Run:

```bash
python src/run_part_a.py
```

The script will:

1. Load the YOLO ONNX pothole detector.
2. Load the Depth Anything V2 ONNX model.
3. Read `input/test_video.mp4`.
4. Detect potholes.
5. Estimate relative depth.
6. Estimate pothole area.
7. Classify severity.
8. Draw results on video frames.
9. Save output video and metrics.

---

## 10. Part A Validation

If the dataset is included under:

```text
data/
```

with this structure:

```text
data/
├── train/
├── valid/
├── test/
└── data.yaml
```

you can validate the fine-tuned detector:

```bash
yolo detect val model=models/best_finetune_200epochs.pt data=data/data.yaml imgsz=448 project=runs/detect name=val_yolov8s_finetune_448
```

Or validate the ONNX detector:

```bash
yolo detect val model=models/pothole_detector.onnx data=data/data.yaml imgsz=448 project=runs/detect name=val_onnx_detector_448
```

Best validation result from the fine-tuned model:

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Input Size |
|---|---:|---:|---:|---:|---:|
| Fine-tuned YOLOv8s | 0.781 | 0.713 | 0.789 | 0.482 | 448 |

Baseline result before fine-tuning:

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Input Size |
|---|---:|---:|---:|---:|---:|
| Baseline YOLOv8s | 0.592 | 0.455 | 0.457 | 0.188 | 320 |

The fine-tuned model improved mAP@0.5 from:

```text
45.7% → 78.9%
```

---

## 11. Part A Notes

The depth value is a calibrated relative depth estimate, not an absolute physical measurement.

For production-level metric depth, the system should use:

- stereo camera calibration,
- known camera height,
- ground-plane estimation,
- or measured depth ground truth.

The current area estimation is based on bounding boxes. A segmentation model would improve area accuracy.

---

# Part B — GPS-Degraded Visual Localization

## 12. Part B Overview

Part B implements a prototype localization fallback system for GPS-degraded environments.

The pipeline demonstrates:

1. GPS integrity monitoring.
2. GPS state classification.
3. State-machine handover logic.
4. Visual Odometry fallback.
5. Lane detection / unstructured-road fallback.
6. Landmark database management.
7. EKF-inspired pose fusion.
8. Output trajectory and state logging.

---

## 13. Part B Input and Output

### Input Files

Part B requires:

```text
part_b/input/test_video.mp4
part_b/input/gps_log.csv
```

`test_video.mp4` is the road video.

`gps_log.csv` contains timestamped GPS data:

```text
timestamp,lat,lon,hdop,num_satellites,snr
```

If you do not have a GPS log, generate a synthetic GPS log:

```bash
python part_b/src/generate_gps_log.py
```

This creates:

```text
part_b/input/gps_log.csv
```

The synthetic GPS scenario includes:

```text
0–15s   : GPS_GOOD
15–25s  : GPS_DEGRADED
25–40s  : GPS_LOST
40s+    : GPS_GOOD again
```

---

### Output Files

After running Part B, outputs are saved under:

```text
part_b/output/
```

Expected outputs:

```text
part_b/output/result_part_b.mp4
part_b/output/trajectory.csv
part_b/output/gps_state_log.csv
part_b/output/part_b_metrics.txt
```

| File | Description |
|---|---|
| `part_b/output/result_part_b.mp4` | Output video with GPS state, pose, lane state, VO features, FPS, and trajectory mini-map |
| `part_b/output/trajectory.csv` | Per-frame pose and trajectory log |
| `part_b/output/gps_state_log.csv` | GPS quality/state/transition log |
| `part_b/output/part_b_metrics.txt` | Runtime and system metrics summary |

---

## 14. Part B Configuration

Open:

```text
part_b/src/config_b.py
```

Important settings:

```python
input_video_path = "part_b/input/test_video.mp4"
gps_log_path = "part_b/input/gps_log.csv"

output_video_path = "part_b/output/result_part_b.mp4"
output_trajectory_csv = "part_b/output/trajectory.csv"
output_gps_state_csv = "part_b/output/gps_state_log.csv"
output_metrics_txt = "part_b/output/part_b_metrics.txt"

good_hdop_threshold = 5.0
min_satellites_good = 4
min_snr_good = 20.0

vo_scale = 0.02
max_features = 300

gps_good_weight = 0.80
gps_degraded_weight = 0.30
gps_lost_weight = 0.0
```

---

## 15. Run Part B

### Step 1 — Prepare input video

Create the input folder if needed:

```bash
mkdir part_b\input
```

Copy your video to:

```text
part_b/input/test_video.mp4
```

---

### Step 2 — Generate GPS log

```bash
python part_b/src/generate_gps_log.py
```

This generates:

```text
part_b/input/gps_log.csv
```

---

### Step 3 — Run Part B pipeline

```bash
python part_b/src/run_part_b.py
```

The script will:

1. Load input video.
2. Load GPS log.
3. Classify GPS quality.
4. Update GPS state machine.
5. Estimate visual odometry.
6. Detect lane or switch to unstructured-road fallback.
7. Update landmark database.
8. Fuse GPS and VO pose.
9. Draw overlay and trajectory mini-map.
10. Save video, CSV logs, and metrics.

---

## 16. Part B Results

Metrics from the current demo:

```text
===== PART B METRICS =====

[Runtime]
Total frames        : 692
Average FPS         : 27.45
Min FPS             : 1.70
Max FPS             : 38.46

[GPS State Distribution]
GPS_GOOD frames     : 374
GPS_DEGRADED frames : 250
GPS_LOST frames     : 68

[Events]
U-turn events       : 0

[Lane Distribution]
unstructured_road   : 669
left_or_center      : 23
```

### Part B Acceptance Discussion

| ID | Criterion | Target | Prototype Result |
|---|---|---|---|
| B1 | Visual Odometry drift / 500m | ≤ 5% | VO trajectory generated; true drift requires ground-truth trajectory |
| B2 | Landmark re-identification | Recall ≥ 85% | Landmark database prototype implemented; recall requires landmark ID ground truth |
| B3 | U-turn detection latency | ≤ 2s | Logic implemented; current demo has no U-turn event |
| B4 | Lane position accuracy | ≥ 90% | Road has no clear lane marking; system uses `unstructured_road` fallback |
| B5 | Localization under GPS loss | Demo active | GPS degraded/lost simulated; pose continues using VO/fusion |
| B6 | GPS handover latency | ≤ 2s | State machine transitions logged frame-by-frame |
| B7 | End-to-end FPS | ≥ 15 FPS | Average FPS = 27.45, target satisfied |
| B8 | GPS re-lock correction | Error ≤ 5m | Fusion correction behavior implemented; absolute error requires ground truth |

---

## 17. Part B Notes

The current Part B implementation is a prototype-level GPS fallback localization system.

It demonstrates:

- GPS state monitoring,
- GPS degraded/lost scenario simulation,
- visual odometry fallback,
- lane fallback for unstructured roads,
- landmark database concept,
- EKF-inspired fusion,
- real-time CPU performance.

Some metrics require ground truth for strict quantitative validation:

- VO drift over 500m,
- landmark recall,
- lane accuracy,
- GPS re-lock absolute error.

---

# Reports

The repository includes technical reports:

```text
Report_Part_A.pdf
Report_Part_B.pdf
```

These reports include:

- system architecture,
- setup instructions,
- benchmark results,
- validation discussion,
- failure analysis,
- improvement roadmap.

---

# Quick Commands

## Full Setup

```bash
git clone https://github.com/PhatDo14/PothHole_detection.git
cd PothHole_detection

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run Part A

```bash
mkdir input
```

Place video at:

```text
input/test_video.mp4
```

Run:

```bash
python src/run_part_a.py
```

Check outputs:

```text
output/result_part_a.mp4
output/part_a_metrics.txt
output/part_a_frame_metrics.csv
```

---

## Run Part B

```bash
mkdir part_b\input
```

Place video at:

```text
part_b/input/test_video.mp4
```

Generate GPS log:

```bash
python part_b/src/generate_gps_log.py
```

Run:

```bash
python part_b/src/run_part_b.py
```

Check outputs:

```text
part_b/output/result_part_b.mp4
part_b/output/trajectory.csv
part_b/output/gps_state_log.csv
part_b/output/part_b_metrics.txt
```

---

# Common Issues

## ONNX input size error in Part A

If you see:

```text
Got invalid dimensions for input
Expected: 448
Got: 320
```

open:

```text
src/config.py
```

and set:

```python
yolo_input_size = 448
```

The input size in config must match the ONNX detector export size.

---

## Missing Part A input video

If you see:

```text
Missing input video: input/test_video.mp4
```

create the folder:

```bash
mkdir input
```

then put your video at:

```text
input/test_video.mp4
```

---

## Missing Part B GPS log

If you see:

```text
Missing GPS log: part_b/input/gps_log.csv
```

run:

```bash
python part_b/src/generate_gps_log.py
```

---

## Missing model files

If you see:

```text
Missing YOLO ONNX
```

or:

```text
Missing depth ONNX
```

make sure these files exist:

```text
models/pothole_detector.onnx
models/depth_anything_v2.onnx
```

If missing, run:

```bash
python download_models.py
python export_yolo_to_onnx.py
```

---

# Notes for Evaluators

- Part A uses a fine-tuned YOLOv8s detector exported to ONNX.
- Part A best validation result: 78.9% mAP@0.5 at input size 448.
- Part A depth is calibrated relative depth, not absolute metric depth.
- Part B demonstrates GPS degraded/lost fallback behavior.
- Part B average end-to-end FPS is 27.45 on CPU.
- Part B includes GPS state logs, trajectory CSV, and output video visualization.
- Some Part B metrics require additional ground truth for strict quantitative validation.

---

# Author

```text
Do Tan Phat
AI Engineer
```