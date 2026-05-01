# Pothole Detection and Depth Estimation — MET EV  Test

This repository contains the implementation for **Part A — Pothole Detection & Depth/Area Estimation** of the AI Engineer EV Vision Test.

The system detects potholes from road videos, estimates relative pothole depth and surface area, classifies severity levels, and generates an annotated output video with runtime metrics.

---

## 1. Project Overview

The goal of this project is to build an end-to-end CPU-based Computer Vision pipeline for road pothole analysis.

The pipeline includes:

- Pothole detection using a fine-tuned YOLOv8 model
- ONNX Runtime CPU inference
- Monocular relative depth estimation using Depth Anything V2
- Pothole surface area estimation
- Rule-based severity classification
- Output video visualization
- Runtime and per-frame metrics logging

The system is designed to run on CPU only, following the test requirement that all AI models must be exported to `.onnx` and executed using ONNX Runtime.

---

## 2. Key Features

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
├── download_models.py
├── export_yolo_to_onnx.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Quick Start: Clone, Setup, and Run

This section shows how to clone the repository, set up the environment, prepare the input video, and run the full Part A pipeline using the provided ONNX models.

---

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
| `opencv-python` | Read/write video and draw visualization |
| `numpy` | Numerical processing |
| `onnxruntime` | Run ONNX models on CPU |
| `ultralytics` | YOLO validation/training/export |
| `tqdm` | Progress bar |
| `huggingface_hub` | Download baseline models if needed |
| `roboflow` | Download Roboflow dataset if needed |

---

### 4.4 Check Model Files

The project uses two ONNX models:

```text
models/pothole_detector.onnx
models/depth_anything_v2.onnx
```

Expected structure:

```text
models/
├── pothole_detector.onnx
└── depth_anything_v2.onnx
```

`pothole_detector.onnx` is the fine-tuned YOLOv8s pothole detector.  
`depth_anything_v2.onnx` is the monocular relative depth estimation model.

Important:

```text
The detector ONNX model was exported with input size 448.
Therefore, src/config.py must use yolo_input_size = 448.
```

If these model files are missing, run:

```bash
python download_models.py
python export_yolo_to_onnx.py
```

Note: the automatically downloaded YOLO model is the baseline model. For best performance, use the provided fine-tuned ONNX detector.

---

### 4.5 Prepare Input Video

Put your test video at:

```text
input/test_video.mp4
```

If the `input/` folder does not exist, create it:

```bash
mkdir input
```

Then copy your video into:

```text
PothHole_detection/input/test_video.mp4
```

If your video has a different name, either rename it to:

```text
test_video.mp4
```

or update this path in `src/config.py`:

```python
input_video_path = "input/test_video.mp4"
```

---

### 4.6 Configuration

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

Important note:

```text
yolo_input_size must match the input size used when exporting pothole_detector.onnx.
For the provided model, use yolo_input_size = 448.
```

---

### 4.7 Run Inference

Run the full pipeline:

```bash
python src/run_part_a.py
```

The script will:

1. Load the YOLO ONNX pothole detector
2. Load the Depth Anything V2 ONNX model
3. Read `input/test_video.mp4`
4. Detect potholes
5. Estimate relative depth
6. Estimate pothole area
7. Classify severity: `minor`, `moderate`, `severe`
8. Draw results on video frames
9. Save output video and metrics

---

### 4.8 Output Files

After running inference, results are saved to:

```text
output/
```

Expected outputs:

```text
output/result_part_a.mp4
output/part_a_metrics.txt
output/part_a_frame_metrics.csv
```

Output meaning:

| File | Description |
|---|---|
| `output/result_part_a.mp4` | Annotated video with bbox, confidence, depth, area, severity, and FPS |
| `output/part_a_metrics.txt` | Summary metrics for the whole video |
| `output/part_a_frame_metrics.csv` | Per-frame runtime and detection logs |

The output video can be opened directly after inference:

```text
output/result_part_a.mp4
```

---

### 4.9 Example End-to-End Commands

```bash
git clone https://github.com/PhatDo14/PothHole_detection.git
cd PothHole_detection

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir input
```

Copy your video to:

```text
input/test_video.mp4
```

Run:

```bash
python src/run_part_a.py
```

Check results:

```text
output/result_part_a.mp4
output/part_a_metrics.txt
output/part_a_frame_metrics.csv
```

---

## 5. Run Validation on Dataset

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

you can validate the detector using:

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

---

## 6. Common Issues

### ONNX input size error

If you see an error similar to:

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

The input size in config must match the size used to export the ONNX detector.

---

### Missing input video

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

### Missing model files

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

