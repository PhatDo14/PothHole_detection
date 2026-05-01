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
└── .gitignore[result_part_a.mp4](output/result_part_a.mp4)