import os
import sys
import time
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from yolo_onnx_detector import YOLOONNXDetector
from depth_onnx import DepthONNXEstimator
from pothole_measure import (
    estimate_pothole_depth,
    estimate_area_from_bbox,
    classify_severity,
)


def draw_result(frame, det, depth_cm, area_m2, severity, class_name="pothole"):
    x1, y1, x2, y2 = det["bbox"]
    score = det["score"]

    if severity == "minor":
        color = (0, 255, 0)
    elif severity == "moderate":
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label1 = f"{class_name} {score:.2f} | {severity}"
    label2 = f"depth={depth_cm:.2f}cm area={area_m2:.3f}m2"

    cv2.putText(
        frame,
        label1,
        (x1, max(25, y1 - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    cv2.putText(
        frame,
        label2,
        (x1, max(45, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    return frame


def main():
    cfg = Config()

    if not os.path.exists(cfg.yolo_onnx_path):
        raise FileNotFoundError(f"Missing YOLO ONNX: {cfg.yolo_onnx_path}")

    if not os.path.exists(cfg.depth_onnx_path):
        raise FileNotFoundError(f"Missing depth ONNX: {cfg.depth_onnx_path}")

    if not os.path.exists(cfg.input_video_path):
        raise FileNotFoundError(f"Missing input video: {cfg.input_video_path}")

    os.makedirs(os.path.dirname(cfg.output_video_path), exist_ok=True)

    print("Loading YOLO ONNX detector...")
    detector = YOLOONNXDetector(
        model_path=cfg.yolo_onnx_path,
        input_size=cfg.yolo_input_size,
        conf_thres=cfg.conf_thres,
        iou_thres=cfg.iou_thres,
    )

    print("Loading Depth ONNX estimator...")
    depth_estimator = DepthONNXEstimator(
        model_path=cfg.depth_onnx_path,
        input_size=cfg.depth_input_size,
    )

    cap = cv2.VideoCapture(cfg.input_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.input_video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps_in <= 0:
        fps_in = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        cfg.output_video_path,
        fourcc,
        fps_in,
        (width, height),
    )

    frame_idx = 0
    total_time = 0.0
    processed_frames = 0

    pbar = tqdm(total=total_frames, desc="Running Part A")

    last_detections = []
    last_depth_map = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start = time.time()

        if frame_idx % cfg.frame_skip == 0:
            detections = detector.detect(frame)
            depth_map = depth_estimator.estimate(frame)
            if frame_idx == 0:
                depth_vis = (depth_map * 255).astype("uint8")
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                cv2.imwrite("output/debug_depth_map.jpg", depth_vis)

            last_detections = detections
            last_depth_map = depth_map
        else:
            detections = last_detections
            depth_map = last_depth_map

        if depth_map is not None:
            for det in detections:
                bbox = det["bbox"]

                depth_cm = estimate_pothole_depth(
                    depth_map=depth_map,
                    bbox=bbox,
                    depth_scale_cm=cfg.depth_scale_cm,
                )

                area_m2 = estimate_area_from_bbox(
                    bbox=bbox,
                    meter_per_pixel=cfg.meter_per_pixel,
                )

                severity = classify_severity(depth_cm, area_m2)

                class_id = det["class_id"]
                if class_id < len(cfg.class_names):
                    class_name = cfg.class_names[class_id]
                else:
                    class_name = "pothole"

                frame = draw_result(
                    frame=frame,
                    det=det,
                    depth_cm=depth_cm,
                    area_m2=area_m2,
                    severity=severity,
                    class_name=class_name,
                )

        elapsed = time.time() - start
        total_time += elapsed
        processed_frames += 1

        current_fps = 1.0 / max(elapsed, 1e-6)
        avg_fps = processed_frames / max(total_time, 1e-6)

        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f} | AVG: {avg_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    avg_fps = processed_frames / max(total_time, 1e-6)

    print()
    print("===== PART A RESULT =====")
    print(f"Input video : {cfg.input_video_path}")
    print(f"Output video: {cfg.output_video_path}")
    print(f"Frames      : {processed_frames}")
    print(f"Average FPS : {avg_fps:.2f}")


if __name__ == "__main__":
    main()