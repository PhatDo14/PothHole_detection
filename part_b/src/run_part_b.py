import os
import sys
import csv
import time
import math
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_b import ConfigB
from gps_monitor import GPSIntegrityMonitor
from state_machine import GPSStateMachine
from visual_odometry import VisualOdometry
from lane_detection import LaneDetector
from landmark_database import LandmarkDatabase
from fusion_engine import FusionEngine
from metrics import PartBMetrics


def draw_text_panel(
    frame,
    gps_state,
    gps_quality,
    pose,
    lane_position,
    uturn_detected,
    fps,
    num_landmarks,
    num_features,
):
    h, w = frame.shape[:2]

    panel_h = 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    if gps_state == "GPS_GOOD":
        color = (0, 255, 0)
    elif gps_state == "GPS_DEGRADED":
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    x, y, theta = pose.x, pose.y, math.degrees(pose.theta)

    lines = [
        f"GPS State   : {gps_state} | Quality: {gps_quality}",
        f"Pose        : x={x:.2f}m, y={y:.2f}m, theta={theta:.1f}deg",
        f"Lane        : {lane_position}",
        f"U-turn      : {'YES' if uturn_detected else 'NO'}",
        f"Landmarks   : {num_landmarks}",
        f"VO Features : {num_features}",
        f"FPS         : {fps:.1f}",
    ]

    y0 = 30
    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, y0 + idx * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color if idx == 0 else (255, 255, 255),
            2,
        )

    return frame


def draw_trajectory_mini_map(frame, trajectory, gps_state):
    h, w = frame.shape[:2]

    map_w = 260
    map_h = 260

    x0 = w - map_w - 20
    y0 = h - map_h - 20

    cv2.rectangle(frame, (x0, y0), (x0 + map_w, y0 + map_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + map_w, y0 + map_h), (255, 255, 255), 1)

    if len(trajectory) < 2:
        return frame

    xs = np.array([p[0] for p in trajectory])
    ys = np.array([p[1] for p in trajectory])

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    scale_x = map_w / max(max_x - min_x, 1e-6)
    scale_y = map_h / max(max_y - min_y, 1e-6)
    scale = 0.8 * min(scale_x, scale_y)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    def to_map(x, y):
        mx = int(x0 + map_w / 2 + (x - cx) * scale)
        my = int(y0 + map_h / 2 - (y - cy) * scale)
        return mx, my

    for i in range(1, len(trajectory)):
        p1 = to_map(trajectory[i - 1][0], trajectory[i - 1][1])
        p2 = to_map(trajectory[i][0], trajectory[i][1])

        if gps_state == "GPS_GOOD":
            color = (0, 255, 0)
        elif gps_state == "GPS_DEGRADED":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.line(frame, p1, p2, color, 2)

    last = trajectory[-1]
    mx, my = to_map(last[0], last[1])
    cv2.circle(frame, (mx, my), 5, (255, 255, 255), -1)

    cv2.putText(
        frame,
        "Trajectory",
        (x0 + 10, y0 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return frame


def detect_uturn(theta_history, timestamp_history, threshold_deg, window_sec):
    if len(theta_history) < 2:
        return False

    current_t = timestamp_history[-1]
    current_theta = theta_history[-1]

    for theta, t in zip(theta_history, timestamp_history):
        if current_t - t <= window_sec:
            delta = abs(math.degrees(current_theta - theta))
            delta = min(delta, 360 - delta)

            if delta >= threshold_deg:
                return True

    return False


def main():
    cfg = ConfigB()

    os.makedirs(os.path.dirname(cfg.output_video_path), exist_ok=True)

    if not os.path.exists(cfg.input_video_path):
        raise FileNotFoundError(f"Missing input video: {cfg.input_video_path}")

    if not os.path.exists(cfg.gps_log_path):
        raise FileNotFoundError(
            f"Missing GPS log: {cfg.gps_log_path}. "
            f"Run: python part_b/src/generate_gps_log.py"
        )

    gps_monitor = GPSIntegrityMonitor(
        gps_log_path=cfg.gps_log_path,
        good_hdop_threshold=cfg.good_hdop_threshold,
        min_satellites_good=cfg.min_satellites_good,
        min_snr_good=cfg.min_snr_good,
    )

    first_valid_gps = next(
        (m for m in gps_monitor.measurements if m.lat is not None and m.lon is not None),
        None,
    )

    if first_valid_gps is None:
        raise RuntimeError("No valid GPS origin found in GPS log.")

    origin_lat = first_valid_gps.lat
    origin_lon = first_valid_gps.lon

    state_machine = GPSStateMachine(
        relock_stable_frames=cfg.gps_relock_stable_frames,
    )

    vo = VisualOdometry(
        max_features=cfg.max_features,
        quality_level=cfg.quality_level,
        min_distance=cfg.min_distance,
        vo_scale=cfg.vo_scale,
    )

    lane_detector = LaneDetector(
        roi_y_ratio=cfg.lane_roi_y_ratio,
    )

    landmark_db = LandmarkDatabase(
        match_distance_threshold=5.0,
    )

    fusion = FusionEngine(
        gps_good_weight=cfg.gps_good_weight,
        gps_degraded_weight=cfg.gps_degraded_weight,
        gps_lost_weight=cfg.gps_lost_weight,
    )

    metrics = PartBMetrics()

    cap = cv2.VideoCapture(cfg.input_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.input_video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        cfg.output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )

    trajectory = []
    theta_history = deque(maxlen=int(fps_in * cfg.uturn_window_sec) + 5)
    timestamp_history = deque(maxlen=int(fps_in * cfg.uturn_window_sec) + 5)

    trajectory_logs = []
    gps_state_logs = []

    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Running Part B")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start = time.time()
        timestamp = frame_idx / fps_in

        gps_measurement = gps_monitor.get_measurement_by_time(timestamp)
        gps_quality = gps_monitor.classify_quality(gps_measurement)
        gps_state, transition = state_machine.update(gps_quality)

        gps_xy = gps_monitor.gps_to_local_xy(
            gps_measurement,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )

        vo_x, vo_y, vo_theta, dx_world, dy_world, num_features = vo.update(frame)

        lane_position, lane_offset, lane_debug = lane_detector.detect(frame)

        fused_pose = fusion.update(
            vo_pose=(vo_x, vo_y, vo_theta),
            gps_xy=gps_xy,
            gps_state=gps_state,
        )

        # Simulated landmark: every 5 meters, create/update a road landmark.
        # In a full system, this can be replaced by traffic sign / pothole detector.
        if int(abs(fused_pose.x)) % 5 == 0:
            landmark_db.add_or_update(
                class_name="road_landmark",
                x=fused_pose.x,
                y=fused_pose.y,
                timestamp=timestamp,
            )

        theta_history.append(fused_pose.theta)
        timestamp_history.append(timestamp)

        uturn_detected = detect_uturn(
            theta_history=theta_history,
            timestamp_history=timestamp_history,
            threshold_deg=cfg.uturn_angle_threshold_deg,
            window_sec=cfg.uturn_window_sec,
        )

        trajectory.append((fused_pose.x, fused_pose.y, fused_pose.theta))

        frame = lane_detector.draw(frame, lane_debug)

        elapsed = time.time() - start
        current_fps = 1.0 / max(elapsed, 1e-6)

        frame = draw_text_panel(
            frame=frame,
            gps_state=gps_state,
            gps_quality=gps_quality,
            pose=fused_pose,
            lane_position=lane_position,
            uturn_detected=uturn_detected,
            fps=current_fps,
            num_landmarks=len(landmark_db.landmarks),
            num_features=num_features,
        )

        frame = draw_trajectory_mini_map(frame, trajectory, gps_state)

        writer.write(frame)

        metrics.update(
            fps=current_fps,
            gps_state=gps_state,
            lane_position=lane_position,
            uturn_detected=uturn_detected,
        )

        trajectory_logs.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "x": fused_pose.x,
                "y": fused_pose.y,
                "theta_rad": fused_pose.theta,
                "theta_deg": math.degrees(fused_pose.theta),
                "gps_state": gps_state,
                "gps_quality": gps_quality,
                "lane_position": lane_position,
                "lane_offset": lane_offset,
                "uturn_detected": int(uturn_detected),
                "num_features": num_features,
                "num_landmarks": len(landmark_db.landmarks),
                "fps": current_fps,
            }
        )

        gps_state_logs.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "gps_quality": gps_quality,
                "gps_state": gps_state,
                "hdop": gps_measurement.hdop,
                "num_satellites": gps_measurement.num_satellites,
                "snr": gps_measurement.snr,
                "transition": int(transition),
            }
        )

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    # Save trajectory CSV
    with open(cfg.output_trajectory_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_idx",
            "timestamp",
            "x",
            "y",
            "theta_rad",
            "theta_deg",
            "gps_state",
            "gps_quality",
            "lane_position",
            "lane_offset",
            "uturn_detected",
            "num_features",
            "num_landmarks",
            "fps",
        ]

        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(trajectory_logs)

    # Save GPS state CSV
    with open(cfg.output_gps_state_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_idx",
            "timestamp",
            "gps_quality",
            "gps_state",
            "hdop",
            "num_satellites",
            "snr",
            "transition",
        ]

        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(gps_state_logs)

    metrics.save(cfg.output_metrics_txt)

    print()
    print("===== PART B RESULT =====")
    print(f"Output video      : {cfg.output_video_path}")
    print(f"Trajectory CSV    : {cfg.output_trajectory_csv}")
    print(f"GPS state CSV     : {cfg.output_gps_state_csv}")
    print(f"Metrics TXT       : {cfg.output_metrics_txt}")
    print("Done.")


if __name__ == "__main__":
    main()