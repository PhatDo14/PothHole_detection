from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class ConfigB:
    # Input / Output
    input_video_path: str = str(ROOT_DIR / "input" / "test_video.mp4")
    gps_log_path: str = str(ROOT_DIR / "input" / "gps_log.csv")

    output_video_path: str = str(ROOT_DIR / "output" / "result_part_b.mp4")
    output_trajectory_csv: str = str(ROOT_DIR / "output" / "trajectory.csv")
    output_gps_state_csv: str = str(ROOT_DIR / "output" / "gps_state_log.csv")
    output_metrics_txt: str = str(ROOT_DIR / "output" / "part_b_metrics.txt")

    # GPS thresholds
    good_hdop_threshold: float = 5.0
    min_satellites_good: int = 4
    min_snr_good: float = 20.0

    gps_lost_satellites: int = 0
    gps_relock_stable_frames: int = 10

    # Visual odometry
    vo_scale: float = 0.02
    max_features: int = 300
    quality_level: float = 0.01
    min_distance: int = 7

    # Lane detection
    lane_roi_y_ratio: float = 0.55

    # U-turn
    uturn_angle_threshold_deg: float = 150.0
    uturn_window_sec: float = 2.0

    # Fusion weights
    gps_good_weight: float = 0.80
    gps_degraded_weight: float = 0.30
    gps_lost_weight: float = 0.0

    # Runtime
    draw_scale: float = 1.0