import numpy as np


class PartBMetrics:
    def __init__(self):
        self.fps_values = []
        self.gps_states = []
        self.lane_positions = []
        self.uturn_events = 0
        self.handover_latencies = []
        self.total_frames = 0

        self.gps_lost_count = 0
        self.gps_degraded_count = 0
        self.gps_good_count = 0

    def update(self, fps, gps_state, lane_position, uturn_detected):
        self.total_frames += 1
        self.fps_values.append(fps)
        self.gps_states.append(gps_state)
        self.lane_positions.append(lane_position)

        if gps_state == "GPS_GOOD":
            self.gps_good_count += 1
        elif gps_state == "GPS_DEGRADED":
            self.gps_degraded_count += 1
        elif gps_state == "GPS_LOST":
            self.gps_lost_count += 1

        if uturn_detected:
            self.uturn_events += 1

    def summary(self):
        avg_fps = float(np.mean(self.fps_values)) if self.fps_values else 0.0
        min_fps = float(np.min(self.fps_values)) if self.fps_values else 0.0
        max_fps = float(np.max(self.fps_values)) if self.fps_values else 0.0

        lane_distribution = {}

        for lane in self.lane_positions:
            lane_distribution[lane] = lane_distribution.get(lane, 0) + 1

        return {
            "total_frames": self.total_frames,
            "avg_fps": avg_fps,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "gps_good_count": self.gps_good_count,
            "gps_degraded_count": self.gps_degraded_count,
            "gps_lost_count": self.gps_lost_count,
            "uturn_events": self.uturn_events,
            "lane_distribution": lane_distribution,
        }

    def save(self, path):
        summary = self.summary()

        with open(path, "w", encoding="utf-8") as f:
            f.write("===== PART B METRICS =====\n\n")

            f.write("[Runtime]\n")
            f.write(f"Total frames        : {summary['total_frames']}\n")
            f.write(f"Average FPS         : {summary['avg_fps']:.2f}\n")
            f.write(f"Min FPS             : {summary['min_fps']:.2f}\n")
            f.write(f"Max FPS             : {summary['max_fps']:.2f}\n\n")

            f.write("[GPS State Distribution]\n")
            f.write(f"GPS_GOOD frames     : {summary['gps_good_count']}\n")
            f.write(f"GPS_DEGRADED frames : {summary['gps_degraded_count']}\n")
            f.write(f"GPS_LOST frames     : {summary['gps_lost_count']}\n\n")

            f.write("[Events]\n")
            f.write(f"U-turn events       : {summary['uturn_events']}\n\n")

            f.write("[Lane Distribution]\n")
            for lane, count in summary["lane_distribution"].items():
                f.write(f"{lane:20s}: {count}\n")

            f.write("\n[Notes]\n")
            f.write(
                "This Part B implementation is a prototype for GPS-degraded visual localization. "
                "It demonstrates GPS integrity monitoring, state-machine handover, visual odometry, "
                "lane detection, landmark database, and EKF-inspired pose fusion.\n"
            )