import csv
import math
import os
from pathlib import Path

from config_b import ConfigB


def generate_gps_log(
    output_path: str,
    duration_sec: float = 60.0,
    fps: float = 10.0,
):
    """
    Generate a synthetic GPS log.

    Scenario:
    0-15s   : GPS_GOOD
    15-25s  : GPS_DEGRADED
    25-40s  : GPS_LOST
    40-60s  : GPS_GOOD again

    This simulates a vehicle entering a GPS-degraded/lost area such as a tunnel,
    parking basement, or urban canyon.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_rows = int(duration_sec * fps)

    base_lat = 10.762622
    base_lon = 106.660172

    speed_mps = 8.0
    earth_meter_per_deg_lat = 111_320.0
    earth_meter_per_deg_lon = 111_320.0 * math.cos(math.radians(base_lat))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "lat",
                "lon",
                "hdop",
                "num_satellites",
                "snr",
            ],
        )
        writer.writeheader()

        for i in range(num_rows):
            t = i / fps
            distance_m = speed_mps * t

            lat = base_lat
            lon = base_lon + distance_m / earth_meter_per_deg_lon

            if t < 15:
                hdop = 1.2
                satellites = 10
                snr = 35
            elif t < 25:
                hdop = 6.5
                satellites = 3
                snr = 18
            elif t < 40:
                hdop = 99.0
                satellites = 0
                snr = 0
                lat = ""
                lon = ""
            else:
                hdop = 1.8
                satellites = 9
                snr = 32

            writer.writerow(
                {
                    "timestamp": f"{t:.2f}",
                    "lat": lat,
                    "lon": lon,
                    "hdop": hdop,
                    "num_satellites": satellites,
                    "snr": snr,
                }
            )

    print(f"GPS log saved to: {output_path}")


def main():
    cfg = ConfigB()
    generate_gps_log(cfg.gps_log_path)


if __name__ == "__main__":
    main()