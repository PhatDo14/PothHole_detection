import csv
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GPSMeasurement:
    timestamp: float
    lat: Optional[float]
    lon: Optional[float]
    hdop: float
    num_satellites: int
    snr: float


class GPSIntegrityMonitor:
    def __init__(
        self,
        gps_log_path: str,
        good_hdop_threshold: float = 5.0,
        min_satellites_good: int = 4,
        min_snr_good: float = 20.0,
    ):
        self.gps_log_path = gps_log_path
        self.good_hdop_threshold = good_hdop_threshold
        self.min_satellites_good = min_satellites_good
        self.min_snr_good = min_snr_good

        self.measurements = self._load_gps_log(gps_log_path)

    def _load_gps_log(self, path: str) -> List[GPSMeasurement]:
        measurements = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                lat_raw = row.get("lat", "")
                lon_raw = row.get("lon", "")

                lat = float(lat_raw) if lat_raw not in ["", None] else None
                lon = float(lon_raw) if lon_raw not in ["", None] else None

                measurements.append(
                    GPSMeasurement(
                        timestamp=float(row["timestamp"]),
                        lat=lat,
                        lon=lon,
                        hdop=float(row["hdop"]),
                        num_satellites=int(float(row["num_satellites"])),
                        snr=float(row["snr"]),
                    )
                )

        return measurements

    def get_measurement_by_time(self, timestamp: float) -> GPSMeasurement:
        if not self.measurements:
            raise RuntimeError("GPS log is empty.")

        closest = min(
            self.measurements,
            key=lambda m: abs(m.timestamp - timestamp),
        )

        return closest

    def classify_quality(self, measurement: GPSMeasurement) -> str:
        if (
            measurement.lat is None
            or measurement.lon is None
            or measurement.num_satellites <= 0
        ):
            return "GPS_LOST"

        if (
            measurement.hdop <= self.good_hdop_threshold
            and measurement.num_satellites >= self.min_satellites_good
            and measurement.snr >= self.min_snr_good
        ):
            return "GPS_GOOD"

        return "GPS_DEGRADED"

    @staticmethod
    def gps_to_local_xy(
        measurement: GPSMeasurement,
        origin_lat: float,
        origin_lon: float,
    ):
        """
        Convert GPS lat/lon into local meter coordinate.

        This is an approximation suitable for short local trajectories.
        """

        if measurement.lat is None or measurement.lon is None:
            return None

        import math

        meter_per_deg_lat = 111_320.0
        meter_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))

        x = (measurement.lon - origin_lon) * meter_per_deg_lon
        y = (measurement.lat - origin_lat) * meter_per_deg_lat

        return x, y