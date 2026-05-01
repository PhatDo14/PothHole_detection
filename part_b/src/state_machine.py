class GPSStateMachine:
    def __init__(self, relock_stable_frames: int = 10):
        self.state = "GPS_GOOD"
        self.relock_stable_frames = relock_stable_frames
        self.good_counter = 0

    def update(self, gps_quality: str) -> str:
        previous_state = self.state

        if self.state == "GPS_GOOD":
            if gps_quality == "GPS_DEGRADED":
                self.state = "GPS_DEGRADED"
            elif gps_quality == "GPS_LOST":
                self.state = "GPS_LOST"

        elif self.state == "GPS_DEGRADED":
            if gps_quality == "GPS_GOOD":
                self.good_counter += 1
                if self.good_counter >= self.relock_stable_frames:
                    self.state = "GPS_GOOD"
                    self.good_counter = 0
            elif gps_quality == "GPS_LOST":
                self.state = "GPS_LOST"
                self.good_counter = 0
            else:
                self.good_counter = 0

        elif self.state == "GPS_LOST":
            if gps_quality == "GPS_GOOD":
                self.good_counter += 1
                if self.good_counter >= self.relock_stable_frames:
                    self.state = "GPS_GOOD"
                    self.good_counter = 0
            elif gps_quality == "GPS_DEGRADED":
                self.state = "GPS_DEGRADED"
                self.good_counter = 0
            else:
                self.good_counter = 0

        transition = previous_state != self.state

        return self.state, transition