import cv2
import numpy as np
import math


class VisualOdometry:
    def __init__(
        self,
        max_features: int = 300,
        quality_level: float = 0.01,
        min_distance: int = 7,
        vo_scale: float = 0.02,
    ):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.vo_scale = vo_scale

        self.prev_gray = None
        self.prev_points = None

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.total_distance = 0.0

    def reset(self):
        self.prev_gray = None
        self.prev_points = None

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.total_distance = 0.0

    def _detect_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
        )

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return self.x, self.y, self.theta, 0.0, 0.0, 0

        if self.prev_points is None or len(self.prev_points) < 20:
            self.prev_points = self._detect_features(self.prev_gray)

        if self.prev_points is None:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return self.x, self.y, self.theta, 0.0, 0.0, 0

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if next_points is None or status is None:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return self.x, self.y, self.theta, 0.0, 0.0, 0

        good_old = self.prev_points[status.flatten() == 1]
        good_new = next_points[status.flatten() == 1]

        num_tracked = len(good_new)

        if num_tracked < 10:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return self.x, self.y, self.theta, 0.0, 0.0, num_tracked

        displacement = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
        dx_pixels = float(np.median(displacement[:, 0]))
        dy_pixels = float(np.median(displacement[:, 1]))

        # Approximate visual odometry in local coordinate.
        # In forward driving, vertical image motion is loosely mapped to forward movement.
        forward_m = -dy_pixels * self.vo_scale
        lateral_m = -dx_pixels * self.vo_scale

        # Heading change from optical flow horizontal shift
        dtheta = math.atan2(lateral_m, max(abs(forward_m), 1e-6)) * 0.05

        self.theta += dtheta

        dx_world = forward_m * math.cos(self.theta) - lateral_m * math.sin(self.theta)
        dy_world = forward_m * math.sin(self.theta) + lateral_m * math.cos(self.theta)

        self.x += dx_world
        self.y += dy_world

        self.total_distance += math.sqrt(dx_world**2 + dy_world**2)

        self.prev_gray = gray
        self.prev_points = self._detect_features(gray)

        return self.x, self.y, self.theta, dx_world, dy_world, num_tracked