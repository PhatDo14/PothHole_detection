import cv2
import numpy as np


class LaneDetector:
    def __init__(self, roi_y_ratio: float = 0.55):
        self.roi_y_ratio = roi_y_ratio

    def _region_of_interest(self, edges):
        h, w = edges.shape[:2]

        mask = np.zeros_like(edges)

        polygon = np.array(
            [
                [
                    (0, h),
                    (w, h),
                    (int(0.65 * w), int(self.roi_y_ratio * h)),
                    (int(0.35 * w), int(self.roi_y_ratio * h)),
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _average_lane_line(self, lines, y_bottom, y_top):
        """
        Fit x = a*y + b from multiple candidate lines.
        Return one representative line: (x1, y1, x2, y2)
        """

        if len(lines) == 0:
            return None

        points_x = []
        points_y = []

        for x1, y1, x2, y2 in lines:
            points_x.extend([x1, x2])
            points_y.extend([y1, y2])

        if len(points_x) < 2:
            return None

        try:
            fit = np.polyfit(points_y, points_x, deg=1)
            a, b = fit

            x_bottom = int(a * y_bottom + b)
            x_top = int(a * y_top + b)

            return x_bottom, y_bottom, x_top, y_top
        except Exception:
            return None

    def _fallback_road_boundaries(self, h, w):
        """
        Fallback for unstructured roads without visible lane markings.

        This is only a visualization fallback, not a real lane detection result.
        It approximates road boundaries based on perspective geometry.
        """

        y_bottom = h
        y_top = int(self.roi_y_ratio * h)

        left_line = (
            int(0.25 * w),
            y_bottom,
            int(0.42 * w),
            y_top,
        )

        right_line = (
            int(0.75 * w),
            y_bottom,
            int(0.58 * w),
            y_top,
        )

        return left_line, right_line

    def detect(self, frame):
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Slight blur to reduce noise from potholes, puddles, and road texture
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blur, 80, 180)

        # Keep only the road region
        roi = self._region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=80,
            maxLineGap=60,
        )

        left_candidates = []
        right_candidates = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                mid_x = (x1 + x2) / 2

                # Filter noisy short lines
                if length < 80:
                    continue

                # Ignore near-horizontal edges from potholes, puddles, shadows
                if abs(slope) < 0.45:
                    continue

                # Left lane / left road boundary candidates
                if slope < 0 and mid_x < w * 0.70:
                    left_candidates.append((x1, y1, x2, y2))

                # Right lane / right road boundary candidates
                elif slope > 0 and mid_x > w * 0.30:
                    right_candidates.append((x1, y1, x2, y2))

        y_bottom = h
        y_top = int(self.roi_y_ratio * h)

        left_line = self._average_lane_line(left_candidates, y_bottom, y_top)
        right_line = self._average_lane_line(right_candidates, y_bottom, y_top)

        fallback_used = False

        # If no reliable lane markings are found, switch to unstructured road fallback.
        # This is common on rural / damaged roads without clear lane markings.
        if left_line is None and right_line is None:
            left_line, right_line = self._fallback_road_boundaries(h, w)
            lane_position = "unstructured_road"
            center_offset = 0.0
            fallback_used = True

            debug = {
                "left_line": left_line,
                "right_line": right_line,
                "center_offset": center_offset,
                "num_left_candidates": len(left_candidates),
                "num_right_candidates": len(right_candidates),
                "fallback_used": fallback_used,
            }

            return lane_position, center_offset, debug

        lane_position = "unknown"
        center_offset = 0.0

        # If both lane boundaries are available, estimate lane center.
        if left_line is not None and right_line is not None:
            left_bottom_x = left_line[0]
            right_bottom_x = right_line[0]

            lane_center = (left_bottom_x + right_bottom_x) / 2
            car_center = w / 2

            center_offset = (car_center - lane_center) / max(w, 1)

            if center_offset > 0.08:
                lane_position = "right"
            elif center_offset < -0.08:
                lane_position = "left"
            else:
                lane_position = "center"

        elif left_line is not None:
            lane_position = "right_or_center"

        elif right_line is not None:
            lane_position = "left_or_center"

        debug = {
            "left_line": left_line,
            "right_line": right_line,
            "center_offset": center_offset,
            "num_left_candidates": len(left_candidates),
            "num_right_candidates": len(right_candidates),
            "fallback_used": fallback_used,
        }

        return lane_position, center_offset, debug

    @staticmethod
    def draw(frame, debug):
        left_line = debug.get("left_line")
        right_line = debug.get("right_line")
        fallback_used = debug.get("fallback_used", False)

        if fallback_used:
            # Gray lines mean fallback visualization for unstructured road
            left_color = (180, 180, 180)
            right_color = (180, 180, 180)
            thickness = 2

            cv2.putText(
                frame,
                "Lane fallback: unstructured road",
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
            )
        else:
            # Blue = left candidate, yellow = right candidate
            left_color = (255, 0, 0)
            right_color = (0, 255, 255)
            thickness = 4

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(frame, (x1, y1), (x2, y2), left_color, thickness)

        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(frame, (x1, y1), (x2, y2), right_color, thickness)

        return frame