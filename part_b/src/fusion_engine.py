import math
from dataclasses import dataclass


@dataclass
class Pose:
    x: float
    y: float
    theta: float


class FusionEngine:
    """
    Lightweight EKF-inspired fusion.

    This is not a full EKF matrix implementation, but it follows the same idea:
    - prediction from VO
    - correction from GPS if available
    - GPS weight depends on GPS state
    - when GPS is lost, rely on VO
    - when GPS relocks, gradually correct drift
    """

    def __init__(
        self,
        gps_good_weight: float = 0.80,
        gps_degraded_weight: float = 0.30,
        gps_lost_weight: float = 0.0,
    ):
        self.pose = Pose(0.0, 0.0, 0.0)

        self.gps_good_weight = gps_good_weight
        self.gps_degraded_weight = gps_degraded_weight
        self.gps_lost_weight = gps_lost_weight

        self.last_gps_pose = None
        self.max_relock_correction = 3.0

    def update(
        self,
        vo_pose,
        gps_xy,
        gps_state: str,
    ):
        vo_x, vo_y, vo_theta = vo_pose

        predicted = Pose(vo_x, vo_y, vo_theta)

        if gps_xy is None:
            gps_weight = 0.0
        elif gps_state == "GPS_GOOD":
            gps_weight = self.gps_good_weight
        elif gps_state == "GPS_DEGRADED":
            gps_weight = self.gps_degraded_weight
        else:
            gps_weight = self.gps_lost_weight

        if gps_xy is not None and gps_weight > 0.0:
            gps_x, gps_y = gps_xy

            fused_x = gps_weight * gps_x + (1.0 - gps_weight) * predicted.x
            fused_y = gps_weight * gps_y + (1.0 - gps_weight) * predicted.y
            fused_theta = predicted.theta

            self.last_gps_pose = Pose(gps_x, gps_y, predicted.theta)
        else:
            fused_x = predicted.x
            fused_y = predicted.y
            fused_theta = predicted.theta

        self.pose = Pose(fused_x, fused_y, fused_theta)

        return self.pose

    @staticmethod
    def pose_distance(p1: Pose, p2: Pose) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)