import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class Landmark:
    landmark_id: int
    class_name: str
    x: float
    y: float
    descriptor: str
    first_seen: float
    last_seen: float
    num_observations: int


class LandmarkDatabase:
    def __init__(self, match_distance_threshold: float = 5.0):
        self.landmarks: Dict[int, Landmark] = {}
        self.next_id = 1
        self.match_distance_threshold = match_distance_threshold

    def add_or_update(
        self,
        class_name: str,
        x: float,
        y: float,
        timestamp: float,
        descriptor: str = "visual_descriptor_placeholder",
    ):
        matched_id = self.find_nearest(class_name, x, y)

        if matched_id is not None:
            lm = self.landmarks[matched_id]
            lm.x = 0.8 * lm.x + 0.2 * x
            lm.y = 0.8 * lm.y + 0.2 * y
            lm.last_seen = timestamp
            lm.num_observations += 1
            return lm, True

        landmark = Landmark(
            landmark_id=self.next_id,
            class_name=class_name,
            x=x,
            y=y,
            descriptor=descriptor,
            first_seen=timestamp,
            last_seen=timestamp,
            num_observations=1,
        )

        self.landmarks[self.next_id] = landmark
        self.next_id += 1

        return landmark, False

    def find_nearest(self, class_name: str, x: float, y: float) -> Optional[int]:
        best_id = None
        best_dist = float("inf")

        for lm_id, lm in self.landmarks.items():
            if lm.class_name != class_name:
                continue

            dist = math.sqrt((lm.x - x) ** 2 + (lm.y - y) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_id = lm_id

        if best_dist <= self.match_distance_threshold:
            return best_id

        return None

    def get_reidentified_count(self):
        return sum(1 for lm in self.landmarks.values() if lm.num_observations > 1)

    def to_list(self) -> List[dict]:
        return [asdict(lm) for lm in self.landmarks.values()]