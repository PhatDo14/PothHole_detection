import numpy as np


def expand_box(bbox, image_shape, ratio=0.45):
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    ex = int(bw * ratio)
    ey = int(bh * ratio)

    nx1 = max(0, x1 - ex)
    ny1 = max(0, y1 - ey)
    nx2 = min(w - 1, x2 + ex)
    ny2 = min(h - 1, y2 + ey)

    return [nx1, ny1, nx2, ny2]


def estimate_pothole_depth(depth_map, bbox, depth_scale_cm=300.0):
    """
    Estimate pothole relative depth from monocular depth map.

    Depth Anything gives relative depth, not metric depth.
    This function compares the pothole ROI against the surrounding road region.

    Strategy:
    - Use surrounding road median as reference.
    - Use pothole ROI percentiles instead of median only.
    - Take the strongest robust difference because depth direction can vary.
    - Convert relative depth difference to pseudo-centimeter using calibration scale.
    """

    h, w = depth_map.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    pothole_roi = depth_map[y1:y2, x1:x2]

    ex1, ey1, ex2, ey2 = expand_box(bbox, image_shape=(h, w, 3), ratio=0.65)
    context = depth_map[ey1:ey2, ex1:ex2].copy()

    mask = np.ones_like(context, dtype=bool)

    bx1 = x1 - ex1
    by1 = y1 - ey1
    bx2 = x2 - ex1
    by2 = y2 - ey1

    mask[by1:by2, bx1:bx2] = False

    road_pixels = context[mask]
    pothole_pixels = pothole_roi.reshape(-1)

    if len(road_pixels) < 20 or len(pothole_pixels) < 20:
        return 0.0

    # Robust statistics
    road_ref = float(np.median(road_pixels))

    pothole_p10 = float(np.percentile(pothole_pixels, 10))
    pothole_p25 = float(np.percentile(pothole_pixels, 25))
    pothole_median = float(np.median(pothole_pixels))
    pothole_p75 = float(np.percentile(pothole_pixels, 75))
    pothole_p90 = float(np.percentile(pothole_pixels, 90))

    # Depth direction may be inverted depending on model/export.
    # Use robust maximum deviation from road reference.
    diffs = [
        abs(road_ref - pothole_p10),
        abs(road_ref - pothole_p25),
        abs(road_ref - pothole_median),
        abs(road_ref - pothole_p75),
        abs(road_ref - pothole_p90),
    ]

    relative_diff = max(diffs)

    depth_cm = relative_diff * depth_scale_cm

    return float(depth_cm)


def estimate_area_from_bbox(bbox, meter_per_pixel=0.003):
    x1, y1, x2, y2 = bbox

    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)

    area_pixel = bw * bh
    area_m2 = area_pixel * (meter_per_pixel ** 2)

    return float(area_m2)


def classify_severity(depth_cm, area_m2):
    """
    Rule-based severity classification.

    Since monocular depth is relative, severity is estimated from both:
    - calibrated relative depth
    - estimated surface area

    Thresholds are tuned for demo visualization.
    """

    # Minor: shallow and small potholes
    if depth_cm < 4.0 and area_m2 < 0.12:
        return "minor"

    # Moderate: medium depth or medium area
    if depth_cm < 8.0 and area_m2 < 0.35:
        return "moderate"

    # Severe: deep or large potholes
    return "severe"