from dataclasses import dataclass


@dataclass
class Config:
    yolo_onnx_path: str = "models/pothole_detector.onnx"
    depth_onnx_path: str = "models/depth_anything_v2.onnx"

    input_video_path: str = "input/test_video.mp4"
    output_video_path: str = "output/result_part_a.mp4"

    yolo_input_size: int = 320 #640
    conf_thres: float = 0.25
    iou_thres: float = 0.45

    class_names = ["pothole"]

    depth_input_size: int = 256

    # Scale giả định cho prototype.
    # Muốn depth thật theo cm thì phải calibration.
    depth_scale_cm: float = 300.0

    # Scale giả định cho area.
    # Muốn area thật theo m2 thì phải calibration camera/IPM.
    meter_per_pixel: float = 0.003

    # 1 = chạy mọi frame.
    # 2 hoặc 3 = nhanh hơn nhưng ít smooth hơn.
    frame_skip: int = 5