import os
import shutil
from ultralytics import YOLO

model = YOLO("models/best.pt")

exported_path = model.export(
    format="onnx",
    imgsz=320,
    opset=12,
    simplify=False,
    dynamic=False,
)

print("Exported:", exported_path)

target = "models/pothole_detector.onnx"

if isinstance(exported_path, str) and os.path.exists(exported_path):
    shutil.copy(exported_path, target)
elif os.path.exists("models/best.onnx"):
    shutil.copy("models/best.onnx", target)
else:
    raise FileNotFoundError("Cannot find exported ONNX file.")

print("Saved to:", target)