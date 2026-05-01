import os
import shutil
from huggingface_hub import hf_hub_download

os.makedirs("models", exist_ok=True)

print("Downloading pothole detector YOLOv8 model...")
pothole_pt = hf_hub_download(
    repo_id="peterhdd/pothole-detection-yolov8",
    filename="best.pt",
    local_dir="models",
    local_dir_use_symlinks=False,
)
print("Pothole model:", pothole_pt)

print("Downloading Depth Anything V2 Small ONNX...")
depth_onnx = hf_hub_download(
    repo_id="onnx-community/depth-anything-v2-small",
    filename="onnx/model.onnx",
    local_dir="models",
    local_dir_use_symlinks=False,
)
print("Depth ONNX original:", depth_onnx)

target_depth = os.path.join("models", "depth_anything_v2.onnx")
shutil.copy(depth_onnx, target_depth)
print("Depth ONNX saved to:", target_depth)

print("Done.")