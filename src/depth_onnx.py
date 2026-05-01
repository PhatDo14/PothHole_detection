import cv2
import numpy as np
import onnxruntime as ort


class DepthONNXEstimator:
    def __init__(self, model_path, input_size=518):
        self.model_path = model_path
        self.input_size = input_size

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC,
        )

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def estimate(self, image):
        h, w = image.shape[:2]

        inp = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: inp})

        depth = np.squeeze(outputs[0])

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        d_min = float(np.min(depth))
        d_max = float(np.max(depth))

        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)

        return depth_norm.astype(np.float32)