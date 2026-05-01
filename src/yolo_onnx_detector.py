import cv2
import numpy as np
import onnxruntime as ort


class YOLOONNXDetector:
    def __init__(self, model_path, input_size=640, conf_thres=0.25, iou_thres=0.45):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def letterbox(self, image):
        h, w = image.shape[:2]

        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return padded, scale, left, top

    def preprocess(self, image):
        img, scale, pad_x, pad_y = self.letterbox(image)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, scale, pad_x, pad_y

    @staticmethod
    def xywh_to_xyxy(boxes):
        out = np.zeros_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= self.iou_thres)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, outputs, original_shape, scale, pad_x, pad_y):
        pred = outputs[0]

        if pred.ndim == 3:
            pred = pred[0]

        # YOLO ONNX thường là [84, 8400] hoặc [8400, 84]
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 100:
            pred = pred.T

        boxes_xywh = pred[:, :4]

        # Trường hợp YOLOv8 export thường là [x,y,w,h,class_score...]
        if pred.shape[1] == 5:
            class_ids = np.zeros(len(pred), dtype=np.int32)
            scores = pred[:, 4]
        else:
            class_probs = pred[:, 4:]
            class_ids = np.argmax(class_probs, axis=1)
            scores = class_probs[np.arange(len(class_probs)), class_ids]

        mask = scores >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        boxes = self.xywh_to_xyxy(boxes_xywh)

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= scale

        h, w = original_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        keep = self.nms(boxes, scores)

        detections = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float(scores[i]),
                    "class_id": int(class_ids[i]),
                }
            )

        return detections

    def detect(self, image):
        inp, scale, pad_x, pad_y = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: inp})

        detections = self.postprocess(
            outputs=outputs,
            original_shape=image.shape,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        return detections