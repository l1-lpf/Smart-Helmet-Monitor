import cv2
import numpy as np
import onnxruntime as ort
import os


class HelmetDetector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45):
        # 1. 加载 ONNX 模型
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = (640, 640)

        # 类别名称 (根据你的 data.yaml 来，顺序很重要)
        self.class_names = ["hat", "person"]

    def preprocess(self, img):
        # 图像预处理：Resize -> RGB -> 归一化 -> 转 NCHW
        h, w = img.shape[:2]
        # 使用 letterbox 保持比例缩放 (为了简化，这里直接用 cv2.resize，实际工程需更严谨)
        img_resized = cv2.resize(img, self.input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        # HWC -> CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))
        # 增加 Batch 维度: CHW -> NCHW
        img_batch = np.expand_dims(img_chw, axis=0)
        return img_batch, h, w

    def postprocess(self, output, orig_h, orig_w):
        # YOLOv11 的输出格式通常是 [1, 4 + num_classes, 8400]
        # 取第一个 batch，并转置为 [8400, 4 + num_classes]
        predictions = np.squeeze(output[0]).T

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            # 提取类别置信度 (第4位之后全是类别分数)
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > self.conf_thres:
                # YOLOv11 输出的是 cx, cy, w, h (中心点坐标和宽高)
                cx, cy, w, h = pred[:4]
                x1 = cx - w / 2
                y1 = cy - h / 2
                # 还原到原图尺寸
                x1 = int(x1 * orig_w / self.input_size[0])
                y1 = int(y1 * orig_h / self.input_size[1])
                w_box = int(w * orig_w / self.input_size[0])
                h_box = int(h * orig_h / self.input_size[1])

                boxes.append([x1, y1, w_box, h_box])
                scores.append(float(confidence))
                class_ids.append(class_id)

        # 非极大值抑制 (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append(
                    {
                        "class_id": int(class_ids[i]),
                        "class_name": self.class_names[class_ids[i]],
                        "confidence": round(scores[i], 2),
                        "box": [x, y, x + w, y + h],
                    }
                )

        return results

    def detect(self, img):
        input_tensor, orig_h, orig_w = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, orig_h, orig_w)

        # ==========================================
        # 核心业务逻辑：精准判断哪个头没戴安全帽
        # ==========================================
        is_alert = False

        # 筛选出所有的头部和所有的安全帽
        heads = [d for d in detections if d["class_name"] == "person"]
        hardhats = [d for d in detections if d["class_name"] == "hat"]

        for head in heads:
            hx1, hy1, hx2, hy2 = head["box"]
            # 算一下这个人头的中心点
            head_cx = (hx1 + hx2) / 2
            head_cy = (hy1 + hy2) / 2

            covered = False
            # 遍历所有安全帽，看看有没有帽子盖在这个人头的中心点上
            for hat in hardhats:
                wx1, wy1, wx2, wy2 = hat["box"]
                # 判断人头的中心点是否在安全帽的框内部
                if wx1 <= head_cx <= wx2 and wy1 <= head_cy <= wy2:
                    covered = True
                    break

            # 如果这个头没有被任何安全帽覆盖，立刻触发全局报警
            if not covered:
                is_alert = True
                break  # 只要有一个人没戴，马上报警

        return {"alert": is_alert, "detections": detections}


# 测试代码 (单独运行 infer.py 时生效)
if __name__ == "__main__":
    # 测试一下 (确保你有 models/best.onnx)
    detector = HelmetDetector("models/best.onnx")
    # 随便找一张测试图
    test_img = cv2.imread(
        "dataset/valid/images/004945_jpg.rf.eaaa004e086a50cbe93db6cdff5d93b5.jpg"
    )
    if test_img is not None:
        result = detector.detect(test_img)
        print("报警状态:", result["alert"])
        print("检测框数量:", len(result["detections"]))
    else:
        print("没找到测试图片，请修改路径。")
