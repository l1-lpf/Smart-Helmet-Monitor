import cv2
from core.infer import HelmetDetector

# 1. 初始化检测器
detector = HelmetDetector("models/best.onnx")

# 2. 打开默认摄像头 (0代表电脑自带，1代表外接USB摄像头)
cap = cv2.VideoCapture(0)

print("摄像头已启动，按 'q' 键退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 实时推理
    result = detector.detect(frame)

    # 4. 如果触发报警，在画面左上角显示红字
    if result["alert"]:
        cv2.putText(
            frame,
            "WARNING: NO HARDHAT!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    # 5. 把框画到画面上
    for det in result["detections"]:
        x1, y1, x2, y2 = det["box"]
        class_name = det["class_name"]
        conf = det["confidence"]

        color = (0, 255, 0) if class_name == "hat" else (0, 0, 255)  # 绿框或红框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name} {conf}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # 显示画面
    cv2.imshow("Smart Helmet Monitor", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
