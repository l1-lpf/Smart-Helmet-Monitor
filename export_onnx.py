from ultralytics import YOLO
import shutil
import os

if __name__ == "__main__":
    # 注意：你的截图里路径是 shwd_yolo11s，请核对一下你的实际文件夹名字
    model_path = "runs/detect/shwd_yolo11s/weights/best.pt"
    model = YOLO(model_path)

    print("正在导出 ONNX...")
    export_path = model.export(format="onnx", opset=12, simplify=True)

    os.makedirs("models", exist_ok=True)
    shutil.copy(export_path, "models/best.onnx")
    shutil.copy(model_path, "models/best.pt")
    print("✅ 模型已成功导出到 models/ 文件夹！")
