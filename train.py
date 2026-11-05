from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 加载 YOLOv11 的 small 版本预训练模型（比 n 版本强，适合作为你的全栈项目基石）
    model = YOLO("yolo11s.pt")

    # 2. 开始训练
    results = model.train(
        data="dataset_shwd/data.yaml",  # 指向你刚才修改好相对路径的 yaml 文件
        epochs=50,  # 跑 50 轮，这个量级对安全帽数据集足够了
        imgsz=640,  # 图片缩放尺寸
        batch=8,  # 批次大小（如果你的 3050 显存报错，就把这里改成 4）
        amp=True,  # 开启自动混合精度，省显存
        device=0,  # 使用你的 RTX 3050 GPU
        name="shwd_yolo11s",  # 训练结果保存的文件夹名称
        workers=0,  # Windows 系统下建议设为 0，防止多进程读取数据报错
        patience=15,  # 早停机制：如果连续 15 轮 mAP 都不再提升，就自动提前结束，省时间
    )
