from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import sys
import os

# 把上一级目录加入路径，方便导入 core 里的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.infer import HelmetDetector

app = FastAPI(title="智慧工地安全帽监测系统 API")

# 允许跨域请求（非常重要，否则前端无法调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时加载模型（全局变量，避免每次请求都重新加载）
detector = HelmetDetector("models/best.onnx")


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    # 1. 读取前端传来的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "无法解析图片"}

    # 2. 调用 infer.py 中的检测逻辑
    result = detector.detect(img)

    # 3. 返回 JSON 结果
    return result


if __name__ == "__main__":
    import uvicorn

    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
