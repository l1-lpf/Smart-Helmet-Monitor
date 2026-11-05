import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont
import os


# ================= 核心处理逻辑 =================
def process_image(image, conf_thresh):
    if image is None:
        return (
            None,
            "<div style='color: #94a3b8; font-size: 18px; text-align: center; padding: 15px;'>系统待机中...</div>",
        )

    temp_path = os.path.join(os.getcwd(), "temp_upload.jpg")
    image.save(temp_path)

    try:
        with open(temp_path, "rb") as f:
            files = {"file": ("temp.jpg", f, "image/jpeg")}
            response = requests.post(
                "http://127.0.0.1:8000/api/detect",
                files=files,
                proxies={"http": None, "https": None},
            )
            data = response.json()
    except Exception as e:
        return (
            image,
            "<div style='color: #ef4444; font-size: 18px; text-align: center;'>❌ 后端连接失败</div>",
        )

    draw = ImageDraw.Draw(image)
    has_alert = False

    try:
        font = ImageFont.truetype("simhei.ttf", 24)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

    for det in data.get("detections", []):
        if det["confidence"] < conf_thresh:
            continue

        box = det["box"]
        cls_name = det["class_name"]
        conf = det["confidence"]

        if cls_name == "hat":
            color = "#00e5ff"
            text = f"hat {conf}"
        elif cls_name == "person":
            color = "#ff3366"
            text = f"person {conf}"
            has_alert = True

        draw.rectangle(box, outline=color, width=4)
        text_y = max(0, box[1] - 35)
        bbox = draw.textbbox((box[0], text_y), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text(
            (box[0], text_y),
            text,
            fill="black" if color == "#00e5ff" else "white",
            font=font,
        )

    if has_alert:
        status_html = """
        <div style="background: rgba(255, 51, 102, 0.2); backdrop-filter: blur(10px); border: 1px solid #ff3366; color: #ff3366; padding: 25px; border-radius: 12px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 20px rgba(255, 51, 102, 0.4); animation: pulse 2s infinite;">
            ⚠️ 严重警告：发现未佩戴安全帽人员！<br>请立即整改！
        </div>
        """
    else:
        status_html = """
        <div style="background: rgba(0, 229, 255, 0.1); backdrop-filter: blur(10px); border: 1px solid #00e5ff; color: #00e5ff; padding: 25px; border-radius: 12px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);">
            ✅ 现场安全：所有人员均已佩戴安全帽
        </div>
        """

    return image, status_html


# ================= 终极 CSS 融合 (完美滑动条版) =================
custom_css = """
@keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(255, 51, 102, 0.2); }
    50% { box-shadow: 0 0 30px rgba(255, 51, 102, 0.6); }
    100% { box-shadow: 0 0 10px rgba(255, 51, 102, 0.2); }
}

body, .gradio-container {
    background-color: #0b1120 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #f1f5f9 !important;
}

.block {
    background: rgba(30, 41, 59, 0.4) !important; 
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: none !important; 
    backdrop-filter: blur(12px) !important;
}

.block-label, label, .label-wrap, div[data-testid="block-label"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

span.svelte-15lo2v2, label > span, .svelte-1b6s6ym, span {
    font-size: 18px !important; /* 标题字体稍微缩回18px */
    font-weight: bold !important;
    color: #00e5ff !important; 
    letter-spacing: 1px !important;
}

.image-container, .image-frame, .image-preview {
    background-color: transparent !important;
    border: none !important;
}
img.svelte-1tgtx1p {
    object-fit: contain !important; 
}

/* 【核心修复】完美处理滑动条 */
div[data-testid="slider"] {
    background: #1e293b !important; /* 将灰色底板换成深色 */
    border-radius: 8px !important;
    padding: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* 滑动条两端的极小数字 0.1 和 0.95 */
.slider-container span, .gradio-container .wrap.svelte-1jz2g3n span {
    font-size: 14px !important; /* 缩回正常尺寸 */
    font-weight: bold !important;
    color: #94a3b8 !important;
}

/* 右侧的数字输入框 */
input[type="number"] {
    background-color: #0b1120 !important; 
    color: #00e5ff !important;
    border: 1px solid #00e5ff !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    height: 32px !important; /* 限制高度防止溢出 */
    padding: 0 5px !important;
    text-align: center !important;
}

/* 隐藏输入框旁边那个丑陋的上下小箭头 */
input[type="number"]::-webkit-inner-spin-button, 
input[type="number"]::-webkit-outer-spin-button { 
    -webkit-appearance: none; 
    margin: 0; 
}

button.primary {
    height: 60px !important; 
    font-size: 22px !important; 
    font-weight: 900 !important;
    letter-spacing: 4px !important; 
    background: linear-gradient(90deg, #00e5ff 0%, #0072ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4) !important;
    transition: all 0.3s !important;
    margin-top: 10px !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 114, 255, 0.7) !important;
}

h1 {
    background: linear-gradient(90deg, #00e5ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 2.8em !important;
    font-weight: 900 !important;
    margin-bottom: 10px !important;
}
footer { display: none !important; }
"""

theme = gr.themes.Base(
    primary_hue="blue", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")
).set(
    body_background_fill="#0b1120",
    block_background_fill="rgba(30, 41, 59, 0.4)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.05)",
    body_text_color="#f1f5f9",
    block_label_text_color="#e2e8f0",
)

with gr.Blocks(theme=theme, css=custom_css, title="智慧工地安全帽监测系统") as demo:
    gr.Markdown("<h1>👷 智慧工地安全帽智能监测系统</h1>")

    gr.Markdown(
        "<p style='text-align: center; color: #cbd5e1; font-size: 20px; font-weight: bold; letter-spacing: 2px; margin-bottom: 25px;'>基于 YOLOv11 + ONNX + FastAPI 的工业级视觉应用</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="📷 现场画面接入",
                sources=["upload", "webcam"],
                type="pil",
                height=550,
            )

            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.5,
                step=0.05,
                label="⚙️ 置信度阈值 (Confidence Threshold)",
            )

            btn = gr.Button("🚀 开始智能检测", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_img = gr.Image(
                label="🎯 检测结果输出", interactive=False, height=550
            )
            status_output = gr.HTML(
                "<div style='color: #94a3b8; font-size: 18px; text-align: center; padding: 20px;'>系统待机中...</div>"
            )

    btn.click(
        fn=process_image,
        inputs=[input_img, conf_slider],
        outputs=[output_img, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
