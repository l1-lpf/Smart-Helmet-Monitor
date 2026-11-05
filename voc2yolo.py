import os
import xml.etree.ElementTree as ET
import random
import shutil

# ================= 配置路径 =================
# 注意：这里是你刚刚解压的 VOC2028 的路径
VOC_DIR = r"E:\CV-projects\Industrial-Defect-Detection-Service\VOC2028"
# 转换后自动生成的 YOLO 数据目录
YOLO_DIR = r"E:\CV-projects\Industrial-Defect-Detection-Service\dataset_shwd"

# SHWD 只有两个类别
classes = ["hat", "person"]

# 创建 YOLO 目录结构
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)


def convert_annotation(xml_file, txt_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(txt_path, "w") as out_file:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )

            bb_w = b[1] - b[0]
            bb_h = b[3] - b[2]
            x_center = (b[0] + bb_w / 2.0) / w
            y_center = (b[2] + bb_h / 2.0) / h
            bb_w /= w
            bb_h /= h

            out_file.write(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {bb_w:.6f} {bb_h:.6f}\n"
            )


annotations_dir = os.path.join(VOC_DIR, "Annotations")
jpeg_dir = os.path.join(VOC_DIR, "JPEGImages")

xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
random.shuffle(xml_files)

train_count = int(len(xml_files) * 0.8)
train_files = xml_files[:train_count]
val_files = xml_files[train_count:]

print(
    f"总数据量: {len(xml_files)}, 训练集: {len(train_files)}, 验证集: {len(val_files)}"
)


def process_data(files, split):
    for xml_file in files:
        img_name = xml_file.replace(".xml", ".jpg")
        img_path = os.path.join(jpeg_dir, img_name)

        if not os.path.exists(img_path):
            continue

        txt_name = xml_file.replace(".xml", ".txt")
        txt_path = os.path.join(YOLO_DIR, "labels", split, txt_name)
        convert_annotation(os.path.join(annotations_dir, xml_file), txt_path)
        shutil.copy(img_path, os.path.join(YOLO_DIR, "images", split, img_name))


process_data(train_files, "train")
process_data(val_files, "val")

yaml_content = f"""
path: {YOLO_DIR}
train: images/train
val: images/val

names:
  0: hat
  1: person
"""
with open(os.path.join(YOLO_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)
print("✅ 转换完成！data.yaml 已生成！")
