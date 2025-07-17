import os
import cv2
import numpy as np


def yolo_to_pixel_coords(img_w, img_h, xc, yc, w, h):
    """将 YOLO 格式 (x_center, y_center, width, height) 转换为像素坐标"""
    xc_px = int(xc * img_w)
    yc_px = int(yc * img_h)
    w_px = int(w * img_w)
    h_px = int(h * img_h)

    x1 = max(0, xc_px - w_px // 2)
    y1 = max(0, yc_px - h_px // 2)
    x2 = min(img_w, xc_px + w_px // 2)
    y2 = min(img_h, yc_px + h_px // 2)

    return x1, y1, x2, y2


def process_images(image_path, label_path, output_dir='mini_txt_v1'):
    """按YOLO标注裁剪图像"""
    print(f"开始裁剪图像: {image_path}，使用标注: {label_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 - {image_path}")
        return False

    # 检查标注文件是否存在
    if not os.path.exists(label_path):
        print(f"错误: 标注文件不存在 - {label_path}")
        return False

    try:
        # 读取YOLO格式的标注
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 打开图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 - {image_path}")
            return False

        img_h, img_w = image.shape[:2]

        # 获取图片基础名称（不带扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 处理每个标注框
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"警告: 第 {i} 行标注格式不正确 - {line}")
                continue

            # 解析YOLO格式的标注
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # 转换为像素坐标
            x1 = max(0, int((cx - w / 2) * img_w))
            y1 = max(0, int((cy - h / 2) * img_h))
            x2 = min(img_w, int((cx + w / 2) * img_w))
            y2 = min(img_h, int((cy + h / 2) * img_h))

            # 裁剪文字区域
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                print(f"警告: 裁剪区域为空 - {image_path} 中的第 {i} 个对象")
                continue

            # 保存裁剪结果
            output_path = os.path.join(output_dir, f"{base_name}_{i}.png")
            if cv2.imwrite(output_path, cropped):
                print(f"已裁剪并保存: {output_path}")
            else:
                print(f"错误: 无法保存裁剪结果 - {output_path}")

        return True

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return False


