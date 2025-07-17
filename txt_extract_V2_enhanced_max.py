import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageSegmentation
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 确保中文显示正常
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# 初始化超分辨率模型
def init_sr_model(model_path="realesrgan-x4plus.pth", device="cuda"):
    """初始化Real-ESRGAN超分辨率模型"""
    print(f"正在加载超分辨率模型: {model_path}")

    # 模型配置
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False  # 半精度加速
    )

    print(f"超分辨率模型加载完成，运行在 {device} 设备上")
    return upsampler


# 图像超分辨率处理
def super_resolve(image, upsampler, min_size=200):
    """对图像进行超分辨率处理，确保最短边不小于min_size"""
    # 转换为RGB模式
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    # 如果图像已经足够大，直接返回
    if min(width, height) >= min_size:
        return image

    # 转换为numpy数组进行超分辨率处理
    img_np = np.array(rgb_image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 执行超分辨率（处理返回值）
    output, _ = upsampler.enhance(img_np, outscale=max(min_size / min(width, height), 2.0))
    # output, _ = upsampler.enhance(img_np, outscale=6.0)

    # 转换回PIL图像
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output)


def binarize_by_depth(image, depth_weight=(0.299, 0.587, 0.114)):
    """
    基于颜色深度平均值的二值化处理

    参数:
        image: PIL Image对象
        depth_weight: 颜色深度权重(R, G, B)，默认使用亮度公式
    """
    # 转换为RGB模式
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    # 计算每个像素的颜色深度
    depths = []
    for y in range(height):
        for x in range(width):
            r, g, b = rgb_image.getpixel((x, y))
            depth = r * depth_weight[0] + g * depth_weight[1] + b * depth_weight[2]
            depths.append(depth)

    if not depths:
        return image, None, None, False

    # 计算最深、最浅颜色和平均值
    min_depth = min(depths)
    max_depth = max(depths)
    avg_depth = (min_depth + max_depth) / 2  # 取最深和最浅的平均值

    # 创建二值化图像
    binary_image = Image.new("RGB", (width, height))
    dark_color = None
    light_color = None

    for y in range(height):
        for x in range(width):
            r, g, b = rgb_image.getpixel((x, y))
            depth = r * depth_weight[0] + g * depth_weight[1] + b * depth_weight[2]

            if depth > avg_depth:
                binary_image.putpixel((x, y), (0, 0, 0))  # 深色转黑色
                if dark_color is None:
                    dark_color = (r, g, b)
            else:
                binary_image.putpixel((x, y), (255, 255, 255))  # 浅色转白色
                if light_color is None:
                    light_color = (r, g, b)

    return binary_image, dark_color, light_color, True


def analyze_and_convert(image):
    """
    分析图像并基于亮度平均值二值化为纯黑白图像，记录出现次数最多的颜色

    参数:
        image: PIL Image对象，待处理图像

    返回:
        converted_image: 二值化后的图像
        dark_color: 原始图像中出现次数最多的深色（低亮度）
        light_color: 原始图像中出现次数最多的浅色（高亮度）
        is_converted: 是否已转换（始终为True）
    """
    # 转换为RGB模式
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    # 计算所有像素的亮度和颜色统计
    luminances = []
    color_counts = {}  # 颜色出现次数统计

    for y in range(height):
        for x in range(width):
            pixel = rgb_image.getpixel((x, y))
            r, g, b = pixel
            luminance = r * 0.299 + g * 0.587 + b * 0.114  # 亮度公式
            luminances.append(luminance)

            # 统计颜色出现次数
            if pixel in color_counts:
                color_counts[pixel] += 1
            else:
                color_counts[pixel] = 1

    if not luminances:
        return image, None, None, False

    # 计算平均亮度作为阈值
    avg_luminance = sum(luminances) / len(luminances)

    # 分离深色和浅色像素（修正逻辑）
    dark_pixels = []  # 低亮度像素 → 黑色
    light_pixels = []  # 高亮度像素 → 白色

    for y in range(height):
        for x in range(width):
            r, g, b = rgb_image.getpixel((x, y))
            luminance = r * 0.299 + g * 0.587 + b * 0.114

            if luminance < avg_luminance:  # 低于平均亮度 → 深色
                dark_pixels.append((r, g, b))
            else:  # 高于平均亮度 → 浅色
                light_pixels.append((r, g, b))

    # 找出出现次数最多的深色和浅色
    dark_color = max(dark_pixels, key=lambda p: color_counts.get(p, 0)) if dark_pixels else (0, 0, 0)
    light_color = max(light_pixels, key=lambda p: color_counts.get(p, 0)) if light_pixels else (255, 255, 255)

    # 创建二值化图像（修正逻辑）
    converted = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            r, g, b = rgb_image.getpixel((x, y))
            luminance = r * 0.299 + g * 0.587 + b * 0.114

            if luminance < avg_luminance:  # 低于平均 → 转换为黑色
                converted.putpixel((x, y), (0, 0, 0))
            else:  # 高于平均 → 转换为白色
                converted.putpixel((x, y), (255, 255, 255))

    return converted, dark_color, light_color, True


# 颜色恢复函数
def restore_color(image, dark_color, light_color):
    """将处理后的黑白图像恢复为原始颜色（基于出现次数最多的颜色）"""
    if dark_color is None or light_color is None:
        return image  # 未转换，直接返回

    # 转换为RGBA模式处理
    result = image.convert("RGBA")
    pixels = result.load()

    for x in range(result.width):
        for y in range(result.height):
            r, g, b, a = pixels[x, y]

            # 统一逻辑：黑色恢复为深色，白色恢复为浅色
            if (r, g, b) == (0, 0, 0):
                pixels[x, y] = (dark_color[0], dark_color[1], dark_color[2], a)  # 黑色 → 原始深色
            elif (r, g, b) == (255, 255, 255):
                pixels[x, y] = (light_color[0], light_color[1], light_color[2], a)  # 白色 → 原始浅色

    return result

# 模型加载函数
def load_model(model_path="rmbg"):
    """加载rmbg-2.0模型"""
    print(f"正在加载模型: {model_path}")

    # 关键：显式指定配置类名 + 信任自定义代码
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        config_name="BiRefNet_config.py"
    )

    model = AutoModelForImageSegmentation.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True
    )

    # 检查是否支持GPU，并将模型移至相应设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"模型加载完成，运行在 {device} 设备上")
    return model, device


# 图像预处理和后处理转换
def get_transforms(image_size=(1024, 1024)):
    """获取图像预处理和后处理的转换函数"""
    # 预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return preprocess


def remove_text_background_with_color_adaptation(image, model, device, preprocess,
                                                 text_threshold=0.6, smooth_radius=2,
                                                 color_output_path=None):
    """
    去除图像背景并集成颜色自适应处理（融合版）

    参数:
        image: PIL Image对象，待处理的图像
        model: 加载的分割模型
        device: 计算设备
        preprocess: 图像预处理转换
        text_threshold: 文字掩码阈值
        smooth_radius: 边缘平滑半径

    返回:
        PIL Image对象，处理后的透明背景图像
    """
    # 颜色分析与转换
    converted_image, dark_color, light_color, is_converted = analyze_and_convert(image)

    print(color_output_path)
    print(is_converted)
    if color_output_path and is_converted:
        print("wwwwwwwwwwwwwwwwwwwwwwwwww")
        converted_image.save(color_output_path)
    # 原始图像转换为RGB进行统一处理
    rgb_image = converted_image.convert("RGB")  # 使用颜色转换后的图像

    # 预处理
    input_tensor = preprocess(rgb_image).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    # 处理预测结果
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)

    # 调整掩码尺寸与原图一致
    mask = pred_pil.resize(rgb_image.size)

    # 文字增强处理
    mask_np = np.array(mask)
    _, binary_mask = cv2.threshold(mask_np, int(text_threshold * 255), 255, cv2.THRESH_BINARY)

    # 边缘平滑处理
    if smooth_radius > 0:
        smooth_mask = cv2.GaussianBlur(binary_mask, (smooth_radius * 2 + 1, smooth_radius * 2 + 1), 0)
    else:
        smooth_mask = binary_mask

    # 转换为PIL图像
    smooth_mask_pil = Image.fromarray(smooth_mask).convert("L")

    # 应用掩码到转换后的图像（关键修复点）
    result = converted_image.copy()  # 改为使用颜色转换后的图像
    result.putalpha(smooth_mask_pil)

    # 颜色恢复（如果进行了颜色转换）
    if is_converted:
        final_result = restore_color(result, dark_color, light_color)
    else:
        final_result = result

    return final_result


# 批量处理裁剪图像并去除背景（集成颜色处理）
def batch_remove_background(cropped_dir, output_dir, model_path="rmbg",
                            text_threshold=0.6, smooth_radius=2, image_size=(1024, 1024),
                            use_sr=True, sr_min_size=200, sr_model_path="realesrgan-x4plus.pth",
                            use_color_adaptation=True,
                            save_sr_output=True,  # 新增参数：是否保存超分辨率结果
                            save_color_output=True):  # 新增参数：是否保存颜色转换结果
    """
    批量处理裁剪目录中的所有图像，去除文字背景（集成颜色自适应处理）

    参数:
        ... [保持原有参数不变] ...
        save_sr_output: 是否保存超分辨率处理后的图像
        save_color_output: 是否保存颜色转换后的图像
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 新增：创建保存超分辨率结果的目录
    if save_sr_output:
        sr_output_dir = os.path.join(os.path.dirname(output_dir), "mini_txt_v3")
        os.makedirs(sr_output_dir, exist_ok=True)
    color_output_dir = os.path.join(os.path.dirname(output_dir), "mini_txt_v4")
    os.makedirs(color_output_dir, exist_ok=True)

    # 加载模型
    model, device = load_model(model_path)

    # 初始化超分辨率模型
    if use_sr:
        upsampler = init_sr_model(sr_model_path, device)
    else:
        upsampler = None

    # 获取图像预处理转换
    preprocess = get_transforms(image_size)

    # 获取所有裁剪图像文件
    image_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)

    print(f"找到 {total_images} 张裁剪图像，开始处理...")

    # 处理每个图像
    for i, image_file in enumerate(image_files):
        try:
            # 构建输入输出路径
            image_path = os.path.join(cropped_dir, image_file)
            output_path = os.path.join(output_dir, image_file)

            # 读取图像
            image = Image.open(image_path).convert("RGBA")

            # 应用超分辨率
            if use_sr:
                image = super_resolve(image, upsampler, sr_min_size)

                # 新增：保存超分辨率结果
                if save_sr_output:
                    sr_image_path = os.path.join(sr_output_dir, image_file)
                    image.save(sr_image_path)

            color_output_path = None
            if use_color_adaptation and save_color_output and color_output_dir:
                print("aaaaaaaaaaaaaaaaaaaaaaa")
                color_output_path = os.path.join(color_output_dir, image_file)

            # 去除背景（可选颜色自适应处理）
            if use_color_adaptation:
                processed_image = remove_text_background_with_color_adaptation(
                    image, model, device, preprocess, text_threshold, smooth_radius, color_output_path=color_output_path
                )
            else:
                processed_image = remove_text_background(
                    image, model, device, preprocess, text_threshold, smooth_radius
                )

            # 保存结果
            processed_image.save(output_path)

            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"已处理 {i + 1}/{total_images} 张图像")

        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}")

    print(f"所有裁剪图像背景去除完成！结果保存在 {output_dir}")
    return model, device, preprocess


# 解析YOLO标注文件
def parse_yolo_label(yolo_path, img_width, img_height):
    """
    解析YOLO格式的标注文件

    参数:
        yolo_path: YOLO标注文件路径
        img_width: 图像宽度
        img_height: 图像高度

    返回:
        标注框列表，每个标注框格式为 [x1, y1, x2, y2]
    """
    boxes = []

    if not os.path.exists(yolo_path):
        print(f"警告：未找到YOLO标注文件 {yolo_path}")
        return boxes

    with open(yolo_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # 解析YOLO坐标（归一化值）
        class_id, x_center, y_center, width, height = map(float, parts[:5])

        # 转换为像素坐标
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # 确保坐标有效
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        boxes.append([x1, y1, x2, y2])

    return boxes


# 将处理后的透明图像贴回到壁纸
def paste_processed_images(background_dir, processed_dir, yolo_dir, output_dir, original_images_dir):
    """
    将处理后的透明图像按照YOLO标注信息贴回到背景图像，并确保壁纸尺寸与原图一致

    参数:
        background_dir: 背景图像(壁纸)目录
        processed_dir: 处理后的透明图像目录
        yolo_dir: YOLO标注文件目录
        output_dir: 输出结果目录
        original_images_dir: 原始图像目录（用于获取正确尺寸）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有背景图像文件
    background_files = [f for f in os.listdir(background_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(background_files)

    print(f"找到 {total_images} 张背景图像，开始处理...")

    # 处理每个背景图像
    for i, bg_file in enumerate(background_files):
        try:
            # 获取基础文件名（不带扩展名）
            base_name = os.path.basename(bg_file).split('.')[0]

            # 构建背景图像路径
            bg_path = os.path.join(background_dir, bg_file)

            # 构建原始图像路径（用于获取正确尺寸）
            original_img_path = os.path.join(original_images_dir, f"{base_name}.png")

            # 检查原始图像是否存在
            if not os.path.exists(original_img_path):
                print(f"警告：未找到原始图像 {original_img_path}，跳过此图像")
                continue

            # 打开原始图像以获取正确尺寸
            original_img = Image.open(original_img_path)
            original_width, original_height = original_img.size

            # 打开背景图像
            background = Image.open(bg_path).convert("RGBA")

            # 调整背景图像尺寸以匹配原始图像
            if background.size != (original_width, original_height):
                print(f"调整 {bg_file} 尺寸从 {background.size} 到 {original_width}x{original_height}")
                background = background.resize((original_width, original_height), Image.LANCZOS)

            # 构建YOLO标注文件路径
            yolo_path = os.path.join(yolo_dir, f"{base_name}.txt")

            # 解析YOLO标注文件
            boxes = parse_yolo_label(yolo_path, original_width, original_height)

            # 如果没有标注框，直接保存调整尺寸后的原图
            if not boxes:
                output_path = os.path.join(output_dir, bg_file)
                background.save(output_path)
                print(f"[{i + 1}/{total_images}] {bg_file}: 未找到标注框，保存调整尺寸后的原图")
                continue

            # 查找所有属于此背景图像的处理后的透明图像
            processed_files = [f for f in os.listdir(processed_dir)
                               if f.startswith(f"{base_name}_") and f.lower().endswith(('.png'))]

            # 按序号排序（例如1_1.png, 1_2.png, ...）
            processed_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            # 确保处理后的图像数量与标注框数量一致
            if len(processed_files) != len(boxes):
                print(f"警告：{bg_file} 的标注框数量({len(boxes)})与处理后的图像数量({len(processed_files)})不一致")
                min_count = min(len(boxes), len(processed_files))
                processed_files = processed_files[:min_count]
                boxes = boxes[:min_count]

            # 逐个将处理后的透明图像贴到背景上
            for j, processed_file in enumerate(processed_files):
                # 获取当前标注框
                box = boxes[j]
                x1, y1, x2, y2 = box

                # 打开处理后的透明图像
                processed_path = os.path.join(processed_dir, processed_file)
                processed_img = Image.open(processed_path).convert("RGBA")

                # 调整透明图像大小以匹配标注框
                processed_img = processed_img.resize((x2 - x1, y2 - y1), Image.LANCZOS)

                # 将透明图像粘贴到背景上
                background.paste(processed_img, (x1, y1), processed_img)

            # 保存结果图像
            output_path = os.path.join(output_dir, bg_file)
            background.save(output_path)

            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"[{i + 1}/{total_images}] 已处理 {bg_file}，贴回 {len(processed_files)} 个透明图像")

        except Exception as e:
            print(f"处理背景图像 {bg_file} 时出错: {str(e)}")

    print(f"所有背景图像处理完成！结果保存在 {output_dir}")


# 主函数：整合步骤2和步骤3
def main():
    # 配置路径
    CROPPED_DIR = "mini_txt_v1"  # 步骤1裁剪出的图像目录,v1原版，v2透明，v3超分后，v4二值化后黑白图结果
    PROCESSED_DIR = "mini_txt_v2"  # 步骤2处理后的透明图像目录
    BACKGROUND_DIR = "background"  # 背景图像(壁纸)目录
    YOLO_DIR = "txt_labels"  # YOLO标注文件目录
    ORIGINAL_IMAGES_DIR = "txt_images"  # 原始图像目录（用于获取正确尺寸）
    FINAL_OUTPUT_DIR = "final_output"  # 最终结果目录

    # 可选参数配置
    MODEL_PATH = "rmbg"  # 模型路径
    TEXT_THRESHOLD = 0.55  # 文字掩码阈值（降低以减少文字丢失）
    SMOOTH_RADIUS = 1  # 边缘平滑半径（减小以保留文字细节）
    IMAGE_SIZE = (1280, 1280)  # 增大模型输入尺寸

    # 超分辨率配置
    USE_SR = True  # 是否使用超分辨率
    SR_MIN_SIZE = 200  # 超分辨率后图像最短边的最小尺寸
    SR_MODEL_PATH = "realesrgan-x4plus.pth"  # Real-ESRGAN模型路径

    # 颜色自适应处理配置
    USE_COLOR_ADAPTATION = True  # 是否启用颜色自适应处理

    # 新增：保存中间结果配置
    SAVE_SR_OUTPUT = True  # 保存超分辨率结果
    SAVE_COLOR_OUTPUT = True  # 保存颜色转换结果

    print("=" * 50)
    print("开始执行步骤2：去除裁剪图像的背景")
    print("=" * 50)

    # 步骤2: 批量去除裁剪图像的背景（集成颜色自适应处理）
    batch_remove_background(
        cropped_dir=CROPPED_DIR,
        output_dir=PROCESSED_DIR,
        model_path=MODEL_PATH,
        text_threshold=TEXT_THRESHOLD,
        smooth_radius=SMOOTH_RADIUS,
        image_size=IMAGE_SIZE,
        use_sr=USE_SR,
        sr_min_size=SR_MIN_SIZE,
        sr_model_path=SR_MODEL_PATH,
        use_color_adaptation=USE_COLOR_ADAPTATION,
        save_sr_output=SAVE_SR_OUTPUT,
        save_color_output=SAVE_COLOR_OUTPUT
    )

    print("\n\n")
    print("=" * 50)
    print("开始执行步骤3：将处理后的透明图像贴回到壁纸")
    print("=" * 50)

    # 步骤3: 将处理后的透明图像贴回到背景图像，并确保尺寸与原图一致
    paste_processed_images(
        background_dir=BACKGROUND_DIR,
        processed_dir=PROCESSED_DIR,
        yolo_dir=YOLO_DIR,
        output_dir=FINAL_OUTPUT_DIR,
        original_images_dir=ORIGINAL_IMAGES_DIR
    )

    print("\n\n")
    print("=" * 50)
    print("所有处理完成！")
    print(f"处理后的透明图像保存在: {PROCESSED_DIR}")
    print(f"最终结果保存在: {FINAL_OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()