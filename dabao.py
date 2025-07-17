import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageSegmentation
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import shutil

# 确保中文显示正常
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


class BackgroundReplacer:
    def __init__(self):
        """初始化背景替换器，加载必要的模型"""
        # 配置固定路径
        self.CROPPED_DIR = "mini_txt_v1"
        self.BACKGROUND_DIR = "background"
        self.YOLO_DIR = "txt_labels"
        self.ORIGINAL_IMAGES_DIR = "txt_images"
        self.FINAL_OUTPUT_DIR = "final_output"

        # 确保目录存在
        for dir_path in [self.CROPPED_DIR, self.BACKGROUND_DIR,
                         self.YOLO_DIR, self.ORIGINAL_IMAGES_DIR, self.FINAL_OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)

        # 模型和参数配置
        self.MODEL_PATH = "rmbg"
        self.TEXT_THRESHOLD = 0.55
        self.SMOOTH_RADIUS = 1
        self.IMAGE_SIZE = (1280, 1280)
        self.USE_SR = True
        self.SR_MIN_SIZE = 200
        self.SR_MODEL_PATH = "realesrgan-x4plus.pth"
        self.USE_COLOR_ADAPTATION = True

        # 加载模型
        self.upsampler = self.init_sr_model()
        self.model, self.device, self.preprocess = self.load_bg_removal_model()

    def init_sr_model(self, model_path=None, device=None):
        """初始化超分辨率模型"""
        if model_path is None:
            model_path = self.SR_MODEL_PATH
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"正在加载超分辨率模型: {model_path}")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if device == "cuda" else False
        )
        print(f"超分辨率模型加载完成，运行在 {device} 设备上")
        return upsampler

    def load_bg_removal_model(self, model_path=None):
        """加载背景去除模型"""
        if model_path is None:
            model_path = self.MODEL_PATH

        print(f"正在加载背景去除模型: {model_path}")
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"背景去除模型加载完成，运行在 {device} 设备上")

        preprocess = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return model, device, preprocess

    def super_resolve(self, image, min_size=None):
        """对图像进行超分辨率处理"""
        if min_size is None:
            min_size = self.SR_MIN_SIZE

        if self.USE_SR:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            if min(width, height) >= min_size:
                return image

            img_np = np.array(rgb_image)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            output, _ = self.upsampler.enhance(img_np, outscale=max(min_size / min(width, height), 2.0))
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output)
        return image

    def analyze_and_convert(self, image):
        """分析图像并二值化为黑白图像"""
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        luminances = []
        color_counts = {}

        for y in range(height):
            for x in range(width):
                pixel = rgb_image.getpixel((x, y))
                r, g, b = pixel
                luminance = r * 0.299 + g * 0.587 + b * 0.114
                luminances.append(luminance)
                if pixel in color_counts:
                    color_counts[pixel] += 1
                else:
                    color_counts[pixel] = 1

        if not luminances:
            return image, None, None, False

        avg_luminance = sum(luminances) / len(luminances)
        dark_pixels = []
        light_pixels = []

        for y in range(height):
            for x in range(width):
                r, g, b = rgb_image.getpixel((x, y))
                luminance = r * 0.299 + g * 0.587 + b * 0.114
                if luminance < avg_luminance:
                    dark_pixels.append((r, g, b))
                else:
                    light_pixels.append((r, g, b))

        dark_color = max(dark_pixels, key=lambda p: color_counts.get(p, 0)) if dark_pixels else (0, 0, 0)
        light_color = max(light_pixels, key=lambda p: color_counts.get(p, 0)) if light_pixels else (255, 255, 255)

        converted = Image.new("RGB", (width, height))
        for y in range(height):
            for x in range(width):
                r, g, b = rgb_image.getpixel((x, y))
                luminance = r * 0.299 + g * 0.587 + b * 0.114
                if luminance < avg_luminance:
                    converted.putpixel((x, y), (0, 0, 0))
                else:
                    converted.putpixel((x, y), (255, 255, 255))

        return converted, dark_color, light_color, True

    def restore_color(self, image, dark_color, light_color):
        """将黑白图像恢复为原始颜色"""
        if dark_color is None or light_color is None:
            return image

        result = image.convert("RGBA")
        pixels = result.load()
        for x in range(result.width):
            for y in range(result.height):
                r, g, b, a = pixels[x, y]
                if (r, g, b) == (0, 0, 0):
                    pixels[x, y] = (dark_color[0], dark_color[1], dark_color[2], a)
                elif (r, g, b) == (255, 255, 255):
                    pixels[x, y] = (light_color[0], light_color[1], light_color[2], a)
        return result

    def remove_text_background(self, image):
        """去除图像背景"""
        converted_image, dark_color, light_color, is_converted = self.analyze_and_convert(image)
        rgb_image = converted_image.convert("RGB")

        input_tensor = self.preprocess(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(rgb_image.size)

        mask_np = np.array(mask)
        _, binary_mask = cv2.threshold(mask_np, int(self.TEXT_THRESHOLD * 255), 255, cv2.THRESH_BINARY)

        if self.SMOOTH_RADIUS > 0:
            smooth_mask = cv2.GaussianBlur(binary_mask, (self.SMOOTH_RADIUS * 2 + 1, self.SMOOTH_RADIUS * 2 + 1), 0)
        else:
            smooth_mask = binary_mask

        smooth_mask_pil = Image.fromarray(smooth_mask).convert("L")
        result = converted_image.copy()
        result.putalpha(smooth_mask_pil)

        if is_converted:
            final_result = self.restore_color(result, dark_color, light_color)
        else:
            final_result = result

        return final_result

    def parse_yolo_label(self, yolo_path, img_width, img_height):
        """解析YOLO标注文件"""
        boxes = []
        if not os.path.exists(yolo_path):
            return boxes

        with open(yolo_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts[:5])
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            boxes.append([x1, y1, x2, y2])
        return boxes

    def replace_background(self, bg_image_path):
        """
        主函数：替换图像背景

        参数:
            bg_image_path: 新背景图片的路径

        返回:
            bool: 处理是否成功
        """
        try:
            print("=" * 50)
            print("开始执行步骤1：准备处理")
            print("=" * 50)

            # 创建背景目录（如果不存在）
            os.makedirs(self.BACKGROUND_DIR, exist_ok=True)

            # 获取当前 background 文件夹中的所有 .png 文件
            existing_files = [f for f in os.listdir(self.BACKGROUND_DIR) if f.endswith('.png')]
            if existing_files:
                # 提取文件名中的数字编号
                indices = []
                for f in existing_files:
                    try:
                        idx = int(os.path.splitext(f)[0])
                        indices.append(idx)
                    except ValueError:
                        continue  # 忽略无法解析为数字的文件名
                next_index = max(indices) + 1 if indices else 1
            else:
                next_index = 1

            # 构建新文件名并复制图片
            new_bg_filename = f"{next_index}.png"
            new_bg_path = os.path.join(self.BACKGROUND_DIR, new_bg_filename)
            shutil.copy(bg_image_path, new_bg_path)

            print(f"背景图片已保存为: {new_bg_filename}")

            print("=" * 50)
            print("开始执行步骤2：处理文字图像并贴到背景")
            print("=" * 50)

            # 使用新保存的背景图进行处理
            result = self.process_and_paste_text_images(new_bg_filename)
            if not result:
                print("处理失败")
                return False

            print("\n\n")
            print("=" * 50)
            print("所有处理完成！")
            print(f"最终结果保存在: {self.FINAL_OUTPUT_DIR}")
            print("=" * 50)
            return True

        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            return False


    def process_and_paste_text_images(self, bg_filename):
        """处理文字图像并直接贴到背景"""
        try:
            # 获取基础文件名（不带扩展名）
            base_name = os.path.splitext(bg_filename)[0]

            # 构建原始图像路径（用于获取正确尺寸）
            original_img_path = os.path.join(self.ORIGINAL_IMAGES_DIR, f"{base_name}.png")
            if not os.path.exists(original_img_path):
                print(f"警告：未找到原始图像 {original_img_path}")
                return False

            # 打开原始图像以获取正确尺寸
            original_img = Image.open(original_img_path)
            original_width, original_height = original_img.size

            # 打开背景图像
            bg_path = os.path.join(self.BACKGROUND_DIR, bg_filename)
            background = Image.open(bg_path).convert("RGBA")

            # 调整背景图像尺寸以匹配原始图像
            if background.size != (original_width, original_height):
                print(f"调整 {bg_filename} 尺寸从 {background.size} 到 {original_width}x{original_height}")
                background = background.resize((original_width, original_height), Image.LANCZOS)

            # 构建YOLO标注文件路径
            yolo_path = os.path.join(self.YOLO_DIR, f"{base_name}.txt")
            boxes = self.parse_yolo_label(yolo_path, original_width, original_height)

            # 如果没有标注框，直接保存调整尺寸后的原图
            if not boxes:
                output_path = os.path.join(self.FINAL_OUTPUT_DIR, bg_filename)
                background.save(output_path)
                print(f"未找到标注框，保存调整尺寸后的原图")
                return True

            # 查找所有属于此背景图像的裁剪文字图像
            cropped_files = [f for f in os.listdir(self.CROPPED_DIR)
                             if f.startswith(f"{base_name}_") and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            cropped_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            # 确保裁剪图像数量与标注框数量一致
            if len(cropped_files) != len(boxes):
                print(f"警告：{bg_filename} 的标注框数量({len(boxes)})与裁剪图像数量({len(cropped_files)})不一致")
                min_count = min(len(boxes), len(cropped_files))
                cropped_files = cropped_files[:min_count]
                boxes = boxes[:min_count]

            print(f"找到 {len(cropped_files)} 个文字区域，开始处理...")

            # 逐个处理裁剪图像并贴到背景上
            for j, cropped_file in enumerate(cropped_files):
                try:
                    box = boxes[j]
                    x1, y1, x2, y2 = box

                    cropped_path = os.path.join(self.CROPPED_DIR, cropped_file)
                    cropped_img = Image.open(cropped_path).convert("RGBA")

                    # 执行超分+二值化+去背景处理
                    processed_img = self.super_resolve(cropped_img)
                    processed_img = self.remove_text_background(processed_img)

                    # 调整处理后的图像大小以匹配标注框
                    processed_img = processed_img.resize((x2 - x1, y2 - y1), Image.LANCZOS)

                    # 将处理后的图像粘贴到背景上
                    background.paste(processed_img, (x1, y1), processed_img)

                    if (j + 1) % 10 == 0 or (j + 1) == len(cropped_files):
                        print(f"已处理 {j + 1}/{len(cropped_files)} 个文字区域")

                except Exception as e:
                    print(f"处理文字区域 {cropped_file} 时出错: {str(e)}")

            # 保存结果图像
            output_path = os.path.join(self.FINAL_OUTPUT_DIR, bg_filename)
            background.save(output_path)
            print(f"已处理 {bg_filename}，贴回 {len(cropped_files)} 个文字区域")
            return True

        except Exception as e:
            print(f"处理文字图像时出错: {str(e)}")
            return False


def replace_background(bg_image_path):
    """
    替换图像背景的主函数

    参数:
        bg_image_path (str): 新背景图片的路径

    返回:
        bool: 处理是否成功
    """
    replacer = BackgroundReplacer()
    return replacer.replace_background(bg_image_path)


if __name__ == "__main__":
    # 示例调用
    bg_image_path = "path_to_your_background_image.jpg"  # 替换为实际背景图片路径
    success = replace_background(bg_image_path)
    if success:
        print("背景替换成功！结果保存在final_output目录")
    else:
        print("背景替换失败，请检查错误信息")