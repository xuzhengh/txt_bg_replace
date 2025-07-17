import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import shutil
import threading
from datetime import datetime
from yolo_checker.yolo_checker import YOLOv5Checker
from koutu import process_images
from dabao import BackgroundReplacer
from PIL import Image, ImageTk
# 确保中文显示正常
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
import pathlib
import os
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



class TextBGReplacerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文字识别与背景替换工具")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # 初始化工具
        self.txt_detector = YOLOv5Checker(weights="yolo_checker/weights/txt_yolov5x_fixed.pt", device="0")
        self.bg_replacer = BackgroundReplacer()

        # 初始化变量
        self.current_image_path = None
        self.current_txt_path = None
        self.boxes = []  # 标注框列表 [x1, y1, x2, y2]
        self.selected_box = -1  # 当前选中的标注框索引
        self.dragging = False  # 拖动状态
        self.drag_offset = (0, 0)  # 拖动偏移量
        self.scale = 1.0  # 图像缩放比例
        self.img_width = 0
        self.img_height = 0

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建顶部按钮栏
        self.create_button_bar()

        # 创建图像显示区域
        self.create_image_display()

        # 创建控制面板
        self.create_control_panel()

        # 创建日志区域
        self.create_log_panel()

        # 初始化日志
        self.log("欢迎使用文字识别与背景替换工具")
        self.log("请上传一张图片开始处理")

    def create_button_bar(self):
        """创建顶部按钮栏"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.upload_btn = ttk.Button(button_frame, text="上传图片", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.detect_btn = ttk.Button(button_frame, text="检测文字", command=self.detect_text, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.edit_btn = ttk.Button(button_frame, text="编辑标注", command=self.edit_annotations, state=tk.DISABLED)
        self.edit_btn.pack(side=tk.LEFT, padx=5)

        self.crop_btn = ttk.Button(button_frame, text="裁剪文字", command=self.crop_text, state=tk.DISABLED)
        self.crop_btn.pack(side=tk.LEFT, padx=5)

        self.upload_bg_btn = ttk.Button(button_frame, text="上传背景", command=self.upload_background)
        self.upload_bg_btn.pack(side=tk.LEFT, padx=5)

        self.replace_btn = ttk.Button(button_frame, text="替换背景", command=self.replace_background, state=tk.DISABLED)
        self.replace_btn.pack(side=tk.LEFT, padx=5)

    def create_image_display(self):
        """创建图像显示区域"""
        image_frame = ttk.LabelFrame(self.main_frame, text="图像预览")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建Canvas用于显示图像和标注框
        self.canvas = tk.Canvas(image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        self.h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        # 绑定事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def create_control_panel(self):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(self.main_frame, text="标注框操作")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=5, ipady=5)

        ttk.Label(control_frame, text="标注框信息:").pack(anchor=tk.W, pady=5)

        # 坐标输入框
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=5)

        ttk.Label(info_frame, text="X1:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.x1_var = tk.StringVar()
        self.x1_entry = ttk.Entry(info_frame, textvariable=self.x1_var, width=10)
        self.x1_entry.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        self.x1_entry.bind("<Return>", self.update_box_coords)

        ttk.Label(info_frame, text="Y1:").grid(row=0, column=2, sticky=tk.W, padx=2, pady=2)
        self.y1_var = tk.StringVar()
        self.y1_entry = ttk.Entry(info_frame, textvariable=self.y1_var, width=10)
        self.y1_entry.grid(row=0, column=3, sticky=tk.W, padx=2, pady=2)
        self.y1_entry.bind("<Return>", self.update_box_coords)

        ttk.Label(info_frame, text="X2:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.x2_var = tk.StringVar()
        self.x2_entry = ttk.Entry(info_frame, textvariable=self.x2_var, width=10)
        self.x2_entry.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        self.x2_entry.bind("<Return>", self.update_box_coords)

        ttk.Label(info_frame, text="Y2:").grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        self.y2_var = tk.StringVar()
        self.y2_entry = ttk.Entry(info_frame, textvariable=self.y2_var, width=10)
        self.y2_entry.grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)
        self.y2_entry.bind("<Return>", self.update_box_coords)

        # 操作按钮
        ttk.Button(control_frame, text="添加标注框", command=self.add_box).pack(fill=tk.X, pady=5)
        self.delete_btn = ttk.Button(control_frame, text="删除标注框", command=self.delete_box, state=tk.DISABLED)
        self.delete_btn.pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="保存标注", command=self.save_annotations).pack(fill=tk.X, pady=5)

    def create_log_panel(self):
        """创建日志面板"""
        log_frame = ttk.LabelFrame(self.main_frame, text="操作日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

    def log(self, message):
        """记录操作日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def upload_image(self):
        """上传图片并保存到txt_images目录"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp")],
            title="选择图片"
        )

        if not file_path:
            return

        # 确保txt_images目录存在
        os.makedirs("txt_images", exist_ok=True)

        # 获取当前图片数量，命名为图片数+1.png
        image_files = [f for f in os.listdir("txt_images") if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        next_num = len(image_files) + 1
        file_ext = os.path.splitext(file_path)[1]
        new_file_name = f"{next_num}{file_ext}"
        new_file_path = os.path.join("txt_images", new_file_name)

        # 复制文件
        shutil.copy(file_path, new_file_path)
        self.current_image_path = new_file_path

        self.log(f"图片已上传并保存为: {new_file_name}")
        self.display_image()

        # 启用检测按钮
        self.detect_btn.config(state=tk.NORMAL)

    def display_image(self):
        """显示图像和标注框"""
        if not self.current_image_path:
            return

        try:
            # 打开图像
            img = Image.open(self.current_image_path)
            self.img_width, self.img_height = img.size

            # 调整图像大小以适应Canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 0 or canvas_height <= 0:
                return

            # 计算缩放比例
            self.scale = min(canvas_width / self.img_width, canvas_height / self.img_height)
            new_width = int(self.img_width * self.scale)
            new_height = int(self.img_height * self.scale)

            # 缩放图像
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # 转换为Tkinter可用的格式
            self.tk_img = ImageTk.PhotoImage(img)

            # 清除Canvas
            self.canvas.delete("all")

            # 显示图像
            self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

            # 显示标注框
            self.draw_boxes()

            # 更新滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        except Exception as e:
            self.log(f"显示图像时出错: {str(e)}")

    def draw_boxes(self):
        """在图像上绘制标注框"""
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box

            # 转换为Canvas坐标
            cx1, cy1 = x1 * self.scale, y1 * self.scale
            cx2, cy2 = x2 * self.scale, y2 * self.scale

            # 绘制标注框
            if i == self.selected_box:
                # 选中的框用红色
                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="red", width=2)
                self.draw_resize_handles(cx1, cy1, cx2, cy2)
            else:
                # 未选中的框用绿色
                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="green", width=2)

    def draw_resize_handles(self, x1, y1, x2, y2):
        """绘制调整大小的手柄"""
        handle_size = 8 / self.scale  # 转换为图像坐标
        self.canvas.create_rectangle(x1 - handle_size / 2, y1 - handle_size / 2, x1 + handle_size / 2,
                                     y1 + handle_size / 2, fill="red")
        self.canvas.create_rectangle(x2 - handle_size / 2, y1 - handle_size / 2, x2 + handle_size / 2,
                                     y1 + handle_size / 2, fill="red")
        self.canvas.create_rectangle(x1 - handle_size / 2, y2 - handle_size / 2, x1 + handle_size / 2,
                                     y2 + handle_size / 2, fill="red")
        self.canvas.create_rectangle(x2 - handle_size / 2, y2 - handle_size / 2, x2 + handle_size / 2,
                                     y2 + handle_size / 2, fill="red")

    def on_canvas_resize(self, event):
        """Canvas大小改变时重新显示图像"""
        self.display_image()

    def on_canvas_click(self, event):
        """处理Canvas点击事件"""
        if not self.current_image_path:
            return

        # 将Canvas坐标转换为图像坐标
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        handle_size = 10 / self.scale  # 手柄大小（图像坐标）
        self.dragging = False
        self.drag_offset = (0, 0)

        # 检查是否点击了调整手柄
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box

            # 检查四个角的手柄
            if self.is_point_in_handle(x, y, x1, y1, handle_size):
                self.selected_box = i
                self.dragging = True
                self.drag_offset = (x - x1, y - y1)
                self.drag_handle = "top_left"  # 新增：记录拖动的手柄位置
                self.update_box_info()
                return

            if self.is_point_in_handle(x, y, x2, y1, handle_size):
                self.selected_box = i
                self.dragging = True
                self.drag_offset = (x - x2, y - y1)
                self.drag_handle = "top_right"  # 新增：记录拖动的手柄位置
                self.update_box_info()
                return

            if self.is_point_in_handle(x, y, x1, y2, handle_size):
                self.selected_box = i
                self.dragging = True
                self.drag_offset = (x - x1, y - y2)
                self.drag_handle = "bottom_left"  # 新增：记录拖动的手柄位置
                self.update_box_info()
                return

            if self.is_point_in_handle(x, y, x2, y2, handle_size):
                self.selected_box = i
                self.dragging = True
                self.drag_offset = (x - x2, y - y2)
                self.drag_handle = "bottom_right"  # 新增：记录拖动的手柄位置
                self.update_box_info()
                return

            # 检查是否点击了标注框内部
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_box = i
                self.dragging = True
                self.drag_offset = (x - x1, y - y1)
                self.drag_handle = "move"  # 新增：记录拖动的是整个框
                self.update_box_info()
                return

        # 没有点击任何标注框，取消选择
        self.selected_box = -1
        self.update_box_info()
        self.display_image()

    def is_point_in_handle(self, x, y, bx, by, size):
        """检查点是否在手柄范围内"""
        return abs(x - bx) <= size and abs(y - by) <= size

    def on_canvas_drag(self, event):
        """处理Canvas拖动事件"""
        if not self.dragging or self.selected_box < 0:
            return

        # 将Canvas坐标转换为图像坐标
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        # 获取当前选中的标注框
        box = self.boxes[self.selected_box]
        x1, y1, x2, y2 = box

        if self.drag_handle == "move":
            # 移动标注框
            dx = x - (x1 + self.drag_offset[0])
            dy = y - (y1 + self.drag_offset[1])
            new_x1 = max(0, x1 + dx)
            new_y1 = max(0, y1 + dy)
            new_x2 = min(self.img_width, x2 + dx)
            new_y2 = min(self.img_height, y2 + dy)
            self.boxes[self.selected_box] = [new_x1, new_y1, new_x2, new_y2]

        else:
            # 调整标注框大小
            if self.drag_handle == "top_left":  # 左上角
                new_x1 = max(0, x - self.drag_offset[0])
                new_y1 = max(0, y - self.drag_offset[1])
                if new_x1 < x2 and new_y1 < y2:
                    self.boxes[self.selected_box] = [new_x1, new_y1, x2, y2]

            elif self.drag_handle == "top_right":  # 右上角
                new_x2 = min(self.img_width, x - self.drag_offset[0])
                new_y1 = max(0, y - self.drag_offset[1])
                if new_x2 > x1 and new_y1 < y2:
                    self.boxes[self.selected_box] = [x1, new_y1, new_x2, y2]

            elif self.drag_handle == "bottom_left":  # 左下角
                new_x1 = max(0, x - self.drag_offset[0])
                new_y2 = min(self.img_height, y - self.drag_offset[1])
                if new_x1 < x2 and new_y2 > y1:
                    self.boxes[self.selected_box] = [new_x1, y1, x2, new_y2]

            elif self.drag_handle == "bottom_right":  # 右下角
                new_x2 = min(self.img_width, x - self.drag_offset[0])
                new_y2 = min(self.img_height, y - self.drag_offset[1])
                if new_x2 > x1 and new_y2 > y1:
                    self.boxes[self.selected_box] = [x1, y1, new_x2, new_y2]

        # 更新显示
        self.update_box_info()
        self.display_image()

    def on_canvas_release(self, event):
        """处理Canvas释放事件"""
        self.dragging = False

    def update_box_info(self):
        """更新标注框信息显示"""
        if self.selected_box >= 0 and self.selected_box < len(self.boxes):
            x1, y1, x2, y2 = self.boxes[self.selected_box]

            self.x1_var.set(str(x1))
            self.y1_var.set(str(y1))
            self.x2_var.set(str(x2))
            self.y2_var.set(str(y2))

            # 启用删除按钮
            self.delete_btn.config(state=tk.NORMAL)
        else:
            # 清空输入框
            self.x1_var.set("")
            self.y1_var.set("")
            self.x2_var.set("")
            self.y2_var.set("")

            # 禁用删除按钮
            self.delete_btn.config(state=tk.DISABLED)

    def update_box_coords(self, event):
        """更新标注框坐标"""
        if self.selected_box < 0 or self.selected_box >= len(self.boxes):
            return

        try:
            x1 = int(self.x1_var.get())
            y1 = int(self.y1_var.get())
            x2 = int(self.x2_var.get())
            y2 = int(self.y2_var.get())

            # 确保坐标有效
            x1 = max(0, min(x1, self.img_width - 1))
            y1 = max(0, min(y1, self.img_height - 1))
            x2 = max(x1 + 1, min(x2, self.img_width))
            y2 = max(y1 + 1, min(y2, self.img_height))

            self.boxes[self.selected_box] = [x1, y1, x2, y2]
            self.display_image()

        except ValueError:
            # 输入无效，恢复原来的值
            x1, y1, x2, y2 = self.boxes[self.selected_box]
            self.x1_var.set(str(x1))
            self.y1_var.set(str(y1))
            self.x2_var.set(str(x2))
            self.y2_var.set(str(y2))

    def add_box(self):
        """添加新的标注框"""
        if not self.current_image_path:
            return

        # 创建默认大小的标注框
        default_size = min(self.img_width, self.img_height) // 5
        x1 = self.img_width // 2 - default_size // 2
        y1 = self.img_height // 2 - default_size // 2
        x2 = x1 + default_size
        y2 = y1 + default_size

        self.boxes.append([x1, y1, x2, y2])
        self.selected_box = len(self.boxes) - 1
        self.update_box_info()
        self.display_image()

    def delete_box(self):
        """删除选中的标注框"""
        if self.selected_box >= 0 and self.selected_box < len(self.boxes):
            del self.boxes[self.selected_box]
            self.selected_box = -1
            self.update_box_info()
            self.display_image()

    def save_annotations(self):
        """保存标注到txt文件"""
        if not self.current_image_path or not self.boxes:
            return

        # 确保txt_labels目录存在
        os.makedirs("txt_labels", exist_ok=True)

        # 获取对应的txt文件路径
        base_name = os.path.basename(self.current_image_path).split('.')[0]
        txt_path = os.path.join("txt_labels", f"{base_name}.txt")

        try:
            with open(txt_path, 'w') as f:
                for box in self.boxes:
                    x1, y1, x2, y2 = box

                    # 转换为YOLO格式 (center_x, center_y, width, height)
                    center_x = (x1 + x2) / 2 / self.img_width
                    center_y = (y1 + y2) / 2 / self.img_height
                    width = (x2 - x1) / self.img_width
                    height = (y2 - y1) / self.img_height

                    # 写入文件
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

            self.current_txt_path = txt_path
            self.log(f"标注已保存到: {os.path.basename(txt_path)}")
            self.crop_btn.config(state=tk.NORMAL)

        except Exception as e:
            self.log(f"保存标注时出错: {str(e)}")

    def detect_text(self):
        """执行文字检测"""
        if not self.current_image_path:
            return

        self.log(f"开始检测文字: {os.path.basename(self.current_image_path)}")

        # 在新线程中执行检测，避免界面卡顿
        def do_detection():
            try:
                # 调用用户的检测函数
                results = self.txt_detector.predict(self.current_image_path)

                # 解析结果并更新标注框
                self.boxes = []
                for result in results:
                    bbox = result["bbox"]
                    self.boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

                # 更新显示
                self.root.after(0, self.display_image)
                self.root.after(0, self.update_box_info)

                self.log(f"检测完成，找到 {len(results)} 个文字区域")

                # 保存标注
                self.root.after(0, self.save_annotations)

                # 启用编辑按钮
                self.root.after(0, lambda: self.edit_btn.config(state=tk.NORMAL))

            except Exception as e:
                self.log(f"检测过程中出错: {str(e)}")

        threading.Thread(target=do_detection, daemon=True).start()

    def edit_annotations(self):
        """进入编辑标注模式"""
        self.log("进入标注编辑模式，可以调整标注框或添加新标注")

    def crop_text(self):
        """执行裁剪操作"""
        if not self.current_image_path or not self.current_txt_path:
            self.log("错误: 没有选择图像或标注文件")
            return

        self.log("开始裁剪文字区域...")

        # 在新线程中执行裁剪
        def do_crop():
            try:
                # 调用裁剪函数，传递具体的图像路径和标注路径
                success = process_images(self.current_image_path, self.current_txt_path)

                if success:
                    self.log("文字区域裁剪完成")
                    self.upload_bg_btn.config(state=tk.NORMAL)
                else:
                    self.log("裁剪失败")

            except Exception as e:
                self.log(f"裁剪过程中出错: {str(e)}")

        threading.Thread(target=do_crop, daemon=True).start()

    def upload_background(self):
        """上传背景图片"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp")],
            title="选择背景图片"
        )

        if file_path:
            self.current_bg_path = file_path
            self.log(f"背景图片已选择: {os.path.basename(file_path)}")
            self.replace_btn.config(state=tk.NORMAL)

    def replace_background(self):
        """执行背景替换"""
        if not self.current_bg_path:
            return

        self.log(f"开始替换背景: {os.path.basename(self.current_bg_path)}")

        # 在新线程中执行背景替换
        def do_replace():
            try:
                # 调用背景替换函数，传入背景图片路径
                success = self.bg_replacer.replace_background(self.current_bg_path)

                if success:
                    self.log("背景替换完成")
                    messagebox.showinfo("成功", "背景替换完成，结果保存在final_output目录")
                else:
                    self.log("背景替换失败")

            except Exception as e:
                self.log(f"背景替换过程中出错: {str(e)}")

        threading.Thread(target=do_replace, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = TextBGReplacerGUI(root)
    root.mainloop()