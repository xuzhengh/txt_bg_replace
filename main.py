import os
import pandas as pd
from yolo_checker.yolo_checker import YOLOv5Checker
from koutu import process_images
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
from dabao import replace_background

if __name__ == "__main__":
    image_path="txt_images/1.png"
    txt_checker = YOLOv5Checker(weights="./yolo_checker/weights/txt_yolov5x_rmbg.pt", device="0")
    txt_detect_result = txt_checker.predict(image_path)
    process_images()
    bg_image_path = "imgs/1.png"  # 替换为实际背景图片路径
    success = replace_background(bg_image_path)
