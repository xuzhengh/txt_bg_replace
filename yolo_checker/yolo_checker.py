import torch
from pathlib import Path
from yolo_checker.models.common import DetectMultiBackend
from yolo_checker.utils.dataloaders import LoadImages
from yolo_checker.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh
)
from yolo_checker.utils.torch_utils import select_device
import pathlib
import os
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class YOLOv5Checker:
    def __init__(self, weights='weights/yolov5s.pt', device='', img_size=640, conf_thres=0.1, iou_thres=0.5):
        self.weights = weights
        self.device = select_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 加载模型
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt

        # 确保 imgsz 是 [h, w] 的形式
        imgsz = [self.img_size, self.img_size]  # 强制设置为二维
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # 创建 txt_labels 文件夹用于保存预测标签
        self.output_dir = 'txt_labels'
        os.makedirs(self.output_dir, exist_ok=True)

        # warmup
        self.model.warmup(imgsz=(1, 3, *self.imgsz))



    def predict(self, source):
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        results = []

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to float32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im)

            # NMS
            det = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

            # 准备保存文件
            base_name = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}.txt")

            with open(output_path, 'w', encoding='utf-8') as f:

                if len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape)

                    # Process detections
                    for *xyxy, conf, cls in reversed(det):
                        xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        class_id = int(cls)
                        h, w = im0s.shape[:2]
                        xc = (xyxy[0] + xyxy[2]) / 2 / w
                        yc = (xyxy[1] + xyxy[3]) / 2 / h
                        box_w = (xyxy[2] - xyxy[0]) / w
                        box_h = (xyxy[3] - xyxy[1]) / h

                        # 写入 YOLO 标注格式
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}\n")

                        # 添加到返回结果
                        results.append({
                            "class_id": class_id,
                            "class_name": self.names[class_id],
                            "confidence": float(conf),
                            "bbox": [float(x) for x in xyxy],
                            "bbox_xywh": [xc, yc, box_w, box_h]
                        })

        return results


if __name__ == "__main__":
    image_path = "../imgs/3.png"

    icon_detector = YOLOv5Checker(weights="weights/icon_yolo_background_best.pt", device="0")
    icon_detect_result = icon_detector.predict(image_path)
    print("icon检测结果：")
    has_abnormal = any(obj['class_id'] == 1 for obj in icon_detect_result)
    if has_abnormal:
        print("这张图片存在图标显示不全异常")
    else:
        print("这张图片没有图标异常")

    txt_detector = YOLOv5Checker(weights="weights/txt_yolov5x_rmbg.pt", device="0")
    txt_detect_result=txt_detector.predict(image_path)
    print("txt检测结果：")
    has_abnormal = any(obj['class_id'] == 1 for obj in txt_detect_result)
    if has_abnormal:
        print("这张图片存在文字显示不全异常")
    else:
        print("这张图片没有文字异常")