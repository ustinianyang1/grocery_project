from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import logging
from datetime import datetime
import numpy as np

try:
    from ensemble_boxes import weighted_boxes_fusion
    ENSEMBLE_BOXES_AVAILABLE = True
except ImportError:
    ENSEMBLE_BOXES_AVAILABLE = False
    print("Warning: ensemble-boxes not installed, WBF will be disabled")

try:
    import torch
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("Warning: sahi not installed, SAHI will be disabled")

# 配置日志
log_filename = 'predict.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [PREDICT] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

# 简化日志函数
def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)

# ====== 配置选项 ======
USE_TTA = True  # 启用 TTA (Test Time Augmentation)
USE_WBF = True  # 启用 WBF (Weighted Boxes Fusion)
USE_SAHI = True  # 启用 SAHI 切片推理
SAHI_SLIDE_STRIDE = 256  # 切片滑动步长
SAHI_SLICE_SIZE = 512  # 切片大小
WBF_IOU_THRESH = 0.55  # WBF IoU 阈值
WBF_SCORE_THRESH = 0.15  # WBF 置信度阈值
# ======================

def apply_wbf(boxes_list, scores_list, labels_list, iou_thr=WBF_IOU_THRESH, skip_box_thr=WBF_SCORE_THRESH):
    """应用 WBF 融合多个预测框"""
    if not ENSEMBLE_BOXES_AVAILABLE:
        return boxes_list[0], scores_list[0], labels_list[0]
    
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, 
        scores_list, 
        labels_list,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )
    return boxes, scores, labels


# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt, 
                     use_tta=False, use_wbf=False, use_sahi=False, 
                     sahi_model=None, img_width=1920, img_height=1080):
    
    # 如果启用 SAHI，使用 SAHI 进行切片推理
    if use_sahi and SAHI_AVAILABLE and sahi_model is not None:
        try:
            log_info(f"  使用 SAHI 切片推理: {image_path.name}")
            result = get_sliced_prediction(
                str(image_path),
                sahi_model,
                slice_height=SAHI_SLICE_SIZE,
                slice_width=SAHI_SLICE_SIZE,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=WBF_IOU_THRESH,
                verbose=0
            )
            
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
            else:
                w, h = img_width, img_height
            
            boxes = []
            scores = []
            labels = []
            
            for pred in result.object_prediction_list:
                # SAHI 的 score 是 PredictionScore 对象，需要转换为 float
                score_value = float(pred.score.value) if hasattr(pred.score, 'value') else float(pred.score)
                if score_value >= WBF_SCORE_THRESH:
                    bbox = pred.bbox
                    x1, y1 = bbox.minx, bbox.miny
                    x2, y2 = bbox.maxx, bbox.maxy
                    boxes.append([x1/w, y1/h, x2/w, y2/h])
                    scores.append(score_value)
                    labels.append(pred.category.id)
            
            if boxes:
                boxes = np.array(boxes)
                scores = np.array(scores)
                labels = np.array(labels)
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{int(label)}:{score:.2f}", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                with open(output_path_txt, 'w') as f:
                    for box, score, label in zip(boxes, scores, labels):
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        f.write(f"{int(label)} {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                with open(output_path_txt, 'w') as f:
                    pass
                
            cv2.imwrite(str(output_path), img)
            return
        except Exception as e:
            log_warning(f"SAHI 推理失败，回退到普通推理: {e}")
    
    # 普通推理（可能带 TTA 和 WBF）
    results = model.predict(image_path, conf=0.15, augment=use_tta)

    result = results[0]
    
    # 获取原始图像尺寸用于坐标转换
    if use_wbf and len(results) > 0:
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for r in results:
            if len(r.boxes) > 0:
                # 获取归一化坐标 (xyxy 格式)
                boxes_xyxy = r.boxes.xyxyn.cpu().numpy()  # 归一化的 xyxy
                scores = r.boxes.conf.cpu().numpy()
                labels = r.boxes.cls.cpu().numpy().astype(int)
                
                boxes_list.append(boxes_xyxy)
                scores_list.append(scores)
                labels_list.append(labels)
        
        if boxes_list and len(boxes_list) > 0:
            boxes, scores, labels = apply_wbf(boxes_list, scores_list, labels_list)
            
            # 保存融合后的结果
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
            else:
                w, h = img_width, img_height
            
            # 绘制融合后的框
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{int(label)}:{score:.2f}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(output_path), img)
            
            # 保存标签
            with open(output_path_txt, 'w') as f:
                for box, score, label in zip(boxes, scores, labels):
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    f.write(f"{int(label)} {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            cv2.imwrite(str(output_path), result.plot())
            with open(output_path_txt, 'w') as f:
                pass
    else:
        # 不使用 WBF，直接保存结果
        img = result.plot()
        cv2.imwrite(str(output_path), img)
        
        with open(output_path_txt, 'w') as f:
            for box in result.boxes:
                cls_id = int(box.cls)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


if __name__ == '__main__': 

    log_info("=====================================")
    log_info("开始执行预测脚本")
    log_info("=====================================")

    # 获取工作目录路径
    this_dir = Path(__file__).parent
    log_info(f"工作目录: {this_dir}")
    
    os.chdir(this_dir)
    
    # 加载配置文件
    config_path = this_dir / 'yolo_params.yaml'
    log_info(f"加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
        log_info("配置文件加载成功")
        
        if 'test' in data and data['test'] is not None:
            # 直接使用配置文件中的测试路径
            images_dir = Path(data['test'])
            # 确保路径是绝对路径
            if not images_dir.is_absolute():
                images_dir = this_dir / images_dir
            log_info(f"测试集路径: {images_dir}")
        else:
            log_error("配置文件中未找到test字段，请添加test字段并指定测试图片路径")
            exit()
    except Exception as e:
        log_error(f"加载配置文件失败: {e}")
        exit()
    
    # 检查测试目录是否存在
    if not images_dir.exists():
        log_error(f"测试目录不存在: {images_dir}")
        exit()

    if not images_dir.is_dir():
        log_error(f"测试路径不是目录: {images_dir}")
        exit()
    
    if not any(images_dir.iterdir()):
        log_error(f"测试目录为空: {images_dir}")
        exit()

    # 加载YOLO模型
    # 只使用工作区中的模型路径
    model_path = this_dir / "grocery_local" / "v11m_optimized" / "weights" / "best.pt"
    log_info(f"模型路径: {model_path}")
    
    # 确保模型文件存在
    if not model_path.exists():
        log_error(f"模型文件不存在: {model_path}")
        log_error("请先运行train.py训练模型")
        exit()
    
    try:
        log_info("正在加载模型...")
        model = YOLO(model_path)
        log_info("模型加载成功")
        
        sahi_model = None
        if USE_SAHI and SAHI_AVAILABLE:
            try:
                log_info("正在加载 SAHI 模型...")
                sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='ultralytics',
                    model_path=str(model_path),
                    confidence_threshold=WBF_SCORE_THRESH,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    load_at_init=True,
                )
                # 确保模型已加载
                if hasattr(sahi_model, 'load_model'):
                    sahi_model.load_model()
                log_info("SAHI 模型加载成功")
            except Exception as e:
                log_warning(f"SAHI 模型加载失败: {e}")
                sahi_model = None
    except Exception as e:
        log_error(f"模型加载失败: {e}")
        exit()

    # 预测结果保存目录
    output_dir = this_dir / "predictions" # 预测结果保存到工作区
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"预测结果保存目录: {output_dir}")

    # 创建images和labels子目录
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"预测图片保存目录: {images_output_dir}")
    log_info(f"预测标签保存目录: {labels_output_dir}")

    # 遍历测试目录中的图片
    image_count = 0
    log_info("开始预测...")
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg']:
            continue
        image_count += 1
        output_path_img = images_output_dir / img_path.name  # 保存图片到images文件夹
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # 保存标签到labels文件夹
        
        try:
            predict_and_save(model, img_path, output_path_img, output_path_txt, 
                           use_tta=USE_TTA, use_wbf=USE_WBF, use_sahi=USE_SAHI, sahi_model=sahi_model)
            log_info(f"处理图片: {img_path.name}")
        except Exception as e:
            log_error(f"处理图片 {img_path.name} 失败: {e}")

    log_info(f"预测完成，共处理 {image_count} 张图片")
    log_info(f"预测图片保存位置: {images_output_dir}")
    log_info(f"预测标签保存位置: {labels_output_dir}")
    log_info("=====================================")
    log_info("预测脚本执行完毕")
    log_info("=====================================")
