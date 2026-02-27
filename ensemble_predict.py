import os
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ENSEMBLE] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ensemble_predict.log', encoding='utf-8')
    ]
)

def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)


ENSEMBLE_CONFIDENCE = 0.15
ENSEMBLE_IOU_THRESH = 0.55


def load_models(model_paths):
    models = []
    for path in model_paths:
        if Path(path).exists():
            model = YOLO(path)
            models.append((path, model))
            log_info(f"加载模型: {path}")
        else:
            log_warning(f"模型不存在: {path}")
    return models


def predict_with_model(model, image_path, conf=0.15):
    results = model.predict(image_path, conf=conf, augment=True, verbose=False)
    result = results[0]
    
    boxes = []
    scores = []
    labels = []
    
    if len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxyn.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        boxes = boxes_xyxy.tolist()
    
    return boxes, scores, labels


def apply_wbf(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.15):
    if not boxes_list or len(boxes_list) == 0:
        return [], [], []
    
    valid_boxes = []
    valid_scores = []
    valid_labels = []
    
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        if len(boxes) > 0:
            valid_boxes.append(np.array(boxes))
            valid_scores.append(np.array(scores))
            valid_labels.append(np.array(labels))
    
    if not valid_boxes:
        return [], [], []
    
    boxes, scores, labels = weighted_boxes_fusion(
        valid_boxes,
        valid_scores,
        valid_labels,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )
    
    return boxes.tolist(), scores.tolist(), labels.tolist()


def ensemble_predict(models, image_path, output_path, output_label_path, conf=0.15, use_wbf=True):
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for model_name, model in models:
        boxes, scores, labels = predict_with_model(model, image_path, conf=conf)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    if not all_boxes:
        log_warning(f"没有模型检测到目标: {image_path.name}")
        img = cv2.imread(str(image_path))
        if img is not None:
            cv2.imwrite(str(output_path), img)
        with open(output_label_path, 'w') as f:
            pass
        return
    
    if use_wbf and len(all_boxes) > 1:
        final_boxes, final_scores, final_labels = apply_wbf(
            all_boxes, all_scores, all_labels,
            iou_thr=ENSEMBLE_IOU_THRESH,
            skip_box_thr=conf
        )
    else:
        final_boxes = all_boxes[0]
        final_scores = all_scores[0]
        final_labels = all_labels[0]
    
    img = cv2.imread(str(image_path))
    if img is not None:
        h, w = img.shape[:2]
    else:
        h, h = 1080, 1920
    
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{int(label)}:{score:.2f}", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), img)
    
    with open(output_label_path, 'w') as f:
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            f.write(f"{int(label)} {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    log_info("="*60)
    log_info("模型集成预测")
    log_info("="*60)
    
    this_dir = Path(__file__).parent
    
    with open(this_dir / 'yolo_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    test_images_dir = this_dir / config['test']
    if not test_images_dir.is_absolute():
        test_images_dir = this_dir / config['test']
    
    if not test_images_dir.exists():
        log_error(f"测试图片目录不存在: {test_images_dir}")
        return
    
    model_paths = [
        this_dir / "grocery_local" / "v11m_optimized" / "weights" / "best.pt",
        this_dir / "kfold_results" / "fold_0" / "weights" / "best.pt",
        this_dir / "kfold_results" / "fold_1" / "weights" / "best.pt",
    ]
    
    model_paths = [str(p) for p in model_paths if p.exists()]
    
    if not model_paths:
        log_error("没有找到可用的模型")
        return
    
    models = load_models(model_paths)
    
    if not models:
        log_error("无法加载任何模型")
        return
    
    output_dir = this_dir / "ensemble_predictions"
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    log_info(f"测试图片目录: {test_images_dir}")
    log_info(f"输出目录: {output_dir}")
    log_info(f"使用模型数量: {len(models)}")
    
    image_files = [f for f in test_images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    for img_path in image_files:
        output_path = images_output_dir / img_path.name
        output_label_path = labels_output_dir / (img_path.stem + '.txt')
        
        ensemble_predict(models, img_path, output_path, output_label_path, 
                        conf=ENSEMBLE_CONFIDENCE, use_wbf=True)
        
        log_info(f"处理完成: {img_path.name}")
    
    log_info("="*60)
    log_info(f"集成预测完成，共处理 {len(image_files)} 张图片")
    log_info(f"结果保存在: {output_dir}")
    log_info("="*60)


if __name__ == '__main__':
    main()
