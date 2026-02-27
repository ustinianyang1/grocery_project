import os
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)


PSEUDO_LABEL_CONFIDENCE_THRESHOLD = 0.85
PSEUDO_LABEL_EXTEND_RATIO = 1.1


def generate_pseudo_labels(model_path, test_images_dir, output_train_dir, output_labels_dir, conf_threshold=0.85):
    log_info("="*50)
    log_info("开始生成伪标签")
    log_info("="*50)
    
    model = YOLO(model_path)
    
    test_images_dir = Path(test_images_dir)
    output_train_dir = Path(output_train_dir)
    output_labels_dir = Path(output_labels_dir)
    
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = [f for f in test_images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    log_info(f"找到 {len(image_files)} 张测试图片")
    
    total_boxes = 0
    high_conf_boxes = 0
    
    for img_path in image_files:
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        result = results[0]
        
        img_output_path = output_train_dir / img_path.name
        shutil.copy(img_path, img_output_path)
        
        label_path = output_labels_dir / (img_path.stem + '.txt')
        
        boxes_added = 0
        with open(label_path, 'w') as f:
            for box in result.boxes:
                cls_id = int(box.cls)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                
                if conf >= conf_threshold:
                    f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    boxes_added += 1
                    high_conf_boxes += 1
        
        total_boxes += len(result.boxes)
        
        if boxes_added > 0:
            log_info(f"  {img_path.name}: {boxes_added} 个高置信度框 (>{conf_threshold})")
    
    log_info("="*50)
    log_info(f"伪标签生成完成")
    log_info(f"  - 总检测框: {total_boxes}")
    log_info(f"  - 高置信度框 (>={conf_threshold}): {high_conf_boxes}")
    log_info(f"  - 输出图片目录: {output_train_dir}")
    log_info(f"  - 输出标签目录: {output_labels_dir}")
    log_info("="*50)
    
    return high_conf_boxes


def merge_with_original_dataset(pseudo_train_dir, pseudo_labels_dir, original_train_dir, original_labels_dir, output_dir):
    log_info("="*50)
    log_info("合并数据集")
    log_info("="*50)
    
    output_dir = Path(output_dir)
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    original_train_dir = Path(original_train_dir)
    original_labels_dir = Path(original_labels_dir)
    
    count = 0
    
    if original_train_dir.exists():
        for img_path in original_train_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img_path, output_images_dir / img_path.name)
                count += 1
                
                label_path = original_labels_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    shutil.copy(label_path, output_labels_dir / label_path.name)
    
    log_info(f"  复制原始训练图片: {count} 张")
    
    if pseudo_train_dir.exists() and pseudo_labels_dir.exists():
        pseudo_count = 0
        for img_path in pseudo_train_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                new_name = f"pseudo_{img_path.name}"
                shutil.copy(img_path, output_images_dir / new_name)
                pseudo_count += 1
                
                label_path = pseudo_labels_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    new_label_name = f"pseudo_{label_path.name}"
                    shutil.copy(label_path, output_labels_dir / new_label_name)
        
        log_info(f"  添加伪标签图片: {pseudo_count} 张")
    
    log_info(f"合并后总计: {count + pseudo_count} 张图片")
    log_info(f"  输出目录: {output_images_dir.parent}")
    log_info("="*50)


if __name__ == '__main__':
    this_dir = Path(__file__).parent
    
    with open(this_dir / 'yolo_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = this_dir / "grocery_local" / "v11m_optimized" / "weights" / "best.pt"
    
    if not model_path.exists():
        log_error(f"模型文件不存在: {model_path}")
        log_error("请先运行 train.py 训练模型")
        exit()
    
    test_images_dir = this_dir / "testImages" / "images"
    
    if not test_images_dir.exists():
        log_error(f"测试图片目录不存在: {test_images_dir}")
        exit()
    
    pseudo_train_dir = this_dir / "pseudo_dataset" / "images"
    pseudo_labels_dir = this_dir / "pseudo_dataset" / "labels"
    
    num_pseudo = generate_pseudo_labels(
        model_path=str(model_path),
        test_images_dir=test_images_dir,
        output_train_dir=pseudo_train_dir,
        output_labels_dir=pseudo_labels_dir,
        conf_threshold=PSEUDO_LABEL_CONFIDENCE_THRESHOLD
    )
    
    if num_pseudo > 0:
        log_info("伪标签已生成！你可以:")
        log_info(f"  1. 将 {pseudo_train_dir} 中的图片添加到训练集")
        log_info(f"  2. 将 {pseudo_labels_dir} 中的标签添加到对应的 labels 目录")
        log_info("  3. 重新训练模型以获得更好的性能")
    else:
        log_warning("没有生成足够的伪标签，请尝试降低置信度阈值")
