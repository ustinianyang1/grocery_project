from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import logging
from datetime import datetime

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


# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt):
    # Perform prediction
    results = model.predict(image_path,conf=0.15)

    result = results[0]
    # Draw boxes on the image
    img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            conf = float(box.conf[0])  # confidence is a tensor with 1 value
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
    model_path = this_dir / "grocery_local" / "v11m_amp" / "weights" / "best.pt"
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
            predict_and_save(model, img_path, output_path_img, output_path_txt)
            log_info(f"处理图片: {img_path.name}")
        except Exception as e:
            log_error(f"处理图片 {img_path.name} 失败: {e}")

    log_info(f"预测完成，共处理 {image_count} 张图片")
    log_info(f"预测图片保存位置: {images_output_dir}")
    log_info(f"预测标签保存位置: {labels_output_dir}")
    log_info("=====================================")
    log_info("预测脚本执行完毕")
    log_info("=====================================")
