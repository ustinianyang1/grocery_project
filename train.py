import os
import yaml
import logging
import time
from datetime import datetime

# 设置环境变量，确保ultralytics在当前目录工作
os.environ['YOLOv5_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_HOME'] = os.path.dirname(os.path.abspath(__file__))

from ultralytics import YOLO

# 配置日志
log_filename = 'training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [TRAIN] %(message)s',
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

def load_yolo_params():
    """加载yolo_params.yaml配置文件"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(BASE_DIR, 'yolo_params.yaml')
    
    if not os.path.exists(params_path):
        log_error(f"配置文件不存在: {params_path}")
        return None
    
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        log_info(f"配置文件加载成功: {params_path}")
        return params
    except Exception as e:
        log_error(f"加载配置文件失败: {e}")
        return None

def main():
    log_info("=====================================")
    log_info("开始执行本地训练脚本")
    log_info("=====================================")
    
    # 1. 加载配置
    params = load_yolo_params()
    if not params:
        return
    
    # 2. 配置本地路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    TRAIN_IMAGES = params['train']
    VAL_IMAGES = params['val']
    
    # 确保路径是绝对路径
    if not os.path.isabs(TRAIN_IMAGES):
        TRAIN_IMAGES = os.path.join(BASE_DIR, TRAIN_IMAGES)
    if not os.path.isabs(VAL_IMAGES):
        VAL_IMAGES = os.path.join(BASE_DIR, VAL_IMAGES)
    
    log_info(f"项目根目录: {BASE_DIR}")
    log_info(f"训练集路径: {TRAIN_IMAGES}")
    log_info(f"验证集路径: {VAL_IMAGES}")
    
    # 3. 验证路径存在
    if not os.path.exists(TRAIN_IMAGES):
        log_error(f"训练集路径不存在: {TRAIN_IMAGES}")
        return
    
    if not os.path.exists(VAL_IMAGES):
        log_error(f"验证集路径不存在: {VAL_IMAGES}")
        return
    
    # 4. 准备类别信息
    if 'nc' in params:
        nc = params['nc']
    else:
        nc = 3
        log_warning("配置文件中未指定类别数量，使用默认值: 3")
    
    if 'names' in params and params['names']:
        names = params['names']
    else:
        names = ['cheerios', 'soup', 'candle']
        log_warning("配置文件中未指定类别名称，使用默认值: ['cheerios', 'soup', 'candle']")
    
    log_info(f"类别数量: {nc}")
    log_info(f"类别名称: {names}")
    
    # 6. 启动训练
    try:
        log_info("正在加载 YOLOv11m 模型...")
        model = YOLO('yolo11m.pt')
        
        log_info("开始训练...")
        log_info("训练参数:")
        log_info("- epochs: 100")
        log_info("- imgsz: 800")
        log_info("- batch: 自动分配")
        log_info("- device: 0")
        log_info("- workers: 4")
        log_info("- project: grocery_local")
        log_info("- name: v11m_optimized")
        log_info("- patience: 20")
        
        # 检查GPU是否可用
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        log_info(f"使用设备: {device}")
        
        # 根据设备调整参数
        if device == '0':
            batch_size = 8
            workers = 4
            log_info("使用GPU训练，自动调整参数")
        else:
            batch_size = 4
            workers = 2
            log_warning("使用CPU训练，自动调整参数")
        
        # 记录训练开始信息
        log_info("开始训练，共 300 个epoch")
        
        # 构建绝对路径，确保结果保存到工作区
        project_path = os.path.join(BASE_DIR, 'grocery_local')
        log_info(f"训练结果保存目录: {project_path}")
        
        # 直接使用 yolo_params.yaml 作为数据配置
        model.train(
            data='yolo_params.yaml',
            epochs=300,
            imgsz=800,
            batch=batch_size,
            device=device,
            workers=workers,
            project=project_path,
            name='v11m_optimized',
            patience=50,
            exist_ok=True,
            weight_decay=0.001,
            dropout=0.1,
            copy_paste=0.4,
            mixup=0.2,
            cls=1.5,
            lr0=0.001,
            cos_lr=True,
            warmup_epochs=5.0
        )
        
        log_info("训练完成！")
        log_info(f"训练结果保存路径: {os.path.join(project_path, 'v11s_optimized')}")
        
    except Exception as e:
        log_error(f"训练失败: {e}")
        return

if __name__ == '__main__':
    main()
    log_info("=====================================")
    log_info("训练脚本执行完毕")
    log_info("=====================================")