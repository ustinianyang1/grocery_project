import os
import yaml
import logging
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
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
    
    # 使用配置文件中的路径，或使用默认路径
    if 'train' in params and params['train']:
        TRAIN_IMAGES = params['train']
    else:
        TRAIN_IMAGES = os.path.join(BASE_DIR, 'train/train/images')
    
    if 'val' in params and params['val']:
        VAL_IMAGES = params['val']
    else:
        VAL_IMAGES = os.path.join(BASE_DIR, 'val/val/images')
    
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
    
    # 5. 自动生成本地 data.yaml
    data_config = {
        'path': BASE_DIR,
        'train': TRAIN_IMAGES,
        'val': VAL_IMAGES,
        'nc': nc,
        'names': names
    }
    
    yaml_path = os.path.join(BASE_DIR, 'local_data.yaml')
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        log_info(f"配置文件已生成: {yaml_path}")
    except Exception as e:
        log_error(f"生成配置文件失败: {e}")
        return
    
    # 6. 启动训练
    try:
        log_info("正在加载 YOLOv11s 模型...")
        model = YOLO('yolo11s.pt')
        
        log_info("开始训练...")
        log_info("训练参数:")
        log_info("- epochs: 100")
        log_info("- imgsz: 800")
        log_info("- batch: 自动分配")
        log_info("- device: 0")
        log_info("- workers: 4")
        log_info("- project: grocery_local")
        log_info("- name: v11s_optimized")
        log_info("- patience: 20")
        
        # 检查GPU是否可用
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        log_info(f"使用设备: {device}")
        
        # 根据设备调整参数
        if device == 'cpu':
            batch_size = 4
            workers = 2
            log_warning("未检测到GPU，使用CPU训练，自动调整参数")
        else:
            batch_size = -1  # 自动分配batch size
            workers = 4
        
        model.train(
            data=yaml_path,
            epochs=100,
            imgsz=800,
            batch=batch_size,
            device=device,
            workers=workers,
            project='grocery_local',
            name='v11s_optimized',
            patience=20,
            exist_ok=True
        )
        
        log_info("训练完成！")
        log_info("训练结果保存路径: grocery_local/v11s_optimized/")
        
    except Exception as e:
        log_error(f"训练失败: {e}")
        return

if __name__ == '__main__':
    main()
    log_info("=====================================")
    log_info("训练脚本执行完毕")
    log_info("=====================================")