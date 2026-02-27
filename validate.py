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
        logging.FileHandler('validation.log')
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
    log_info("开始执行验证脚本")
    log_info("=====================================")
    
    # 1. 加载配置
    params = load_yolo_params()
    if not params:
        return
    
    # 2. 配置路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'grocery_local', 'v11s_optimized', 'weights', 'best.pt')
    data_path = os.path.join(BASE_DIR, 'local_data.yaml')
    
    # 检查是否需要重新生成local_data.yaml
    if not os.path.exists(data_path):
        log_info("local_data.yaml不存在，正在根据yolo_params.yaml生成...")
        
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
        
        # 准备类别信息
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
        
        # 生成local_data.yaml
        data_config = {
            'path': BASE_DIR,
            'train': TRAIN_IMAGES,
            'val': VAL_IMAGES,
            'nc': nc,
            'names': names
        }
        
        try:
            with open(data_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            log_info(f"配置文件已生成: {data_path}")
        except Exception as e:
            log_error(f"生成配置文件失败: {e}")
            return
    
    log_info(f"项目根目录: {BASE_DIR}")
    log_info(f"模型路径: {model_path}")
    log_info(f"数据配置路径: {data_path}")
    
    # 3. 验证路径存在
    if not os.path.exists(model_path):
        log_error(f"模型文件不存在: {model_path}")
        log_error("请先运行训练脚本生成模型")
        return
    
    if not os.path.exists(data_path):
        log_error(f"数据配置文件不存在: {data_path}")
        log_error("请先运行训练脚本生成配置文件")
        return
    
    # 4. 加载模型
    try:
        log_info("正在加载模型...")
        model = YOLO(model_path)
        log_info("模型加载成功")
    except Exception as e:
        log_error(f"模型加载失败: {e}")
        return
    
    # 5. 执行验证
    try:
        log_info("开始验证...")
        log_info("验证参数:")
        log_info("- imgsz: 800")
        log_info("- device: 0")
        log_info("- workers: 4")
        
        results = model.val(
            data=data_path,
            imgsz=800,
            device=0,
            workers=4
        )
        
        # 6. 输出验证结果
        log_info("验证完成！")
        log_info("验证结果:")
        log_info(f"- 平均精度 (mAP@0.5): {results.box.map50:.4f}")
        log_info(f"- 平均精度 (mAP@0.5:0.95): {results.box.map:.4f}")
        log_info(f"- 精确率: {results.box.precision.mean():.4f}")
        log_info(f"- 召回率: {results.box.recall.mean():.4f}")
        
        log_info("详细验证结果已保存到: grocery_local/v11s_optimized/val_results.json")
        
    except Exception as e:
        log_error(f"验证失败: {e}")
        return
    
    log_info("=====================================")
    log_info("验证脚本执行完毕")
    log_info("=====================================")

if __name__ == '__main__':
    main()