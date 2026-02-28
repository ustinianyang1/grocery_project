import os
import yaml
import logging
from datetime import datetime
from ultralytics import YOLO

# 配置日志
log_filename = 'validation.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [VALIDATE] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

# 简化日志函数
def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

def load_yolo_params():
    """加载yolo_params.yaml配置文件"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(BASE_DIR, 'yolo_params.yaml') #
    
    if not os.path.exists(params_path):
        log_error(f"配置文件不存在: {params_path}")
        return None
    
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
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
    
    # 2. 配置路径 (强制使用绝对路径以避免目录混淆)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 路径指向你之前训练生成的最佳模型权重
    model_path = os.path.join(BASE_DIR, 'grocery_local', 'v11m_amp', 'weights', 'best.pt')
    data_path = os.path.join(BASE_DIR, 'yolo_params.yaml')
    
    log_info(f"项目根目录: {BASE_DIR}")
    log_info(f"模型路径: {model_path}")
    
    # 3. 验证模型文件是否存在
    if not os.path.exists(model_path):
        log_error(f"模型文件不存在: {model_path}")
        log_error("请先确认训练是否成功完成并生成了权重文件")
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
        # 保持与训练一致的参数: imgsz=800 
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
        log_info(f"- 全类平均精确率 (MP): {results.box.mp:.4f}")
        log_info(f"- 全类平均召回率 (MR): {results.box.mr:.4f}")
        
        log_info(f"详细验证结果已保存到: {os.path.dirname(model_path)}")
        
    except Exception as e:
        log_error(f"验证过程中发生错误: {e}")
        return
    
    log_info("=====================================")
    log_info("验证脚本执行完毕")
    log_info("=====================================")

if __name__ == '__main__':
    main()