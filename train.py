import os
import yaml
import logging
import torch
from ultralytics import YOLO

os.environ['YOLO_HOME'] = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log', encoding='utf-8')]
)
log = logging.info

def load_params():
    with open('yolo_params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    train_path = params['train']
    val_path = params['val']
    
    if not os.path.isabs(train_path):
        train_path = os.path.join(BASE_DIR, train_path)
    if not os.path.isabs(val_path):
        val_path = os.path.join(BASE_DIR, val_path)
    
    nc = params.get('nc', 3)
    names = params.get('names', ['cheerios', 'soup', 'candle'])
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    is_cuda = device == '0'
    
    log(f"模型: YOLOv11m | 设备: {device} | 类别: {nc}")
    
    model = YOLO('yolo11m.pt')
    
    project_path = os.path.join(BASE_DIR, 'grocery_local')
    
    model.train(
        data='yolo_params.yaml',
        epochs=300,
        imgsz=640,
        batch=8 if is_cuda else 4,
        device=device,
        workers=4 if is_cuda else 2,
        project=project_path,
        name='v11m_amp',
        exist_ok=True,
        weight_decay=0.001,
        dropout=0.1,
        copy_paste=0.3,
        mixup=0.15,
        cls=1.2,
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5.0,
        amp=True,
        cache=True,
        pretrained=True,
        optimizer='AdamW',
        seed=0,
        close_mosaic=10,
        overlap_mask=True,
        mask_ratio=4,
        val=True,
        plots=True,
        save=True,
    )
    
    log(f"训练完成! 结果保存至: {os.path.join(project_path, 'v11m_amp')}")

if __name__ == '__main__':
    main()
    log("训练脚本执行完毕")
