import os
import yaml
import logging
import shutil
import json
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [KFOLD] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kfold_training.log', encoding='utf-8')
    ]
)

def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)


N_FOLDS = 5
RANDOM_SEED = 42


def create_kfold_splits(images_dir, labels_dir, output_base_dir, n_splits=5, seed=42):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_base_dir = Path(output_base_dir)
    
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(image_files) == 0:
        log_error(f"No images found in {images_dir}")
        return []
    
    log_info(f"Found {len(image_files)} images")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        fold_dir = output_base_dir / f"fold_{fold_idx}"
        fold_train_images = fold_dir / 'train' / 'images'
        fold_train_labels = fold_dir / 'train' / 'labels'
        fold_val_images = fold_dir / 'val' / 'images'
        fold_val_labels = fold_dir / 'val' / 'labels'
        
        fold_train_images.mkdir(parents=True, exist_ok=True)
        fold_train_labels.mkdir(parents=True, exist_ok=True)
        fold_val_images.mkdir(parents=True, exist_ok=True)
        fold_val_labels.mkdir(parents=True, exist_ok=True)
        
        train_count = 0
        for idx in train_idx:
            img_path = image_files[idx]
            shutil.copy(img_path, fold_train_images / img_path.name)
            
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy(label_path, fold_train_labels / label_path.name)
            train_count += 1
        
        val_count = 0
        for idx in val_idx:
            img_path = image_files[idx]
            shutil.copy(img_path, fold_val_images / img_path.name)
            
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy(label_path, fold_val_labels / label_path.name)
            val_count += 1
        
        log_info(f"Fold {fold_idx}: Train={train_count}, Val={val_count}")
        
        splits.append({
            'fold': fold_idx,
            'train_dir': str(fold_dir / 'train'),
            'val_dir': str(fold_dir / 'val'),
            'train_count': train_count,
            'val_count': val_count
        })
    
    return splits


def train_fold(fold_info, base_config, model_name='yolo11m.pt', epochs=100, imgsz=800):
    fold = fold_info['fold']
    train_dir = fold_info['train_dir']
    val_dir = fold_info['val_dir']
    
    log_info("="*60)
    log_info(f"开始训练 Fold {fold}")
    log_info("="*60)
    
    fold_config = base_config.copy()
    fold_config['train'] = train_dir
    fold_config['val'] = val_dir
    
    config_path = f'yolo_params_fold_{fold}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(fold_config, f)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    batch_size = 8 if device == '0' else 4
    
    model = YOLO(model_name)
    
    project_path = 'kfold_results'
    
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=4,
        project=project_path,
        name=f'fold_{fold}',
        patience=30,
        exist_ok=True,
        weight_decay=0.001,
        dropout=0.1,
        copy_paste=0.3,
        mixup=0.15,
        cls=1.5,
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3.0,
        verbose=True
    )
    
    best_map = results.box.map
    log_info(f"Fold {fold} 训练完成, mAP: {best_map:.4f}")
    
    return best_map


def main():
    log_info("="*60)
    log_info("K-Fold 交叉验证训练")
    log_info("="*60)
    
    this_dir = Path(__file__).parent
    
    with open(this_dir / 'yolo_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_images = this_dir / config['train']
    train_labels = this_dir / config.get('train', 'train').replace('images', 'labels')
    
    if not train_images.exists():
        log_error(f"训练图片目录不存在: {train_images}")
        return
    
    log_info(f"训练图片目录: {train_images}")
    log_info(f"训练标签目录: {train_labels}")
    log_info(f"使用 {N_FOLDS} 折交叉验证")
    
    output_base = this_dir / 'kfold_dataset'
    
    if (output_base / 'fold_0').exists():
        log_info("检测到已有的 K-Fold 数据集，跳过数据分割")
        splits = []
        for i in range(N_FOLDS):
            fold_dir = output_base / f"fold_{i}"
            splits.append({
                'fold': i,
                'train_dir': str(fold_dir / 'train'),
                'val_dir': str(fold_dir / 'val'),
            })
    else:
        log_info("创建 K-Fold 数据集...")
        splits = create_kfold_splits(
            images_dir=train_images,
            labels_dir=train_labels,
            output_base_dir=output_base,
            n_splits=N_FOLDS,
            seed=RANDOM_SEED
        )
    
    map_scores = []
    
    for fold_info in splits:
        map_score = train_fold(
            fold_info=fold_info,
            base_config=config,
            model_name='yolo11m.pt',
            epochs=150,
            imgsz=800
        )
        map_scores.append(map_score)
    
    log_info("="*60)
    log_info("K-Fold 训练完成!")
    log_info("="*60)
    
    for i, score in enumerate(map_scores):
        log_info(f"  Fold {i}: mAP = {score:.4f}")
    
    avg_map = sum(map_scores) / len(map_scores)
    log_info(f"  平均 mAP: {avg_map:.4f}")
    
    best_fold = map_scores.index(max(map_scores))
    log_info(f"  最佳 Fold: {best_fold} (mAP = {max(map_scores):.4f})")
    
    log_info("="*60)
    log_info("可以使用最佳模型进行预测:")
    log_info(f"  模型路径: kfold_results/fold_{best_fold}/weights/best.pt")
    log_info("="*60)


if __name__ == '__main__':
    main()
