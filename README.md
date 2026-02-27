# Grocery Detection Project

## 项目概述

本项目基于YOLOv11模型，用于检测和识别超市商品。通过本地训练的方式，利用自定义数据集实现高精度的商品检测。项目支持完整的训练、验证、预测和结果转换流程。

## 目录结构

```
D:/grocery_project/
├── train.py            # 本地训练脚本
├── validate.py         # 模型验证脚本
├── predict.py          # 模型预测脚本
├── convert_preds_to_csv.py  # 预测结果转换为CSV脚本
├── requirements.txt    # 项目依赖文件
├── .gitignore          # Git忽略文件
├── README.md           # 项目文档
├── yolo_params.yaml    # 模型配置文件
├── submission.csv      # Kaggle提交文件
├── train/              # 训练数据集
│   ├── images/         # 训练图片
│   └── labels/         # 训练标签
├── val/                # 验证数据集
│   ├── images/         # 验证图片
│   └── labels/         # 验证标签
├── testImages/         # 测试数据集
│   └── images/         # 测试图片
├── grocery_local/      # 训练结果保存目录
│   └── v11s_optimized/ # 模型训练结果
└── predictions/        # 预测结果保存目录
    ├── images/         # 带标注的预测结果图片
    └── labels/         # 边界框数据文本文件
```

**训练过程中会生成的文件：**
- `yolo11s.pt` - YOLOv11s预训练权重（自动下载）
- `training.log` - 训练日志
- `validation.log` - 验证日志
- `predict.log` - 预测日志

## 安装指南

### 1. 环境准备

确保你使用的是项目虚拟环境，而非系统全局环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

使用 requirements.txt 文件安装项目依赖：

```bash
pip install -r requirements.txt
```

### 3. 准备数据集

从Kaggle下载数据集并按照以下结构组织：
- 创建 `train/images/` 目录，放入训练图片
- 创建 `train/labels/` 目录，放入训练标签
- 创建 `val/images/` 目录，放入验证图片
- 创建 `val/labels/` 目录，放入验证标签
- 创建 `testImages/images/` 目录，放入测试图片

### 4. 配置模型参数

项目已自带 `yolo_params.yaml` 文件，无需手动创建。文件中设置了模型的类别信息和数据集路径：

```yaml
train: D:\grocery_project\train\images
val: D:\grocery_project\val\images
test: D:\grocery_project\testImages\images
nc: 3
names: ['cheerios', 'soup', 'candle']
```

- **nc**: 类别数量（3个类别：cheerios、soup、candle）
- **names**: 类别名称列表
- **train/val/test**: 数据集的绝对路径

## 使用说明

### 训练模型

运行训练脚本开始训练：

```bash
python train.py
```

训练过程中，模型会自动：
1. 加载 `yolo_params.yaml` 配置文件
2. 加载预训练权重 `yolo11s.pt`（如果不存在会自动下载）
3. 检测设备（优先使用GPU，无GPU时自动切换到CPU）
4. 根据设备自动调整训练参数（batch size和workers）
5. 启动训练过程（300个epoch）
6. 将训练结果保存到 `grocery_local/v11s_optimized/` 目录（工作区内）
7. 生成训练日志文件 `training.log`

### 验证模型

训练完成后，运行验证脚本评估模型性能：

```bash
python validate.py
```

验证过程会：
1. 加载 `yolo_params.yaml` 配置文件
2. 加载训练好的最佳模型 `grocery_local/v11s_optimized/weights/best.pt`
3. 在验证集上执行评估
4. 输出详细的验证指标（mAP、精确率、召回率等）
5. 生成验证日志文件 `validation.log`

### 预测模型

使用训练好的模型进行预测：

```bash
python predict.py
```

预测过程会：
1. 加载 `yolo_params.yaml` 配置文件
2. 从 `grocery_local/v11s_optimized/weights/best.pt` 加载训练好的最佳模型
3. 对配置文件中指定的测试图片目录中的测试图片进行预测
4. 生成带标注的预测结果图片，保存到 `predictions/images/` 目录（工作区内）
5. 保存边界框数据（包含置信度）到文本文件，保存到 `predictions/labels/` 目录（工作区内）
6. 生成预测日志文件 `predict.log`

### 转换预测结果

将预测结果转换为CSV格式：

```bash
python convert_preds_to_csv.py
```

转换过程会：
1. 读取 `predictions/labels/` 目录中的预测结果
2. 对预测结果进行严格验证（确保格式正确）
3. 转换为Kaggle提交格式的CSV文件
4. 保存为 `submission.csv` 文件（工作区内）
5. 输出详细的转换统计信息

### 训练参数

- **epochs**: 300 轮训练
- **imgsz**: 800 分辨率
- **batch**: 自动分配（GPU: 16, CPU: 4）
- **device**: 自动检测（优先使用GPU）
- **workers**: 自动分配（GPU: 4, CPU: 2）
- **project**: grocery_local（训练结果保存目录）
- **name**: v11s_optimized（模型训练结果子目录）
- **patience**: 50 轮早停机制
- **weight_decay**: 0.001
- **dropout**: 0.1
- **copy_paste**: 0.4
- **mixup**: 0.2
- **cls**: 1.5
- **lr0**: 0.001
- **cos_lr**: True
- **warmup_epochs**: 5.0

## 功能介绍

1. **自动路径检测**：脚本会自动检测本地路径，无需手动配置
2. **智能参数调整**：根据设备类型自动调整batch size和workers数量
3. **设备自适应**：自动检测GPU是否可用，无GPU时自动切换到CPU模式
4. **早停机制**：当验证精度不再提升时自动停止训练（patience=50）
5. **完整日志**：详细的训练、验证和预测日志
6. **工作区保存**：所有结果都保存在项目工作区内，避免使用YOLO默认目录
7. **严格的结果验证**：转换预测结果时进行严格的验证，确保提交格式正确
8. **增强的数据增强**：使用copy_paste和mixup等数据增强技术提升模型性能
9. **优化的训练参数**：包含weight_decay、dropout等正则化参数，避免过拟合
10. **灵活的学习率策略**：使用cosine学习率调度和warmup机制

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

如有问题或建议，请通过 GitHub Issues 提交。