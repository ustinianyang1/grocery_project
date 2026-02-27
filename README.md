# Grocery Detection Project

## 项目概述

本项目基于YOLOv11模型，用于检测和识别超市商品。通过本地训练的方式，利用自定义数据集实现高精度的商品检测。

## 目录结构

```
D:/grocery_project/
├── train/              # 训练集
│   └── train/
│       ├── images/     # 训练图片
│       └── labels/     # 训练标签
├── val/                # 验证集
│   └── val/
│       ├── images/     # 验证图片
│       └── labels/     # 验证标签
├── train.py            # 本地训练脚本
├── validate.py         # 模型验证脚本
├── predict.py          # 模型预测脚本
├── convert_preds_to_csv.py  # 预测结果转换为CSV脚本
├── local_data.yaml     # 本地数据配置文件
├── yolo_params.yaml    # YOLO模型参数配置
├── yolo11s.pt          # 预训练模型权重
├── training.log        # 训练日志文件
├── .gitignore          # Git忽略文件
└── README.md           # 项目文档
```

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

```bash
pip install ultralytics pyyaml opencv-python
```

### 3. 准备数据集

从Kaggle下载数据集并解压到上述目录结构中：
- 将训练图片放入 `train/train/images/`
- 将训练标签放入 `train/train/labels/`
- 将验证图片放入 `val/val/images/`
- 将验证标签放入 `val/val/labels/`

### 4. 配置模型参数

编辑 `yolo_params.yaml` 文件，设置模型的类别信息和数据集路径：

```yaml
train: train/train/images
val: val/val/images
test: testImages/images
nc: 3
names: ['cheerios', 'soup', 'candle']
```

- **nc**: 类别数量（3个类别：cheerios、soup、candle）
- **names**: 类别名称列表
- **train/val/test**: 数据集路径

### 5. 准备测试数据集

- 创建 `testImages/images/` 目录
- 将测试图片放入该目录

## 使用说明

### 训练模型

运行训练脚本开始训练：

```bash
python train.py
```

训练过程中，模型会自动：
1. 生成本地配置文件 `local_data.yaml`
2. 下载预训练权重 `yolo11s.pt`
3. 启动训练过程
4. 将训练结果保存到 `grocery_local/v11s_optimized/` 目录
5. 生成训练日志文件 `training.log`

### 验证模型

训练完成后，运行验证脚本评估模型性能：

```bash
python validate.py
```

验证过程会：
1. 加载训练好的最佳模型 `grocery_local/v11s_optimized/weights/best.pt`
2. 在验证集上执行评估
3. 输出详细的验证指标（mAP、精确率、召回率等）
4. 将验证结果保存到 `grocery_local/v11s_optimized/val_results.json`

### 预测模型

使用训练好的模型进行预测：

```bash
python predict.py
```

预测过程会：
1. 加载训练好的最佳模型 `grocery_local/v11s_optimized/weights/best.pt`
2. 对 `testImages/images/` 目录中的测试图片进行预测
3. 生成带标注的预测结果图片
4. 保存边界框数据到文本文件

### 转换预测结果

将预测结果转换为CSV格式：

```bash
python convert_preds_to_csv.py
```

转换过程会：
1. 读取预测结果
2. 转换为CSV格式
3. 保存为可提交的格式

### 训练参数

- **epochs**: 100 轮训练
- **imgsz**: 800 分辨率
- **batch**: 自动分配（根据显存大小）
- **device**: 使用第一块显卡
- **workers**: 4 个数据加载线程
- **patience**: 20 轮早停机制

## 功能介绍

1. **自动路径检测**：脚本会自动检测本地路径，无需手动配置
2. **智能Batch分配**：根据显卡显存自动分配最佳Batch Size
3. **实时监控**：训练过程中可查看实时结果图表
4. **早停机制**：当验证精度不再提升时自动停止训练
5. **完整日志**：详细的训练日志和终端输出

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