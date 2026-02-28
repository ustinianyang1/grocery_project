# Grocery Detection Project

## 项目概述

本项目基于 YOLO 系列模型进行超市商品检测，当前默认训练版本为 **YOLOv11m + AMP**。项目已迭代多个版本，支持完整的训练、验证、预测与提交文件转换流程。

当前主流程对应目录为：
- 训练输出：`grocery_local/v11m_amp/`
- 验证/预测权重：`grocery_local/v11m_amp/weights/best.pt`
- 预测输出：`predictions/images/` 与 `predictions/labels/`

## 目录结构

> 带 `*` 标注的文件/目录在 `.gitignore` 中，运行后生成或需手动准备，不纳入版本控制。

```text
grocery_project/
├── train.py                  # 训练脚本
├── validate.py               # 验证脚本
├── predict.py                # 预测脚本
├── convert_preds_to_csv.py   # 预测结果转 CSV 脚本
├── yolo_params.yaml          # YOLO 数据配置文件
├── requirements.txt          # Python 依赖列表
├── README.md
├── .gitignore
├── yolo11m.pt              * # 预训练权重（训练前需下载）
├── submission.csv          * # 提交文件（由转换脚本生成）
├── training.log            * # 训练日志（运行后生成）
├── validation.log          * # 验证日志（运行后生成）
├── predict.log             * # 预测日志（运行后生成）
├── grocery_local/          * # 训练输出目录（运行后生成）
│   └── v11m_amp/
│       ├── args.yaml
│       ├── results.csv
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── train/                  * # 训练数据（需手动准备）
│   ├── images/
│   └── labels/
├── val/                    * # 验证数据（需手动准备）
│   ├── images/
│   └── labels/
├── testImages/             * # 测试数据（需手动准备）
│   └── images/
└── predictions/            * # 预测结果（运行后生成）
    ├── images/
    └── labels/
```

## 环境与安装

### 1) 创建并激活虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：
- `ultralytics`
- `torch==2.1.2+cu121`、`torchvision==0.16.2+cu121`
- `opencv-python==4.8.0.76`
- `albumentations`、`pandas`、`pyyaml`
- `numpy<2`

## 数据与配置

配置文件：`yolo_params.yaml`

```yaml
train: D:\grocery_project\train\images
val: D:\grocery_project\val\images
test: D:\grocery_project\testImages\images
nc: 3
names: ['cheerios', 'soup', 'candle']
```

说明：
- `nc`: 类别数（当前为 3）
- `names`: 类别名称
- `train/val/test`: 数据路径（可改为相对路径，脚本会自动拼接项目根目录）

## 使用说明

### 1) 训练

```bash
python train.py
```

当前训练脚本关键行为：
- 默认模型：`YOLO('yolo11m.pt')`
- 设备：自动检测（CUDA 可用时 `device='0'`，否则 `cpu`）
- 输出目录：`grocery_local/v11m_amp/`
- 日志文件：`training.log`

当前主要训练参数：
- `epochs=300`
- `imgsz=640`
- `batch=8 (GPU) / 4 (CPU)`
- `workers=4 (GPU) / 2 (CPU)`
- `optimizer='AdamW'`
- `amp=True`
- `cache=True`
- `weight_decay=0.001`
- `dropout=0.1`
- `copy_paste=0.3`
- `mixup=0.15`
- `cls=1.2`
- `lr0=0.001`
- `cos_lr=True`
- `warmup_epochs=5.0`

### 2) 验证

```bash
python validate.py
```

当前验证脚本：
- 默认读取模型：`grocery_local/v11m_amp/weights/best.pt`
- 使用数据配置：`yolo_params.yaml`
- 验证参数：`imgsz=800`、`device=0`、`workers=4`
- 日志文件：`validation.log`

说明：
- 当前 `validate.py` 固定使用 `device=0`，仅在有可用 CUDA 设备时可直接运行。

### 3) 预测

```bash
python predict.py
```

当前预测脚本：
- 默认读取模型：`grocery_local/v11m_amp/weights/best.pt`
- 测试目录来源：`yolo_params.yaml` 中的 `test`
- 置信度阈值：`conf=0.15`
- 支持图片后缀：`.jpg`、`.png`
- 输出目录：
    - 可视化图片：`predictions/images/`
    - 标签文本：`predictions/labels/`
- 日志文件：`predict.log`

标签文件每行格式为：
`class_id confidence x_center y_center width height`

### 4) 转换为 Kaggle 提交 CSV

```bash
python convert_preds_to_csv.py
```

或自定义参数：

```bash
python convert_preds_to_csv.py --preds_folder predictions/labels --output_csv submission.csv --test_images_folder testImages/images
```

转换脚本会：
- 读取 `predictions/labels/*.txt`
- 对每行预测结果进行字段数量与数值合法性校验
- 以测试集图片清单为基准生成 `submission.csv`

## 版本说明

当前脚本默认使用的主线版本是：
- 训练：`yolo11m.pt`
- 输出实验名：`v11m_amp`
- 验证/预测：`grocery_local/v11m_amp/weights/best.pt`

如需切换到其他预训练权重或实验名，请同步修改：
- `train.py` 中的预训练权重路径与 `name`
- `validate.py` / `predict.py` 中的模型路径

## 注意事项

- 训练脚本使用 `imgsz=640`，验证脚本使用 `imgsz=800`，属于当前代码中的既定配置。
- 预测脚本当前仅处理后缀为 `.jpg` 和 `.png` 的图片文件。
- 转换脚本默认读取 `predictions/labels`，并以 `testImages/images` 的图片清单作为提交基准。

## 许可证

本项目采用 MIT 许可证（如仓库内提供 `LICENSE` 文件则以其为准）。