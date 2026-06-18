# AI Training Platform v0.1.0 · AI训练平台

Industrial visual inspection platform — semantic segmentation, object detection, image classification, OCR + OCV.
工业级视觉检测平台，集语义分割、目标检测、图像分类、OCR文字识别、OCV字符质检于一体。

## Modules · 功能模块

| Module · 模块 | Technology · 技术 | Description · 说明 |
|:---|:---|:---|
| **Semantic Segmentation** 语义分割 | DeepLabV3 R50/R101/Mobile, FCN, LRASPP | Pixel-level defect detection · 像素级缺陷检测 |
| **Object Detection** 目标检测 | YOLOv8 (ultralytics) | HBB + OBB bounding box detection · 水平框+旋转框 |
| **Image Classification** 图像分类 | ResNet/EfficientNet/MobileNet/ViT | OK/NG defect classification + Grad-CAM heatmap · 缺陷分类+热力图 |
| **OCR Text Recognition** 文字识别 | PP-OCRv4 ONNX (RapidOCR) | Text detection + recognition, zero PaddlePaddle · 文字检测识别，零依赖 |
| **OCV Quality Inspection** 字符质检 | ResNet18 Feature Embedding | Character quality: broken strokes, blur, skew · 断笔/模糊/歪斜检测 |

## Features · 核心功能

### Annotation · 标注工具
- Brush, polygon, line, eraser tools · 画笔、多边形、线条、橡皮擦
- Zoom, pan, undo, auto-save JSON (LabelMe compatible) · 缩放拖拽、撤销、自动保存
- Annotation overlay with hatch pattern · 标注层斜线网格纹理
- Annotation block statistics table · 标注块统计表（面积/周长/灰度/半径）

### Training · 训练
- K-fold cross-validation · K折交叉验证
- AMP mixed precision training · 混合精度训练
- Lovász-Softmax / Dice / Focal / CrossEntropy loss · 多种损失函数
- Data augmentation (3 intensity levels) · 三档数据增强
- Checkpoint resume, best model auto-save · 断点续训、最佳模型保存
- Real-time loss/accuracy curves · 实时训练曲线

### Inference · 推理
- Single / batch inference with progress · 单张/批量推理+进度
- PyTorch / TorchScript / ONNX backends · 多后端支持
- Tiled vs full-image inference · 分块/整图推理
- Stop mid-inference · 中途停止

### Visualization · 可视化
- Annotation overlay (hatch pattern) · 标注层（斜线纹理）
- Prediction overlay (semi-transparent fill) · 推理层（半透明填充）
- Grad-CAM heatmap for classification · 分类热力图（模型注意力）
- OCR bounding boxes + recognized text · OCR检测框+文字
- Confusion matrix with click-to-filter · 混淆矩阵（点击筛选）
- Canvas maximization (`` ` `` key) · 画布最大化

### Evaluation · 评估
- mIoU per class + per-image IoU · 每类/每图交并比
- Confusion matrix visualization · 混淆矩阵可视化
- Loss / Accuracy training curves · 训练曲线
- Click cell → filter image list · 点击格子→筛选图片

### Project Management · 工程管理
- Independent project folders · 独立工程目录
- Task type auto-detect from project.json · 自动识别任务类型
- Version history for annotations · 标注版本历史
- Import / Export JSON annotations · 标注导入导出

## Quick Start · 快速开始

### Requirements · 环境要求
- Python 3.10+
- PyTorch 2.x with CUDA
- PyQt5
- ONNX Runtime

### Install · 安装
```bash
pip install torch torchvision pyqt5 opencv-python pillow numpy matplotlib
pip install rapidocr-onnxruntime  # For OCR · OCR模块
pip install ultralytics            # For object detection · 目标检测
pip install scikit-learn           # For evaluation · 评估指标
```

### Run · 运行
```bash
cd ai_training_platform
python main.py
```

## Project Structure · 项目结构
```
ai_training_platform/
├── main.py                 # Entry point · 入口
├── ui/main_window.py       # Main UI · 主界面
├── annotation/             # Annotation tools & canvas · 标注工具
├── training/               # Segmentation training · 分割训练
├── inference/              # Segmentation inference · 分割推理
├── detection/              # Object detection module · 目标检测
├── classification/         # Image classification module · 图像分类
│   └── gradcam.py          # Grad-CAM heatmap generator · 热力图
├── ocr/                    # OCR module · 文字识别
│   ├── engine.py           # RapidOCR wrapper · OCR引擎
│   └── pipeline.py         # Batch pipeline & overlay · 批量流水线
├── ocv/                    # OCV module · 字符质检
│   └── inspector.py        # Character quality inspector · 质检引擎
└── core/                   # Config, project manager · 配置
```

## Keyboard Shortcuts · 快捷键

| Key · 按键 | Action · 功能 |
|:--|:--|
| Space · 空格 | Toggle prediction/heatmap overlay · 切换推理/热力图 |
| Ctrl+Space | Toggle annotation overlay · 切换标注层 |
| Ctrl+[ / Ctrl+] | Toggle left/right panel · 折叠左/右栏 |
| Ctrl+L | Toggle log panel · 折叠日志栏 |
| `` ` `` | Maximize canvas · 画布最大化 |
| Ctrl+Z | Undo annotation · 撤销标注 |

## License · 许可

Proprietary — All rights reserved.
