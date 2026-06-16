# 图像分类模块 PRD（Product Requirements Document）

## 1. 概述

### 1.1 背景
AI Training Platform 当前支持 **语义分割**（`training/`）和 **目标检测**（`detection/`）两大任务。本 PRD 规划新增 **图像分类** 模块，形成三大 CV 任务闭环：分类 → 检测 → 分割。

### 1.2 与现有模块的关系

| 现有模块 | 核心数据结构 | 标注格式 | 训练框架 | 推理输出 |
|----------|-------------|---------|---------|---------|
| **语义分割** | 像素级 mask (H×W) | LabelMe JSON | torchvision | 每类 mask + 叠加图 |
| **目标检测** | 矩形框列表 [x,y,w,h,cls] | COCO JSON / YOLO TXT | Ultralytics YOLO | 框 + 类别 + 置信度 |
| **图像分类** (新) | 单标签/多标签 per 图 | 文件夹名 / JSON | timm / torchvision | 类别名 + 置信度 |

### 1.3 设计原则
- **复用优先**：工程管理、数据划分 (train/val/test)、UI 框架、推理后端全部复用
- **模式对齐**：分类模块文件结构镜像 `detection/` 包，降低认知负担
- **轻量切入**：分类是最简单的 CV 任务，优先实现单标签，后续扩展多标签

---

## 2. 功能需求

### 2.1 工程管理 (优先级 P0)

| 序号 | 功能 | 详细说明 | 复用/新建 |
|------|------|---------|----------|
| C01 | 工程任务类型扩展 | `project.json` 中 `task_type` 新增 `"图像分类"` 选项 | 修改 `core/project_manager.py` |
| C02 | 分类工程目录结构 | `images/` + `annotations/`(存class_labels.json) + `models/` + `outputs/` | 复用现有模板 |
| C03 | 任务类型切换 | 工程级别切换三种任务类型，切换后面板适配 | 扩展 `ui/main_window.py` |

### 2.2 数据管理 (优先级 P0)

| 序号 | 功能 | 详细说明 |
|------|------|---------|
| C04 | 图片导入 | 复用现有导入逻辑，支持 BMP/PNG/JPG |
| C05 | 类别标签管理 | `class_labels.json`：`{"0": "cat", "1": "dog", ...}`，支持手动编辑 |
| C06 | 文件夹自动标注 | 按子文件夹名自动分配类别（ImageFolder 模式）|
| C07 | 训练/验证/测试集划分 | 复用现有三态划分 train/val/test |
| C08 | 自动划分数据集 | 复用 Auto Split 70/20/10 + K折交叉验证 |
| C09 | 图片列表筛选 | 按名称/类别/数据集筛选 |
| C10 | 分类标注统计 | 每类样本数柱状图 + 类别分布饼图 + 标注进度 |

### 2.3 训练 (优先级 P0)

| 序号 | 功能 | 详细说明 |
|------|------|---------|
| C11 | 模型选择 | ResNet18/34/50/101, EfficientNet-B0~B4, MobileNetV3, ViT-B/16 |
| C12 | 预训练权重 | ImageNet 预训练，自动下载/离线缓存 |
| C13 | 分类头替换 | 自动替换 FC 层适配类别数 |
| C14 | 训练参数 | epochs / batch_size / lr / optimizer (Adam/SGD/AdamW) / weight_decay |
| C15 | 损失函数 | CrossEntropy / LabelSmoothing / FocalLoss |
| C16 | 学习率调度 | CosineAnnealing / StepLR / ReduceLROnPlateau |
| C17 | 数据增强 | RandomResizedCrop / HorizontalFlip / ColorJitter / RandAugment，3档强度 |
| C18 | 混合精度 AMP | 复用现有 AMP 逻辑 |
| C19 | 断点续训 | Resume 从 last_model.pth 恢复 |
| C20 | 多进程加载 | 复用 num_workers 逻辑 |
| C21 | 批次进度+指标 | 实时 batch loss + Top-1/Top-5 Accuracy + 每类 Precision/Recall/F1 |
| C22 | 最佳模型保存 | 自动保存最高 Val Accuracy 模型 |
| C23 | K折交叉验证 | 复用现有 K-fold 逻辑 |
| C24 | 训练停止 | 中途停止 + GPU 显存释放 |
| C25 | 类别权重 | 自动逆频率权重处理不均衡数据 |
| C26 | 混淆矩阵 | 训练完成后自动生成 + matplotlib 可视化 |

### 2.4 推理 (优先级 P0)

| 序号 | 功能 | 详细说明 |
|------|------|---------|
| C27 | 单图推理 | 对当前图片分类，显示 Top-5 类别+置信度 |
| C28 | 批量推理 | 遍历全部图片推理，输出 CSV/JSON 结果 |
| C29 | 推理进度+耗时 | 复用现有实时进度 + 每张耗时显示 |
| C30 | 推理停止 | 复用现有停止按钮 |
| C31 | 模型选择 | 复用模型下拉框，自动识别分类/检测/分割模型 |
| C32 | 推理后端 | PyTorch / TorchScript / ONNX Runtime |
| C33 | 推理结果可视化 | 图片列表新增"预测类别"列，右侧显示 Top-5 置信度柱状图 |

### 2.5 评估 (优先级 P1)

| 序号 | 功能 | 详细说明 |
|------|------|---------|
| C34 | Top-1 / Top-5 Accuracy | 验证集整体指标 |
| C35 | 每类 Precision / Recall / F1 | 输出各类别定位短板 |
| C36 | 混淆矩阵可视化 | seaborn/matplotlib 热力图 |
| C37 | 模型管理 | 复用多模型列表对比 |

### 2.6 部署 (优先级 P2)

| 序号 | 功能 | 详细说明 |
|------|------|---------|
| C38 | ONNX 导出 | torch.onnx.export |
| C39 | TorchScript 导出 | torch.jit.script/trace |

### 2.7 可视化 (优先级 P1)

| 序号 | 功能 | 详细说明 |
|------|------|---------|
| C40 | 批量推理结果 CSV 导出 | 文件路径 + Top-1~Top-5 类别 + 置信度 |
| C41 | 推理结果颜色标记 | 正确分类绿色、错误分类红色 |
| C42 | 类别置信度柱状图 | 推理面板显示单图 Top-5 置信度 bar chart |

---

## 3. 技术设计

### 3.1 目录结构

```
work/ai_training_platform/
├── classification/              # 新增包，镜像 detection/
│   ├── __init__.py              # 导出 Classifier, ClassificationDataset, ...
│   ├── dataset.py               # ClassificationDataset (ImageFolder + JSON 标注)
│   ├── trainer.py               # ClassificationTrainer
│   ├── eval.py                  # evaluate_classification()
│   └── label_manager.py         # 分类标签管理
├── training/                    # 现有分割训练 (不变)
├── detection/                   # 现有检测训练 (不变)
├── inference/
│   └── predictor.py             # 扩展：新增 classify() 方法
├── core/
│   └── project_manager.py       # 扩展：task_type 新增 "图像分类"
└── ui/
    └── main_window.py           # 扩展：分类面板 + 模式切换
```

### 3.2 核心类设计

#### 3.2.1 ClassificationDataset (`classification/dataset.py`)

```python
class ClassificationDataset(torch.utils.data.Dataset):
    """
    支持两种标注模式：
    1. ImageFolder 模式: images/cat/xxx.jpg, images/dog/xxx.jpg (自动从文件夹名提取类别)
    2. JSON 标注模式: labels.json -> {"xxx.jpg": 0, "yyy.jpg": 1}
    
    split: "all" | "train" | "val" | "test"  (复用 split.json 或 split_cls.json)
    """
    def __init__(self, project_dir, split="all", image_size=224, augment_fn=None)
    def __len__(self) -> int
    def __getitem__(self, idx) -> Tuple[Tensor, int]
    @property
    def num_classes(self) -> int
    @property
    def class_names(self) -> List[str]
```

#### 3.2.2 ClassificationTrainer (`classification/trainer.py`)

```python
class ClassificationTrainer:
    """
    镜像 detection/trainer.py 和 training/trainer.py 的设计模式：
    - K折交叉验证
    - 混合精度 AMP
    - 断点续训
    - 进度回调 (progress_callback, log_callback, batch_callback, plot_callback)
    - 停止检查 (stop_check)
    - 自动保存最佳模型
    """
    MODELS = {
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v3_small": "MobileNetV3-Small",
        "vit_b_16": "ViT-B/16",
    }
    def __init__(self, project_dir, model_name="resnet18", device=None)
    def prepare_data(self, batch_size, image_size=224, augment="medium")
    def init_model(self)
    def train(self, epochs, ...)
    def cleanup(self)
```

#### 3.2.3 Predictor 扩展 (`inference/predictor.py`)

```python
class Predictor:
    # 现有方法不变
    def predict(self, image_path)  # 分割推理
    def predict_detection(self, image_path)  # 检测推理
    
    # 新增分类推理方法
    def classify(self, image_path, top_k=5) -> List[Tuple[str, float]]:
        """返回 [(类别名, 置信度), ...] Top-K 结果"""
```

### 3.3 数据流设计

```
图片导入 → images/
         ↓
    ┌────────────────┬──────────────────┐
    │ ImageFolder模式 │  JSON 标注模式   │
    │ (文件夹=类别)    │  (class_labels.   │
    │ 自动生成json    │   json 手动标注)  │
    └────────┬───────┴────────┬─────────┘
             ↓                ↓
        ClassificationDataset
             ↓
    ┌────────┴────────┐
    │ split.json 划分  │
    │ train/val/test  │
    └────────┬────────┘
             ↓
    ClassificationTrainer
             ↓
    models/best_model.pth
             ↓
    Predictor.classify()
             ↓
    outputs/result.csv
```

### 3.4 UI 设计要点

#### 3.4.1 分类训练面板 (新增)
```
┌─────────────────────────────────────┐
│ 模型: [ResNet50 ▼]  图像尺寸: [224] │
│ Batch: [32]   Epochs: [100]         │
│ 优化器: [AdamW ▼]  LR: [0.001]     │
│ 数据增强: [中 ▼]                     │
│ 损失函数: [CrossEntropy ▼]          │
│ [✓] 混合精度AMP   [✓] K折交叉 K=[5] │
│           [开始训练]  [停止训练]     │
└─────────────────────────────────────┘
```

#### 3.4.2 分类推理面板 (新增)
```
┌─────────────────────────────────────┐
│ 模型: [best_model.pth ▼]            │
│ 后端: [PyTorch ▼]  Top-K: [5]      │
│ [单图推理]  [批量推理]  [停止推理]  │
│                                      │
│ 结果:                                │
│ #1 cat   ████████████ 98.3%        │
│ #2 dog   ██            1.2%        │
│ #3 bird  █             0.3%        │
└─────────────────────────────────────┘
```

### 3.5 标注数据格式

**class_labels.json**（单标签）:
```json
{
    "labels": ["cat", "dog", "bird", "horse"],
    "mapping": {
        "images/cat/001.jpg": 0,
        "images/cat/002.jpg": 0,
        "images/dog/001.jpg": 1
    }
}
```

**ImageFolder 自动映射**：
- `images/cat/*.jpg` → class 0 ("cat")
- `images/dog/*.jpg` → class 1 ("dog")
- 自动生成 `class_labels.json`

---

## 4. 与现有模块的差异对比

| 维度 | 语义分割 | 目标检测 | 图像分类 |
|------|---------|---------|---------|
| 输出粒度 | 像素级 | 框级 | 图级 |
| 每图标注量 | 数百万像素 | 0~N 个框 | 1 个标签 |
| 标注成本 | 极高 | 高 | 低 |
| 训练框架 | torchvision | Ultralytics | timm + torchvision |
| 典型模型 | DeepLabV3 | YOLOv8 | ResNet/EfficientNet |
| 主要指标 | mIoU | mAP@0.5 | Top-1 Accuracy |
| 损失函数 | CrossEntropy/Dice | CIoU+DFL | CrossEntropy |
| 推理输出 | mask 图 | 框+分数 | 类别+置信度 |

---

## 5. 实现计划

### 阶段 1：核心功能 (预计 6-8 小时)

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 扩展工程管理支持分类 | `core/project_manager.py` | 0.5h |
| 实现 ClassificationDataset | `classification/dataset.py` (新建) | 2h |
| 实现 ClassificationTrainer | `classification/trainer.py` (新建) | 3h |
| 扩展 Predictor.classify() | `inference/predictor.py` | 1h |
| 评估函数 | `classification/eval.py` (新建) | 1h |

### 阶段 2：UI 集成 (预计 8-10 小时)

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 任务类型切换 + 分类面板布局 | `ui/main_window.py` | 5h |
| 分类训练面板 (模型选择/参数/进度) | `ui/main_window.py` | 2h |
| 分类推理面板 (Top-K 结果展示) | `ui/main_window.py` | 2h |
| 分类标注管理界面 | `ui/main_window.py` + `classification/label_manager.py` | 1h |

### 阶段 3：增强功能 (预计 4-6 小时)

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 混淆矩阵可视化 | `classification/eval.py` | 1h |
| ONNX/TorchScript 导出 | `classification/trainer.py` | 1h |
| 多标签分类支持 | `classification/dataset.py` + `trainer.py` | 2h |
| 单元测试 | `tests/test_classification.py` (新建) | 2h |

**总计预估：18-24 小时**

---

## 6. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| UI 文件 `main_window.py` 已达 2000+ 行，加入分类面板后更臃肿 | 高 | 考虑后续重构为多文件 UI 模块 |
| timm 库依赖 (EfficientNet/ViT) | 中 | 优先 torchvision 内置模型 (ResNet/MobileNet)，timm 作为可选 |
| ImageFolder 模式与现有 JSON 标注体系不一致 | 低 | 自动将文件夹结构转为 class_labels.json，统一数据接口 |
| 训练指标展示区域有限 | 低 | 分类指标维度少 (Acc/F1/混淆矩阵)，只需 2-3 个图表 |

---

## 7. 验收标准

- [ ] 创建"图像分类"类型工程正常，目录结构正确
- [ ] ImageFolder 自动标注 + JSON 手动标注两种模式均可用
- [ ] 训练启动/停止正常，K折交叉验证可用
- [ ] 断点续训可从 last_model.pth 恢复
- [ ] 单图推理返回 Top-5 类别+置信度
- [ ] 批量推理输出 CSV，包含每张图的预测结果
- [ ] 混淆矩阵和 Accuracy 图表正确生成
- [ ] 三种任务类型 (分割/检测/分类) 工程切换无异常