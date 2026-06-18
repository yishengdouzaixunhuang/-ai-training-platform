# AI Training Platform v0.1.0

Industrial visual inspection platform — semantic segmentation, object detection, image classification, OCR + OCV.

## Modules

| Module | Technology | Description |
|:---|:---|:---|
| **Semantic Segmentation** | DeepLabV3 R50/R101/Mobile, FCN, LRASPP | Pixel-level defect detection |
| **Object Detection** | YOLOv8 (ultralytics) | HBB + OBB bounding box detection |
| **Image Classification** | ResNet/EfficientNet/MobileNet/ViT | OK/NG defect classification + Grad-CAM heatmap |
| **OCR Text Recognition** | PP-OCRv4 ONNX (RapidOCR) | Text detection + recognition, zero PaddlePaddle |
| **OCV Quality Inspection** | ResNet18 Feature Embedding | Character quality: broken strokes, blur, skew |

## Features

- **Annotation Tools**: Brush, polygon, line, eraser, zoom/pan, undo, auto-save JSON (LabelMe compatible)
- **Training**: K-fold CV, AMP mixed precision, Lovász/Dice/Focal loss, data augmentation, checkpoint resume
- **Inference**: Single/batch, PyTorch/TorchScript/ONNX backends, tiled vs full-image, progress + stop
- **Visualization**: Annotation overlay (hatch pattern), prediction overlay (semi-transparent), Grad-CAM heatmap, OCR bounding boxes
- **Evaluation**: mIoU per class, confusion matrix, loss/accuracy curves, click-to-filter
- **Project Management**: Independent project folders, task type auto-detect, version history, import/export

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA
- PyQt5
- ONNX Runtime

### Install
```bash
pip install torch torchvision pyqt5 opencv-python pillow numpy matplotlib
pip install rapidocr-onnxruntime  # For OCR
pip install ultralytics            # For object detection
pip install scikit-learn           # For evaluation metrics
```

### Run
```bash
cd ai_training_platform
python main.py
```

## Project Structure
```
ai_training_platform/
├── main.py                 # Entry point
├── ui/main_window.py       # Main UI (~3000 lines)
├── annotation/             # Annotation tools & canvas
├── training/               # Segmentation training
├── inference/              # Segmentation inference
├── detection/              # Object detection module
├── classification/         # Image classification module
│   └── gradcam.py          # Grad-CAM heatmap generator
├── ocr/                    # OCR module
│   ├── engine.py           # RapidOCR wrapper
│   └── pipeline.py         # Batch pipeline & overlay
├── ocv/                    # OCV module
│   └── inspector.py        # Character quality inspector
└── core/                   # Config, project manager
```

## Keyboard Shortcuts

| Key | Action |
|:--|:--|
| Space | Toggle prediction/heatmap overlay |
| Ctrl+Space | Toggle annotation overlay |
| Ctrl+[ / Ctrl+] | Toggle left/right panel |
| Ctrl+L | Toggle log panel |
| `` ` `` | Maximize canvas |
| Ctrl+Z | Undo annotation |

## License

Proprietary — All rights reserved.
