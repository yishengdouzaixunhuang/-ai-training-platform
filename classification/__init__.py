# -*- coding: utf-8 -*-
"""Classification Module — image classification training & inference.

Mirrors the detection/ package pattern, providing:
- ClassificationDataset: ImageFolder + JSON dual-mode dataset
- ClassificationTrainer: K-fold / AMP / resume training wrapper
- evaluate_classification: Top-1/Top-5 accuracy + confusion matrix
- LabelManager: class_labels.json I/O + folder auto-labeling
"""

from .dataset import ClassificationDataset
from .trainer import ClassificationTrainer, CLS_MODELS
from .eval import evaluate_classification
from .label_manager import LabelManager

__all__ = [
    "ClassificationDataset",
    "ClassificationTrainer",
    "CLS_MODELS",
    "evaluate_classification",
    "LabelManager",
]