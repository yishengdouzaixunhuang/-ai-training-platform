"""语义分割模型"""
import os, torch, torch.nn as nn, torchvision
from training.models_ext import MODEL_REGISTRY as EXT_MODELS

# Set writable cache path to avoid sandbox permission issues
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), "Documents", "Codex", ".torch_cache"))

def create_model(num_classes, model_name="deeplabv3", pretrained=True):
    if model_name == "deeplabv3":
        try:
            weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
        except Exception:
            # Fallback: no pretrained weights (offline mode)
            import warnings
            warnings.warn("Could not download pretrained weights, training from scratch.")
            model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        return model
    elif model_name == "fcn":
        try:
            weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
        except Exception:
            import warnings
            warnings.warn("Could not download pretrained weights, training from scratch.")
            model = torchvision.models.segmentation.fcn_resnet50(weights=None)
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        return model
    elif model_name in EXT_MODELS:
        return EXT_MODELS[model_name](num_classes, pretrained=pretrained)
    else:
        raise ValueError("Unknown model: %s" % model_name)
