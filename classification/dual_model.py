# -*- coding: utf-8 -*-
"""Dual-input classification model 鈥?early fusion of grayscale + height map.

Approach: stack [R, G, B, Height] as 4-channel input, modify first conv layer.
Backbone: ResNet/EfficientNet with 4-channel input.
"""
import torch
import torch.nn as nn

# Model configs (same as single-input)
DUAL_MODELS = {
    "resnet18":  {"name": "ResNet-18 (Dual)",  "family": "resnet"},
    "resnet34":  {"name": "ResNet-34 (Dual)",  "family": "resnet"},
    "resnet50":  {"name": "ResNet-50 (Dual)",  "family": "resnet"},
    "efficientnet_b0": {"name": "EfficientNet-B0 (Dual)", "family": "efficientnet"},
    "mobilenet_v2":    {"name": "MobileNet V2 (Dual)",    "family": "mobilenet"},
}


def _patch_first_conv(model, in_channels=4):
    """Replace first conv layer to accept in_channels input channels."""
    if hasattr(model, "conv1"):
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # Copy RGB weights, init new channel with mean of RGB
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight.clone()
            new_conv.weight[:, 3] = old_conv.weight[:, :3].mean(dim=1)
        model.conv1 = new_conv
        return model

    # EfficientNet: features[0][0]
    if hasattr(model, "features") and hasattr(model.features[0], "__getitem__"):
        try:
            old_conv = model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight.clone()
                new_conv.weight[:, 3] = old_conv.weight[:, :3].mean(dim=1)
            model.features[0][0] = new_conv
            return model
        except (IndexError, AttributeError):
            pass

    # MobileNet: features[0][0]
    if hasattr(model, "features"):
        try:
            first = model.features[0]
            if isinstance(first, nn.Conv2d):
                old_conv = first
                new_conv = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight.clone()
                    new_conv.weight[:, 3] = old_conv.weight[:, :3].mean(dim=1)
                model.features[0] = new_conv
                return model
        except (IndexError, AttributeError):
            pass

    return model


def create_dual_model(model_name, num_classes, pretrained=True):
    """Create a dual-input classification model."""
    cfg = DUAL_MODELS.get(model_name, DUAL_MODELS["resnet18"])
    family = cfg["family"]

    if family == "resnet":
        import torchvision.models as models
        model_fn = getattr(models, model_name, models.resnet18)
        model = model_fn(weights="DEFAULT" if pretrained else None)
        _patch_first_conv(model, 4)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif family == "efficientnet":
        import torchvision.models as models
        model_fn = getattr(models, model_name, models.efficientnet_b0)
        model = model_fn(weights="DEFAULT" if pretrained else None)
        _patch_first_conv(model, 4)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif family == "mobilenet":
        import torchvision.models as models
        model_fn = getattr(models, "mobilenet_v2", models.mobilenet_v2)
        model = model_fn(weights="DEFAULT" if pretrained else None)
        _patch_first_conv(model, 4)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model family: {family}")

    return model
