
"""Extended segmentation models - pure torchvision, zero extra dependencies"""
import torch.nn as nn
import torchvision


def _replace_classifier(model, num_classes, classifier_attr="classifier", conv_index=4):
    """Replace the final conv layer of a torchvision segmentation model."""
    classifier = getattr(model, classifier_attr)
    in_channels = classifier[conv_index].in_channels
    classifier[conv_index] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model


def deeplabv3_resnet101(num_classes, pretrained=True):
    """DeepLabV3 with ResNet-101 backbone - stronger than ResNet-50."""
    try:
        weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    except Exception:
        import warnings
        warnings.warn("Could not download pretrained weights, training from scratch.")
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
    return _replace_classifier(model, num_classes)


def deeplabv3_mobilenet(num_classes, pretrained=True):
    """DeepLabV3 with MobileNetV3-Large backbone - fast inference, smaller."""
    try:
        weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    except Exception:
        import warnings
        warnings.warn("Could not download pretrained weights, training from scratch.")
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
    return _replace_classifier(model, num_classes)


def fcn_resnet101(num_classes, pretrained=True):
    """FCN with ResNet-101 backbone."""
    try:
        weights = torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT if pretrained else None
        model = torchvision.models.segmentation.fcn_resnet101(weights=weights)
    except Exception:
        import warnings
        warnings.warn("Could not download pretrained weights, training from scratch.")
        model = torchvision.models.segmentation.fcn_resnet101(weights=None)
    return _replace_classifier(model, num_classes)


def lraspp_mobilenet(num_classes, pretrained=True):
    """LRASPP with MobileNetV3-Large - lightweight, good for fast experiments."""
    try:
        weights = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
    except Exception:
        import warnings
        warnings.warn("Could not download pretrained weights, training from scratch.")
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    # LRASPP uses different classifier structure
    low_in = model.classifier.low_classifier.in_channels
    high_in = model.classifier.high_classifier.in_channels
    model.classifier.low_classifier = nn.Conv2d(low_in, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(high_in, num_classes, kernel_size=1)
    return model


# Registry: name -> builder function
MODEL_REGISTRY = {
    "deeplabv3_r101": deeplabv3_resnet101,
    "deeplabv3_mobilenet": deeplabv3_mobilenet,
    "fcn_r101": fcn_resnet101,
    "lraspp": lraspp_mobilenet,
}
