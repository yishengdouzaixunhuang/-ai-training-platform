"""语义分割模型"""
import os, torch, torch.nn as nn, torchvision
from training.models_ext import MODEL_REGISTRY as EXT_MODELS

# Set writable cache path to avoid sandbox permission issues
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), "Documents", "Codex", ".torch_cache"))

def create_model(num_classes, model_name="deeplabv3", pretrained=True, in_channels=3,
                 unet_base=16, unet_depth=4):
    """建分割模型。`in_channels` != 3 时（多视图工程，如三图 9 通道）改写首层卷积。

    ★ 2026-10-15 支持多视图通道：原来首层写死 3 通道，于是三图工程训练**直接报错**
      `RuntimeError: weight of size [64, 3, 7, 7], expected input[2, 9, 512, 512]` ——
      不是变慢，是根本跑不起来（有没有 GPU 都一样）。

    首层改写**复用 `training.multiview.patch_first_conv`**（唯一实现）：新通道权重按
    I3D 式膨胀 —— 每个视图复制同一份 RGB 预训练核并乘 3/in_channels 保持激活量级。
    为什么不用 `weights=None` 从零训：三图工程只有 33 组数据，从零训 DeepLabV3
    几乎不收敛；膨胀初始化保留预训练特征，收敛快得多。
    想要"完全从零训"：`create_model(..., pretrained=False, in_channels=9)`。

    ⚠️ 惰性import multiview：那里已`from torchvision import models`，
      而 multiview 不import 本模块，故无循环依赖。
    """
    if in_channels and int(in_channels) != 3:
        from training.multiview import patch_first_conv
    else:
        patch_first_conv = None

    def _finish(model):
        """换输出层（num_classes）+ 按需改首层（in_channels）。"""
        conv_idx = 4 if hasattr(model, "classifier") and hasattr(model.classifier, "__getitem__") else None
        if conv_idx is not None and hasattr(model.classifier[conv_idx], "in_channels"):
            _in = model.classifier[conv_idx].in_channels
            model.classifier[conv_idx] = nn.Conv2d(_in, num_classes, kernel_size=1)
        if patch_first_conv is not None:
            model = patch_first_conv(model, int(in_channels))
        return model

    # ★ 2026-10-21 新增：单图工程也支持 U-Net（原来只有多视图能用）
    #   为什么：200×200 这种小图，DeeplabV3-ResNet50（3900 万参数）推理 7-11ms，
    #   而小 U-Net（base=16，约 200 万参数）能到 2-3ms，和友商同档。
    #   U-Net 本身不分单图/多视图 —— MultiViewUNet 的第一层就是普通卷积，
    #   in_channels=3 时就是标准单图 U-Net；不加载预训练，所以离线也能用。
    if model_name == "unet":
        # ⚠️ 这里**不要**包成返回 tensor —— 单图训练器写的是
        #   `outputs = self.model(images)["out"]`（trainer.py:262），
        #   torchvision 的 deeplabv3/fcn 也返回 OrderedDict(out=...)，口径一致。
        #   我第一次多包了一层返回 tensor，结果 tensor["out"] 直接报
        #   TypeError: new(): invalid data type 'str'。
        from training.multiview import MultiViewUNet
        return MultiViewUNet(int(in_channels or 3), num_classes,
                             base=unet_base, depth=unet_depth)

    if model_name == "deeplabv3":
        try:
            weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
        except Exception:
            # Fallback: no pretrained weights (offline mode)
            import warnings
            warnings.warn("Could not download pretrained weights, training from scratch.")
            model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
        return _finish(model)
    elif model_name == "fcn":
        try:
            weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
        except Exception:
            import warnings
            warnings.warn("Could not download pretrained weights, training from scratch.")
            model = torchvision.models.segmentation.fcn_resnet50(weights=None)
        return _finish(model)
    elif model_name in EXT_MODELS:
        model = EXT_MODELS[model_name](num_classes, pretrained=pretrained)
        return _finish(model) if patch_first_conv is not None else model
    else:
        raise ValueError("Unknown model: %s" % model_name)
