"""Grad-CAM heatmap generator for classification model interpretability."""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def _find_target_layer(model, model_name: str):
    name_lower = model_name.lower()
    if "resnet" in name_lower and hasattr(model, "layer4"):
        return model.layer4[-1]
    if "efficientnet" in name_lower and hasattr(model, "features"):
        for m in reversed(list(model.features.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
    if "mobilenet" in name_lower and hasattr(model, "features"):
        for m in reversed(list(model.features.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None

def generate_gradcam_heatmap(model, image_np, model_name="resnet18", target_class=None, image_size=None):
    try:
        target_layer = _find_target_layer(model, model_name)
        if target_layer is None:
            return None
        from torchvision.transforms import functional as TF
        from PIL import Image
        h_orig, w_orig = image_np.shape[:2]
        if image_size is not None:
            pil_img = Image.fromarray(image_np).resize((image_size, image_size))
        else:
            pil_img = Image.fromarray(image_np)
        input_tensor = TF.to_tensor(pil_img).unsqueeze(0)
        input_tensor = TF.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        activations = None
        gradients = None
        def forward_hook(m, inp, out):
            nonlocal activations
            activations = out.detach()
        def backward_hook(m, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)
        try:
            model.zero_grad()
            output = model(input_tensor)
            if target_class is None:
                target_class = output.argmax(1).item()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)
            if activations is None or gradients is None:
                return None
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1).squeeze(0)
            cam = F.relu(cam)
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            cam = cam.cpu().numpy()
        finally:
            fh.remove()
            bh.remove()
        cam_resized = cv2.resize(cam, (w_orig, h_orig))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_np, 0.4, heatmap, 0.6, 0)
        return overlay
    except Exception:
        import traceback
        traceback.print_exc()
        return None
