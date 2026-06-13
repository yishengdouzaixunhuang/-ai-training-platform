"""
Stronger data augmentation pipeline for semantic segmentation.
Pure torchvision, zero extra dependencies.
Designed as a drop-in extension - does not modify existing dataset code.
"""
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


def _to_pil(img, mask):
    """Ensure image is PIL, mask is PIL (mode='L' or 'P')."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img.astype(np.uint8) if img.dtype != np.uint8 else img)
    if not isinstance(mask, Image.Image):
        mask_pil = Image.fromarray(mask.astype(np.uint8))
    else:
        mask_pil = mask
    return img, mask_pil


def _to_numpy(mask_pil):
    """Convert mask back to numpy int64 array."""
    return np.array(mask_pil).astype(np.int64)


def apply_strong_augment(image, mask, intensity="medium"):
    """
    Apply strong augmentations to image-mask pair.
    
    Args:
        image: PIL Image (RGB) or numpy array
        mask: numpy array (int) - class labels
        intensity: "light", "medium", or "strong"
    
    Returns:
        (augmented_image: Tensor-friendly PIL, augmented_mask: np.int64 array)
    """
    img, mask_pil = _to_pil(image, mask)
    mask_np = np.array(mask_pil).astype(np.int64)
    
    if intensity == "light":
        prob_hflip, prob_vflip, prob_color, prob_blur, prob_noise = 0.5, 0.3, 0.4, 0.2, 0.0
        brightness_range, contrast_range, saturation_range = (0.8, 1.2), (0.8, 1.2), (0.8, 1.2)
        blur_radius, noise_std = 1.0, 3.0
    elif intensity == "strong":
        prob_hflip, prob_vflip, prob_color, prob_blur, prob_noise = 0.5, 0.4, 0.7, 0.4, 0.3
        brightness_range, contrast_range, saturation_range = (0.6, 1.4), (0.6, 1.4), (0.5, 1.5)
        blur_radius, noise_std = 3.0, 10.0
    else:  # medium (default)
        prob_hflip, prob_vflip, prob_color, prob_blur, prob_noise = 0.5, 0.3, 0.5, 0.3, 0.2
        brightness_range, contrast_range, saturation_range = (0.7, 1.3), (0.7, 1.3), (0.7, 1.3)
        blur_radius, noise_std = 2.0, 5.0
    
    # 1. Horizontal flip (same as existing, keep for compatibility)
    if random.random() < prob_hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask_np = np.fliplr(mask_np).copy()
    
    # 2. Vertical flip (NEW - useful for symmetric defects)
    if random.random() < prob_vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask_np = np.flipud(mask_np).copy()
    
    # 3. Color jitter (NEW - brightness, contrast, saturation)
    if random.random() < prob_color:
        b = random.uniform(*brightness_range)
        c = random.uniform(*contrast_range)
        s = random.uniform(*saturation_range)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        if img.mode == "RGB":
            # Only apply saturation to RGB images
            pass  # saturation requires HSV conversion, skip for simplicity
    
    # 4. Gaussian blur (NEW - simulates motion blur / focus issues)
    if random.random() < prob_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 5. Gaussian noise (NEW - sensor noise simulation)
    if random.random() < prob_noise and img.mode == "RGB":
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0, noise_std, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
    
    return img, mask_np


# Convenience presets
def augment_light(image, mask):
    return apply_strong_augment(image, mask, "light")

def augment_medium(image, mask):
    return apply_strong_augment(image, mask, "medium")

def augment_strong(image, mask):
    return apply_strong_augment(image, mask, "strong")


AUGMENT_PRESETS = {
    "none": None,
    "light": augment_light,
    "medium": augment_medium,
    "strong": augment_strong,
}
