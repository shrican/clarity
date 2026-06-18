"""
Augmentation pipelines for blur detection training.

Sharp → random crop/flip/color jitter (no blur)
Blurry → all of the above + optional additional blur for hard examples
Synthetic blur generation for data augmentation and hard-negative mining.
"""
from __future__ import annotations
import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """Generate a directional motion blur kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    cx = size // 2
    kernel[cx, :] = 1.0
    M = cv2.getRotationMatrix2D((cx, cx), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    return kernel / kernel.sum()


def apply_synthetic_blur(image: np.ndarray, blur_type: str | None = None) -> tuple[np.ndarray, str]:
    """Apply one of gaussian/motion/defocus blur to an ndarray image (H,W,C uint8)."""
    if blur_type is None:
        blur_type = random.choice(["gaussian", "motion", "defocus"])

    if blur_type == "gaussian":
        sigma = random.uniform(2.0, 8.0)
        ksize = int(sigma * 4) | 1  # ensure odd
        result = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    elif blur_type == "motion":
        size = random.choice([7, 11, 15, 21, 31])
        angle = random.uniform(0, 180)
        kernel = _motion_blur_kernel(size, angle)
        result = cv2.filter2D(image, -1, kernel)

    else:  # defocus / lens blur
        radius = random.choice([3, 5, 7, 9, 13])
        kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
        kernel /= kernel.sum()
        result = cv2.filter2D(image, -1, kernel)

    return result, blur_type


def get_train_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        A.RandomRotate90(p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(int(image_size * 1.14), int(image_size * 1.14)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 224) -> list[A.Compose]:
    """Five TTA variants: center crop + four corner crops with hflip."""
    pad = image_size // 8  # ~28px padding for 224
    padded = image_size + pad

    def _make(crop, hflip=False):
        ops = [A.Resize(padded, padded), crop]
        if hflip:
            ops.append(A.HorizontalFlip(p=1.0))
        ops += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        return A.Compose(ops)

    return [
        _make(A.CenterCrop(image_size, image_size)),
        _make(A.Crop(0, 0, image_size, image_size)),
        _make(A.Crop(pad, 0, padded, image_size)),
        _make(A.Crop(0, pad, image_size, padded)),
        _make(A.CenterCrop(image_size, image_size), hflip=True),
    ]
