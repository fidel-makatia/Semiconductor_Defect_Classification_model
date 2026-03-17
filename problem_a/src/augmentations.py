"""Image augmentation pipelines for grayscale defect images.

Handles images up to ~7000x5600 with aspect-ratio-preserving resize.
Uses albumentations for geometric and intensity transforms.
"""

import numpy as np
import torch

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

import torchvision.transforms as T


def get_train_transform(img_size: int = 518, use_albumentations: bool = True):
    """Training augmentation pipeline for grayscale defect images.

    Uses LongestMaxSize + PadIfNeeded to preserve aspect ratio.
    Aggressive augmentations for robust few-shot learning.
    """
    if use_albumentations and HAS_ALBUMENTATIONS:
        return AlbumentationsTransform(
            A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size,
                    border_mode=0,
                ),
                # Geometric
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    translate_percent=(-0.1, 0.1), scale=(0.85, 1.15),
                    rotate=(-45, 45), p=0.4,
                ),
                A.ElasticTransform(alpha=80, sigma=12, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
                # Intensity
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.4,
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                # Regularization
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(int(img_size * 0.05), int(img_size * 0.15)),
                    hole_width_range=(int(img_size * 0.05), int(img_size * 0.15)),
                    p=0.3,
                ),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        )
    else:
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])


def get_eval_transform(img_size: int = 518, use_albumentations: bool = True):
    """Evaluation transform with aspect-ratio-preserving resize."""
    if use_albumentations and HAS_ALBUMENTATIONS:
        return AlbumentationsTransform(
            A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size,
                    border_mode=0,
                ),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        )
    else:
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])


class AlbumentationsTransform:
    """Wrapper to make albumentations transforms work with PIL images."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if hasattr(img, "convert"):
            img = np.array(img)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        result = self.transform(image=img)
        return result["image"]
