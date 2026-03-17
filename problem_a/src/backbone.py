"""Feature extraction backbones for few-shot defect classification.

Provides DINOv2 ViT-S/B/L/14 via timm, adapted for single-channel
grayscale input. Supports gradient checkpointing for memory efficiency.
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class GrayscaleTo3Channel(nn.Module):
    """Replicate single-channel grayscale to 3 channels for pretrained models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT feature extractor.

    Supports ViT-S/14 (384-dim), ViT-B/14 (768-dim), and ViT-L/14 (1024-dim).
    Self-supervised features that transfer well to industrial defect tasks.
    """

    MODELS = {
        "small": ("vit_small_patch14_dinov2.lvd142m", 384),
        "base": ("vit_base_patch14_dinov2.lvd142m", 768),
        "large": ("vit_large_patch14_dinov2.lvd142m", 1024),
    }

    def __init__(self, freeze: bool = True, unfreeze_last_n: int = 0,
                 size: str = "small", grad_checkpointing: bool = False):
        """
        Args:
            freeze: Whether to freeze the backbone weights.
            unfreeze_last_n: Number of final transformer blocks to unfreeze.
            size: "small" (384-dim, 22M), "base" (768-dim, 86M), or "large" (1024-dim, 304M).
            grad_checkpointing: Enable gradient checkpointing to reduce VRAM usage.
        """
        super().__init__()
        self.grayscale_adapter = GrayscaleTo3Channel()

        model_name, embed_dim = self.MODELS.get(size, self.MODELS["small"])

        if HAS_TIMM:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )
            self.embed_dim = self.backbone.embed_dim
        else:
            self.backbone = self._build_fallback()
            self.embed_dim = 384

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

            if unfreeze_last_n > 0 and HAS_TIMM:
                blocks = list(self.backbone.blocks)
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True
                # Also unfreeze final layer norm for better adaptation
                if hasattr(self.backbone, 'norm'):
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True

        if grad_checkpointing and HAS_TIMM:
            if hasattr(self.backbone, 'set_grad_checkpointing'):
                self.backbone.set_grad_checkpointing(True)

    def _build_fallback(self):
        """Simple CNN fallback when timm is not available."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 384),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grayscale_adapter(x)
        if HAS_TIMM:
            return self.backbone(x)  # [B, 384]
        else:
            return self.backbone(x)


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B4 feature extractor (lighter alternative).

    Output: 1792-dimensional embedding per image.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.grayscale_adapter = GrayscaleTo3Channel()

        if HAS_TIMM:
            self.backbone = timm.create_model(
                "efficientnet_b4",
                pretrained=True,
                num_classes=0,
            )
            self.embed_dim = self.backbone.num_features  # 1792
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 512),
            )
            self.embed_dim = 512

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grayscale_adapter(x)
        return self.backbone(x)


def get_backbone(name: str = "dinov2", size: str = "small", **kwargs) -> nn.Module:
    """Factory function to get a backbone by name.

    Args:
        name: 'dinov2' or 'efficientnet'.
        size: For dinov2: 'small' (384-dim), 'base' (768-dim), or 'large' (1024-dim).

    Returns:
        Backbone module with .embed_dim attribute.
    """
    if name == "dinov2":
        return DINOv2Backbone(size=size, **kwargs)
    elif name == "efficientnet":
        return EfficientNetBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose 'dinov2' or 'efficientnet'.")
