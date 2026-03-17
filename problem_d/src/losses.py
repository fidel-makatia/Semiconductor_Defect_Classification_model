"""Loss functions for IR drop prediction.

Aligned with contest metrics:
  - MAE is 60% of score -> use SmoothL1 / L1 for regression
  - F1 is 30% of score  -> use BCE with correct hotspot threshold
  - Hotspot-weighted regression to prevent peak smoothing (critical for F1)

NOTE: Target values are in millivolt range (~1e-4 to 6e-3).
SmoothL1Loss beta must be scaled accordingly, otherwise the loss
degenerates to MSE with vanishing gradients at this scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricL1Loss(nn.Module):
    """L1 loss with heavier penalty for underestimation."""

    def __init__(self, lambda_under: float = 2.0):
        super().__init__()
        self.lambda_under = lambda_under

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        weight = torch.where(diff < 0, self.lambda_under, 1.0)
        return (weight * abs_diff).mean()


class ContestAlignedLoss(nn.Module):
    """Combined regression + hotspot-weighted regression + BCE classification.

    Three stable components:
      1. SmoothL1 regression over all pixels (drives MAE down)
      2. Hotspot-weighted L1 on top-10% pixels with asymmetric penalty
         (penalizes underestimation 2x — prevents peak smoothing for F1)
      3. BCE hotspot classification using the actual contest threshold
         (target-based logits, not pred-based, for proper gradients)
    """

    def __init__(
        self,
        lambda_under: float = 1.5,
        alpha: float = 0.3,
        hotspot_weight: float = 5.0,
        use_smooth_l1: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.hotspot_weight = hotspot_weight
        self.use_smooth_l1 = use_smooth_l1
        if use_smooth_l1:
            # beta=1e-4 matches the data scale (~1e-4 to 6e-3 V)
            self.regression_loss = nn.SmoothL1Loss(beta=1e-4)
        else:
            self.regression_loss = AsymmetricL1Loss(lambda_under)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]

        # 1. Base regression loss over all pixels (drives MAE)
        reg_loss = self.regression_loss(pred, target)

        # Compute contest hotspot mask: top 10% pixels by target value
        with torch.no_grad():
            target_flat = target.view(batch_size, -1)
            target_max = target_flat.max(dim=1).values
            thresholds = (0.9 * target_max).view(-1, 1, 1, 1)  # [B,1,1,1]
            hotspot_mask = (target >= thresholds).float()

        # 2. Hotspot-weighted regression: extra asymmetric L1 on hotspot pixels
        # Distributes gradient across many hotspot pixels (stable, unlike
        # single-pixel peak loss). Underestimation gets 2x penalty.
        hotspot_diff = hotspot_mask * (pred - target)
        hotspot_penalty = torch.where(
            hotspot_diff < 0, -2.0 * hotspot_diff, hotspot_diff.abs()
        )
        n_hotspot = hotspot_mask.sum().clamp(min=1)
        hotspot_loss = hotspot_penalty.sum() / n_hotspot

        if self.alpha <= 0:
            return reg_loss + self.hotspot_weight * hotspot_loss

        # 3. BCE hotspot classification using the ACTUAL contest threshold
        # Uses target-based threshold for logits so model must match
        # absolute peak values, not just relative ordering.
        hotspot_labels = hotspot_mask
        threshold_band = (0.1 * target_max).view(-1, 1, 1, 1).clamp(min=1e-8)
        logits = (pred - thresholds) / threshold_band * 5.0

        cls_loss = F.binary_cross_entropy_with_logits(logits, hotspot_labels)

        return reg_loss + self.hotspot_weight * hotspot_loss + self.alpha * cls_loss


# Keep backward compatibility alias
CombinedIRDropLoss = ContestAlignedLoss
