"""Attention U-Net for static IR drop prediction.

Architecture:
  Encoder: 4 levels with DoubleConv + MaxPool
  Bottleneck: DoubleConv
  Decoder: 4 levels with ConvTranspose2d + Attention Gates + skip connections
  Output: 1x1 Conv -> per-pixel voltage drop prediction

Handles arbitrary spatial dimensions via padding/cropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Handle size mismatch between decoder output and encoder skip
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net for image-to-image regression.

    Handles arbitrary spatial dimensions by padding input to be divisible
    by 16 (2^4 for 4 pooling levels), then cropping output back.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        f = base_filters
        filters = [f, f * 2, f * 4, f * 8, f * 16]

        # Encoder
        self.enc1 = DoubleConv(in_channels, filters[0], dropout)
        self.enc2 = DoubleConv(filters[0], filters[1], dropout)
        self.enc3 = DoubleConv(filters[1], filters[2], dropout)
        self.enc4 = DoubleConv(filters[2], filters[3], dropout)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(filters[3], filters[4], dropout)

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.att4 = AttentionGate(filters[3], filters[3], filters[2])
        self.dec4 = DoubleConv(filters[4], filters[3], dropout)

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.att3 = AttentionGate(filters[2], filters[2], filters[1])
        self.dec3 = DoubleConv(filters[3], filters[2], dropout)

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.att2 = AttentionGate(filters[1], filters[1], filters[0])
        self.dec2 = DoubleConv(filters[2], filters[1], dropout)

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.att1 = AttentionGate(filters[0], filters[0], filters[0] // 2)
        self.dec1 = DoubleConv(filters[1], filters[0], dropout)

        # Output
        self.final = nn.Conv2d(filters[0], out_channels, 1)

        # Number of pooling levels -> input must be divisible by 2^levels
        self._divisor = 16  # 2^4

    def _pad_to_divisible(self, x: torch.Tensor):
        """Pad input so H and W are divisible by self._divisor."""
        _, _, h, w = x.shape
        pad_h = (self._divisor - h % self._divisor) % self._divisor
        pad_w = (self._divisor - w % self._divisor) % self._divisor
        if pad_h > 0 or pad_w > 0:
            # Reflect padding: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, h, w

    def _match_and_cat(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Match spatial dims of upsampled tensor to skip connection, then concat."""
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad to make dimensions divisible by 16
        x, orig_h, orig_w = self._pad_to_divisible(x)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(self._match_and_cat(d4, e4))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(self._match_and_cat(d3, e3))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(self._match_and_cat(d2, e2))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(self._match_and_cat(d1, e1))

        out = self.final(d1)

        # Crop back to original dimensions
        out = out[:, :, :orig_h, :orig_w]
        return out


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check with non-square, non-divisible-by-16 input
    model = AttentionUNet(in_channels=3, out_channels=1)
    for shape in [(2, 3, 256, 256), (1, 3, 80, 120), (1, 3, 73, 97)]:
        x = torch.randn(*shape)
        y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
