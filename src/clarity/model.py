"""
ClarityNet: Dual-branch CNN for blur/focus detection.

Architecture:
  Branch 1 (semantic): ConvNeXt-Tiny pretrained backbone → global avg pool → features
  Branch 2 (frequency): Laplacian gradient map → lightweight CNN → features
  Fusion: concatenate → CBAM attention → dual head (classify + regress blur score)

The frequency branch encodes high-frequency sharpness cues that the semantic
backbone may discard. ConvNeXt uses depthwise convolutions that downsample
quickly; the Laplacian branch preserves fine-grained edge/gradient info.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Frequency / Gradient Feature Branch
# ---------------------------------------------------------------------------

class LaplacianLayer(nn.Module):
    """Fixed Laplacian kernel for edge/sharpness extraction."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        # Apply to each channel independently (groups=C)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  — normalize to [0,1] before applying
        B, C, H, W = x.shape
        x_gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        lap = F.conv2d(x_gray, self.kernel, padding=1)
        lap = lap.abs()
        lap = (lap - lap.flatten(1).min(dim=1).values[:, None, None, None]) / (
            lap.flatten(1).max(dim=1).values[:, None, None, None]
            - lap.flatten(1).min(dim=1).values[:, None, None, None]
            + 1e-8
        )
        return lap.expand(B, C, H, W)  # (B,3,H,W)


class FrequencyBranch(nn.Module):
    """
    Lightweight CNN operating on Laplacian gradient maps.
    4 conv blocks → global average pool → 128-d feature vector.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.laplacian = LaplacianLayer()
        self.encoder = nn.Sequential(
            self._block(3, 32, stride=2),
            self._block(32, 64, stride=2),
            self._block(64, 128, stride=2),
            self._block(128, out_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap = self.laplacian(x)
        feats = self.encoder(lap)
        return self.pool(feats).flatten(1)


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        avg_w = self.mlp(self.avg(x.unsqueeze(-1)).squeeze(-1))
        max_w = self.mlp(self.max(x.unsqueeze(-1)).squeeze(-1))
        return torch.sigmoid(avg_w + max_w)


class FusionCBAM(nn.Module):
    """Channel attention over the fused feature vector."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.attn = ChannelAttention(dim, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)
        return x * w


# ---------------------------------------------------------------------------
# Variance-of-Laplacian as a deterministic sharpness feature
# ---------------------------------------------------------------------------

def laplacian_variance_feature(x: torch.Tensor) -> torch.Tensor:
    """
    Classic no-reference sharpness score: variance of Laplacian.
    Returns (B, 1) tensor — serves as a strong inductive bias feature.
    """
    kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    gray = x.mean(dim=1, keepdim=True)
    lap = F.conv2d(gray, kernel, padding=1)
    var = lap.flatten(2).var(dim=2)  # (B, 1)
    return torch.log1p(var)           # log-scale to compress range


# ---------------------------------------------------------------------------
# ClarityNet
# ---------------------------------------------------------------------------

class ClarityNet(nn.Module):
    """
    Dual-branch blur/focus classifier.

    Output:
      logits: (B, 2)        — SHARP=0, BLURRY=1
      blur_score: (B, 1)    — continuous [0,1] sharpness score (auxiliary head)
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        freq_branch: bool = True,
        cbam_reduction: int = 16,
    ):
        super().__init__()
        self.freq_branch_enabled = freq_branch

        # --- Semantic backbone ---
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,      # remove classifier, keep feature extractor
            global_pool="avg",
        )
        bb_dim = self.backbone.num_features

        # --- Frequency branch ---
        freq_dim = 128 if freq_branch else 0
        if freq_branch:
            self.freq_branch = FrequencyBranch(out_dim=freq_dim)

        # --- Fusion ---
        fused_dim = bb_dim + freq_dim + 1   # +1 for Laplacian variance feature
        self.cbam = FusionCBAM(fused_dim, cbam_reduction)
        self.drop = nn.Dropout(dropout)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # --- Auxiliary regression head (blur score 0→sharp, 1→blurry) ---
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bb_feats = self.backbone(x)                  # (B, bb_dim)

        parts = [bb_feats]
        if self.freq_branch_enabled:
            parts.append(self.freq_branch(x))        # (B, freq_dim)

        lap_var = laplacian_variance_feature(x)      # (B, 1)
        parts.append(lap_var)

        fused = torch.cat(parts, dim=1)              # (B, fused_dim)
        fused = self.cbam(fused)
        fused = self.drop(fused)

        logits = self.classifier(fused)
        blur_score = self.regressor(fused)

        return {"logits": logits, "blur_score": blur_score}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience inference method."""
        out = self.forward(x)
        probs = F.softmax(out["logits"], dim=1)
        pred = probs.argmax(dim=1)
        return {
            "label": pred,                       # 0=sharp, 1=blurry
            "confidence": probs.max(dim=1).values,
            "sharp_prob": probs[:, 0],
            "blurry_prob": probs[:, 1],
            "blur_score": out["blur_score"].squeeze(1),
        }
