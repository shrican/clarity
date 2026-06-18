"""
Loss functions for ClarityNet.

- FocalLoss: handles class imbalance better than BCE, focuses on hard examples
- LabelSmoothingCE: reduces overconfidence
- ClarityLoss: weighted combination of focal CE + auxiliary MSE regression
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    α modulates class imbalance, γ down-weights easy examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        # Label smoothing: soft targets
        with torch.no_grad():
            soft_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Per-sample cross-entropy with soft targets
        ce = -(soft_targets * log_probs).sum(dim=1)

        # Focal weight: based on probability of the TRUE class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma

        # Alpha weighting: alpha for class 1, (1-alpha) for class 0
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        loss = alpha_t * focal_w * ce
        return loss.mean()


class ClarityLoss(nn.Module):
    """
    Combined loss for ClarityNet.
    L = FocalCE(logits, labels) + λ * MSE(blur_score, soft_label)
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        aux_weight: float = 0.3,
    ):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma, label_smoothing)
        self.aux_weight = aux_weight

    def forward(
        self,
        logits: torch.Tensor,
        blur_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = self.focal(logits, labels)

        # Auxiliary regression: soft target = label (0.0 sharp, 1.0 blurry)
        soft_labels = labels.float().unsqueeze(1)
        reg_loss = F.mse_loss(blur_score, soft_labels)

        total = cls_loss + self.aux_weight * reg_loss

        return total, {
            "loss": total.item(),
            "cls_loss": cls_loss.item(),
            "reg_loss": reg_loss.item(),
        }
