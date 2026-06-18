from __future__ import annotations
import time
import json
import logging
from pathlib import Path
import torch
import numpy as np


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation. Returns (mixed_images, labels_a, labels_b, lam)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B = images.size(0)
    perm = torch.randperm(B, device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def mixup_criterion(criterion, logits, blur_score, labels_a, labels_b, lam):
    loss_a, info_a = criterion(logits, blur_score, labels_a)
    loss_b, info_b = criterion(logits, blur_score, labels_b)
    loss = lam * loss_a + (1 - lam) * loss_b
    info = {k: lam * info_a[k] + (1 - lam) * info_b[k] for k in info_a}
    return loss, info


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg_dict: dict,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": cfg_dict,
    }, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/train.log"),
        ],
    )
    return logging.getLogger("clarity")
