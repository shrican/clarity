"""
Training loop for ClarityNet on MPS (Apple Silicon) / CUDA / CPU.

Features:
- Progressive resizing: starts at 128px, escalates to final size
- Mixup augmentation per batch
- Focal loss + auxiliary regression
- Cosine LR schedule with linear warmup
- Gradient clipping
- Best-model checkpointing by val F1
- Structured per-epoch logging
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from clarity.config import Config
from clarity.dataset import BlurDataModule
from clarity.model import ClarityNet
from clarity.losses import ClarityLoss
from clarity.utils import (
    get_device, mixup_batch, mixup_criterion,
    cosine_schedule_with_warmup, save_checkpoint, setup_logging,
)
from clarity.evaluator import compute_metrics

log = logging.getLogger("clarity")


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.inference.device)
        log.info(f"Training device: {self.device}")

        self.data = BlurDataModule(cfg)
        self.model = ClarityNet(
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            freq_branch=cfg.model.freq_branch,
            cbam_reduction=cfg.model.cbam_reduction,
        ).to(self.device)

        self.criterion = ClarityLoss(
            focal_alpha=cfg.training.focal_alpha,
            focal_gamma=cfg.training.focal_gamma,
            label_smoothing=cfg.training.label_smoothing,
            aux_weight=cfg.training.aux_loss_weight,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Compute total steps after we know the data size
        train_loader = self.data.train_loader()
        total_steps = cfg.training.epochs * len(train_loader)
        warmup_steps = cfg.training.warmup_epochs * len(train_loader)

        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        Path("checkpoints").mkdir(exist_ok=True)
        self.best_f1 = 0.0

    def _run_epoch(self, loader, training: bool) -> dict:
        self.model.train(training)
        ctx = torch.enable_grad() if training else torch.no_grad()
        total_loss = 0.0
        all_preds, all_labels = [], []
        n_batches = 0

        with ctx:
            for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if training and self.cfg.training.mixup_alpha > 0:
                    images, labels_a, labels_b, lam = mixup_batch(
                        images, labels, self.cfg.training.mixup_alpha
                    )
                    out = self.model(images)
                    loss, info = mixup_criterion(
                        self.criterion, out["logits"], out["blur_score"],
                        labels_a, labels_b, lam
                    )
                    # Use original labels for metrics
                    pred_labels = out["logits"].argmax(dim=1)
                    all_preds.extend(pred_labels.cpu().tolist())
                    all_labels.extend(labels_a.cpu().tolist())
                else:
                    out = self.model(images)
                    loss, info = self.criterion(out["logits"], out["blur_score"], labels)
                    pred_labels = out["logits"].argmax(dim=1)
                    all_preds.extend(pred_labels.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                total_loss += info["loss"]
                n_batches += 1

        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def train(self) -> dict:
        setup_logging()
        sizes = self.cfg.training.progressive_sizes
        epochs = self.cfg.training.epochs
        switch_points = [int(epochs * i / len(sizes)) for i in range(len(sizes))]

        history = []
        for epoch in range(1, epochs + 1):
            # Progressive resizing: increase image size over training
            stage = sum(1 for sp in switch_points if epoch >= sp) - 1
            stage = max(0, min(stage, len(sizes) - 1))
            current_size = sizes[stage]
            self.data.set_image_size(current_size)

            t0 = time.time()
            train_metrics = self._run_epoch(self.data.train_loader(), training=True)
            val_metrics = self._run_epoch(self.data.val_loader(), training=False)
            elapsed = time.time() - t0

            lr = self.scheduler.get_last_lr()[0]
            log.info(
                f"Epoch {epoch:03d}/{epochs} | size={current_size} | "
                f"lr={lr:.2e} | "
                f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                f"f1={val_metrics['f1']:.4f} auc={val_metrics['auc_roc']:.4f} | "
                f"{elapsed:.0f}s"
            )

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                save_checkpoint(
                    "checkpoints/best.pt",
                    self.model, self.optimizer, epoch,
                    val_metrics, self.cfg.to_dict(),
                )
                log.info(f"  ✓ New best val F1: {self.best_f1:.4f} — checkpoint saved")

        return {"history": history, "best_val_f1": self.best_f1}
