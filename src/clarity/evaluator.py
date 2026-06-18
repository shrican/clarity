"""
Evaluation utilities for ClarityNet.

Metrics: accuracy, precision, recall, F1, AUC-ROC, confusion matrix.
Supports TTA (test-time augmentation) ensembling.
Threshold search for optimal F1.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from tqdm import tqdm

from clarity.config import Config
from clarity.model import ClarityNet
from clarity.utils import get_device, load_checkpoint

log = logging.getLogger("clarity")

LABEL_NAMES = {0: "sharp", 1: "blurry"}


def compute_metrics(
    preds: list[int],
    labels: list[int],
    probs: list[float] | None = None,
) -> dict:
    preds_np = np.array(preds)
    labels_np = np.array(labels)

    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "f1": float(f1_score(labels_np, preds_np, average="binary", zero_division=0)),
        "precision": float(precision_score(labels_np, preds_np, average="binary", zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, average="binary", zero_division=0)),
    }

    if probs is not None and len(set(labels_np)) == 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(labels_np, probs))
        except Exception:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    return metrics


def _optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing F1 over the ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_f1, best_thresh = 0.0, 0.5
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh, best_f1


class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.inference.device)

        self.model = ClarityNet(
            backbone=cfg.model.backbone,
            pretrained=False,
            num_classes=cfg.model.num_classes,
            dropout=0.0,  # no dropout at eval time
            freq_branch=cfg.model.freq_branch,
            cbam_reduction=cfg.model.cbam_reduction,
        ).to(self.device)

    def load_best(self) -> dict:
        ckpt = load_checkpoint(self.cfg.inference.checkpoint, self.model, self.device)
        log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} | val metrics: {ckpt['metrics']}")
        return ckpt

    @torch.no_grad()
    def _run_loader(self, loader) -> tuple[list, list, list]:
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []

        for batch in tqdm(loader, desc="evaluating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].tolist()

            out = self.model(images)
            probs = F.softmax(out["logits"], dim=1)[:, 1]  # P(blurry)
            preds = (probs >= self.cfg.eval.threshold).long()

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels)

        return all_probs, all_preds, all_labels

    def evaluate(self, loader, use_tta: bool = False, tta_loaders: list | None = None) -> dict:
        if use_tta and tta_loaders:
            all_prob_arrays = []
            for tta_loader in tta_loaders:
                probs, _, labels = self._run_loader(tta_loader)
                all_prob_arrays.append(probs)
            probs = np.mean(all_prob_arrays, axis=0).tolist()
            labels_arr = np.array(labels)
        else:
            probs, _, labels = self._run_loader(loader)
            labels_arr = np.array(labels)

        probs_arr = np.array(probs)
        best_thresh, _ = _optimal_threshold(labels_arr, probs_arr)
        log.info(f"Optimal threshold: {best_thresh:.3f}")

        preds = (probs_arr >= best_thresh).astype(int).tolist()
        metrics = compute_metrics(preds, labels, probs)
        metrics["optimal_threshold"] = best_thresh

        cm = confusion_matrix(labels_arr, preds).tolist()
        metrics["confusion_matrix"] = cm

        log.info(
            f"Test metrics | acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc_roc']:.4f} | threshold={best_thresh:.3f}"
        )
        log.info(f"Confusion matrix:\n  TN={cm[0][0]} FP={cm[0][1]}\n  FN={cm[1][0]} TP={cm[1][1]}")

        Path("logs").mkdir(exist_ok=True)
        with open("logs/eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics
