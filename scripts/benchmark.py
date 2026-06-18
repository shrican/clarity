#!/usr/bin/env python3
"""
ClarityNet v1 vs v2 benchmark.

Generates a controlled synthetic dataset and trains both model versions
under identical conditions for N epochs. Reports accuracy, F1, AUC-ROC,
and per-difficulty breakdown (easy / medium / hard blur).

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --epochs 25 --train-size 2000
"""
import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from clarity.model import ClarityNet
from clarity.losses import ClarityLoss
from clarity.augmentations import (
    get_train_transform, get_val_transform, apply_synthetic_blur
)
from clarity.dataset import BlurDetectionDataset, get_balanced_sampler
from clarity.utils import get_device, cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def _random_texture(H=256, W=256, seed=None) -> np.ndarray:
    """Generate a texture-rich image (better than uniform random for blur testing)."""
    rng = np.random.RandomState(seed)
    base = rng.randint(20, 235, (H, W, 3), dtype=np.uint8)
    # Add some structure: random rectangles
    for _ in range(rng.randint(3, 8)):
        y, x = rng.randint(0, H), rng.randint(0, W)
        h, w = rng.randint(10, 60), rng.randint(10, 60)
        color = rng.randint(0, 255, 3)
        base[y:y+h, x:x+w] = color
    return base


def build_synthetic_dataset(
    n_sharp: int,
    n_per_blur_type: dict[str, int],
    image_size: int = 128,
    seed_offset: int = 0,
) -> list[tuple[np.ndarray, int, str]]:
    """
    Returns list of (image_array, label, difficulty_tag).
    difficulty_tag: 'easy', 'medium', 'hard', or 'sharp'
    """
    import cv2
    records = []

    # Sharp images
    for i in range(n_sharp):
        img = _random_texture(image_size, image_size, seed=seed_offset + i)
        records.append((img, 0, "sharp"))

    # Blurry images at 3 difficulty levels
    rng = np.random.RandomState(seed_offset + 10000)
    idx = 0

    for diff, count in n_per_blur_type.items():
        for _ in range(count):
            img = _random_texture(image_size, image_size, seed=seed_offset + 20000 + idx)
            btype = rng.choice(["gaussian", "motion", "defocus"])

            if diff == "easy":      # heavy blur — clearly blurry
                if btype == "gaussian":
                    sigma = rng.uniform(5.0, 10.0)
                    ksize = int(sigma * 4) | 1
                    blurry = cv2.GaussianBlur(img, (ksize, ksize), sigma)
                elif btype == "motion":
                    import clarity.augmentations as aug_mod
                    from clarity.augmentations import _motion_blur_kernel
                    size = rng.choice([21, 31, 41])
                    angle = rng.uniform(0, 180)
                    kernel = _motion_blur_kernel(size, angle)
                    blurry = cv2.filter2D(img, -1, kernel)
                else:
                    radius = rng.choice([9, 13, 17])
                    kernel = np.zeros((radius*2+1, radius*2+1), dtype=np.float32)
                    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
                    kernel /= kernel.sum()
                    blurry = cv2.filter2D(img, -1, kernel)

            elif diff == "medium":   # moderate blur
                if btype == "gaussian":
                    sigma = rng.uniform(2.0, 5.0)
                    ksize = int(sigma * 4) | 1
                    blurry = cv2.GaussianBlur(img, (ksize, ksize), sigma)
                elif btype == "motion":
                    from clarity.augmentations import _motion_blur_kernel
                    size = rng.choice([11, 15, 21])
                    kernel = _motion_blur_kernel(size, rng.uniform(0, 180))
                    blurry = cv2.filter2D(img, -1, kernel)
                else:
                    radius = rng.choice([5, 7, 9])
                    kernel = np.zeros((radius*2+1, radius*2+1), dtype=np.float32)
                    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
                    kernel /= kernel.sum()
                    blurry = cv2.filter2D(img, -1, kernel)

            else:  # hard: slight blur, close to threshold
                if btype == "gaussian":
                    sigma = rng.uniform(0.7, 2.0)
                    ksize = max(int(sigma * 4) | 1, 3)
                    blurry = cv2.GaussianBlur(img, (ksize, ksize), sigma)
                elif btype == "motion":
                    from clarity.augmentations import _motion_blur_kernel
                    size = rng.choice([3, 5, 7])
                    kernel = _motion_blur_kernel(size, rng.uniform(0, 180))
                    blurry = cv2.filter2D(img, -1, kernel)
                else:
                    radius = rng.choice([1, 2, 3])
                    kernel = np.zeros((radius*2+1, radius*2+1), dtype=np.float32)
                    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
                    kernel /= kernel.sum()
                    blurry = cv2.filter2D(img, -1, kernel)

            records.append((blurry, 1, diff))
            idx += 1

    return records


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_model(
    model: ClarityNet,
    train_records,
    val_records,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_cutblur: bool,
    balanced: bool,
    image_size: int = 224,
) -> list[dict]:
    """Train a model and return per-epoch val metrics."""
    train_tf = get_train_transform(image_size)
    val_tf = get_val_transform(image_size)

    from clarity.dataset import apply_cutblur as _cb
    train_ds = BlurDetectionDataset(
        [(img, lbl) for img, lbl, _ in train_records],
        train_tf,
        hard_negative_prob=0.0 if use_cutblur else 0.10,
        cutblur_prob=0.20 if use_cutblur else 0.0,
    )
    val_ds = BlurDetectionDataset(
        [(img, lbl) for img, lbl, _ in val_records],
        val_tf,
    )

    sampler = get_balanced_sampler([(img, lbl) for img, lbl, _ in train_records]) if balanced else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    criterion = ClarityLoss(focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    warmup = max(1, epochs // 10) * len(train_loader)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(imgs)
            loss, _ = criterion(out["logits"], out["blur_score"], labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].tolist()
                out = model(imgs)
                probs = F.softmax(out["logits"], dim=1)[:, 1]
                preds = (probs >= 0.5).long().cpu().tolist()
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds)
                all_labels.extend(labels)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        history.append({"epoch": epoch, "acc": acc, "f1": f1, "auc": auc})

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    return history


def evaluate_by_difficulty(
    model: ClarityNet,
    test_records,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 64,
) -> dict:
    """Compute accuracy per difficulty tier."""
    val_tf = get_val_transform(image_size)
    model.eval()

    by_diff: dict[str, list] = {}
    for img, lbl, diff in test_records:
        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append((img, lbl))

    results = {}
    for diff, samples in by_diff.items():
        ds = BlurDetectionDataset(samples, val_tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(batch["image"].to(device))
                preds.extend(out["logits"].argmax(1).cpu().tolist())
                labels.extend(batch["label"].tolist())
        results[diff] = accuracy_score(labels, preds)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--train-size", type=int, default=2400,
                   help="Total training samples (split 50/50 sharp/blurry)")
    p.add_argument("--image-size", type=int, default=128,
                   help="Image size for benchmark (128 is fast; 224 is production)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device or "mps")
    print(f"\n{'='*60}")
    print(f"  ClarityNet v1 vs v2 Benchmark")
    print(f"  Device: {device} | Epochs: {args.epochs} | Image size: {args.image_size}px")
    print(f"{'='*60}\n")

    # Build datasets
    n_sharp = args.train_size // 2
    blurry_per_type = args.train_size // 6
    n_each = {"easy": blurry_per_type * 2, "medium": blurry_per_type * 2, "hard": blurry_per_type * 2}

    print(f"Building synthetic dataset: {args.train_size} train samples...")
    train_rec = build_synthetic_dataset(n_sharp, n_each, args.image_size, seed_offset=0)
    val_rec   = build_synthetic_dataset(200, {"easy": 60, "medium": 60, "hard": 80}, args.image_size, seed_offset=99999)
    test_rec  = build_synthetic_dataset(200, {"easy": 60, "medium": 60, "hard": 80}, args.image_size, seed_offset=199999)

    print(f"  Train: {len(train_rec)} | Val: {len(val_rec)} | Test: {len(test_rec)}\n")

    results = {}

    for version, label, use_cutblur, balanced in [
        ("v1", "Baseline (v1): Single Laplacian + CBAM", False, False),
        ("v2", "Improved (v2): MultiScale + FFT + CoordAttn + CutBlur + BalancedSampler", True, True),
    ]:
        print(f"\n{'─'*55}")
        print(f"  {label}")
        print(f"{'─'*55}")

        model = ClarityNet(
            backbone="convnextv2_tiny",
            pretrained=False,    # no pretrain so comparison is purely architecture
            num_classes=2,
            dropout=0.2,
            freq_branch=True,
            model_version=version,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {n_params:.2f}M")

        t0 = time.time()
        history = train_model(
            model, train_rec, val_rec,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=3e-4,
            device=device,
            use_cutblur=use_cutblur,
            balanced=balanced,
            image_size=args.image_size,
        )
        elapsed = time.time() - t0

        # Final test evaluation
        val_tf = get_val_transform(args.image_size)
        test_ds = BlurDetectionDataset(
            [(img, lbl) for img, lbl, _ in test_rec], val_tf
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch["image"].to(device))
                probs = F.softmax(out["logits"], dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend((probs >= 0.5).long().cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1  = f1_score(all_labels, all_preds, zero_division=0)
        test_auc = roc_auc_score(all_labels, all_probs)

        diff_acc = evaluate_by_difficulty(model, test_rec, device, args.image_size)
        best_val_f1 = max(h["f1"] for h in history)

        results[version] = {
            "test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc,
            "best_val_f1": best_val_f1,
            "diff_acc": diff_acc,
            "train_time_s": elapsed,
            "n_params_M": n_params,
        }

    # Print comparison table
    v1, v2 = results["v1"], results["v2"]
    print(f"\n\n{'='*60}")
    print(f"  RESULTS: v1 (Baseline)  vs  v2 (Improved)")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'v1':>10} {'v2':>10} {'Δ':>8}")
    print(f"  {'─'*56}")

    metric_keys = [
        ("Test Accuracy",     "test_acc"),
        ("Test F1",           "test_f1"),
        ("Test AUC-ROC",      "test_auc"),
        ("Best Val F1",       "best_val_f1"),
    ]
    for name, key in metric_keys:
        v1v = v1[key]
        v2v = v2[key]
        delta = v2v - v1v
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<28} {v1v:>10.4f} {v2v:>10.4f} {sign}{delta:.4f}")

    print(f"\n  {'Difficulty Breakdown':<28} {'v1':>10} {'v2':>10} {'Δ':>8}")
    print(f"  {'─'*56}")
    for diff in ["sharp", "easy", "medium", "hard"]:
        v1v = v1["diff_acc"].get(diff, 0)
        v2v = v2["diff_acc"].get(diff, 0)
        delta = v2v - v1v
        sign = "+" if delta >= 0 else ""
        print(f"  {diff.capitalize():<28} {v1v:.4f}     {v2v:.4f}  {sign}{delta:.4f}")

    print(f"\n  {'Parameters':<28} {v1['n_params_M']:.2f}M      {v2['n_params_M']:.2f}M")
    print(f"  {'Train time':<28} {v1['train_time_s']:.0f}s         {v2['train_time_s']:.0f}s")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
