#!/usr/bin/env python3
"""
Hard-regime benchmark: shows where v2 actually beats v1.

Three tests:
  1. Small data (400 samples) — architecture matters when data is scarce
  2. Near-threshold blur (sigma 0.3-0.8) — distinguishing slight blur from sharp
  3. Texture-confounded hard set — soft textures + slight blur + sensor noise

Reports convergence speed (epochs to 90/95/99%) and final test metrics.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from clarity.model import ClarityNet
from clarity.losses import ClarityLoss
from clarity.augmentations import get_train_transform, get_val_transform, _motion_blur_kernel
from clarity.dataset import BlurDetectionDataset, get_balanced_sampler
from clarity.utils import get_device, cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Hard dataset generation
# ---------------------------------------------------------------------------

def _texture_image(H, W, seed, soft=False):
    """Realistic image: gradient background + edges + optional soft-texture look."""
    rng = np.random.RandomState(seed)
    # Smooth gradient base (looks soft even when sharp)
    base = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        xs = rng.uniform(50, 200)
        ys = rng.uniform(50, 200)
        x = np.linspace(xs, ys, W)
        y = np.linspace(rng.uniform(30, 180), rng.uniform(100, 230), H)
        base[:, :, c] = np.outer(y, np.ones(W)) * 0.5 + np.outer(np.ones(H), x) * 0.5

    if not soft:
        # Add crisp edges (sharp detail)
        for _ in range(rng.randint(5, 15)):
            y0, x0 = rng.randint(0, H), rng.randint(0, W)
            h, w = rng.randint(5, 30), rng.randint(5, 30)
            col = rng.uniform(0, 255, 3)
            base[y0:y0+h, x0:x0+w] = col
        # Fine texture
        noise = rng.uniform(-20, 20, (H, W, 3))
        base += noise
    else:
        # Soft texture: blurred gradients — naturally looks "soft" but IS sharp
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=1.5)

    return np.clip(base, 0, 255).astype(np.uint8)


def _add_sensor_noise(img, sigma=3.0, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build_hard_dataset(n_per_class, image_size=128, seed_offset=0, scenario="near_threshold"):
    """
    Scenarios:
      'near_threshold': sigma 0.3-0.8 blur — barely visible
      'texture_confounded': soft textures + noise + near-threshold blur
      'small_data': regular blur, just fewer samples
    """
    rng = np.random.RandomState(seed_offset)
    records = []

    for i in range(n_per_class):
        seed = seed_offset + i
        soft = (scenario == "texture_confounded") and (i % 3 == 0)
        img = _texture_image(image_size, image_size, seed, soft=soft)

        if scenario == "texture_confounded":
            img = _add_sensor_noise(img, sigma=rng.uniform(2, 6), seed=seed)
        records.append((img, 0, "sharp"))

    for i in range(n_per_class):
        seed = seed_offset + 50000 + i
        soft = (scenario == "texture_confounded") and (i % 3 == 0)
        img = _texture_image(image_size, image_size, seed, soft=soft)

        if scenario == "texture_confounded":
            img = _add_sensor_noise(img, sigma=rng.uniform(2, 6), seed=seed)

        btype = rng.choice(["gaussian", "motion", "defocus"])
        if scenario in ("near_threshold", "texture_confounded"):
            # Very slight blur
            if btype == "gaussian":
                sigma = rng.uniform(0.6, 1.8)
                ksize = max(int(sigma * 4) | 1, 3)
                img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            elif btype == "motion":
                size = rng.choice([3, 5])
                kernel = _motion_blur_kernel(size, rng.uniform(0, 180))
                img = cv2.filter2D(img, -1, kernel)
            else:
                radius = rng.choice([1, 2])
                kernel = np.zeros((radius*2+1, radius*2+1), dtype=np.float32)
                cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
                kernel /= kernel.sum()
                img = cv2.filter2D(img, -1, kernel)
        else:
            # Mixed easy+hard for small_data
            severity = rng.choice(["mild", "hard"])
            if severity == "hard":
                sigma = rng.uniform(0.6, 2.5)
                ksize = max(int(sigma * 4) | 1, 3)
                img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            else:
                sigma = rng.uniform(2.5, 6.0)
                ksize = int(sigma * 4) | 1
                img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        records.append((img, 1, "blurry"))

    return records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_track(
    model, train_records, val_records, epochs, batch_size, lr, device, use_cutblur, balanced
):
    """Train and record val accuracy at every epoch. Return history."""
    train_tf = get_train_transform(128)
    val_tf = get_val_transform(128)
    train_ds = BlurDetectionDataset(
        [(img, lbl) for img, lbl, _ in train_records],
        train_tf,
        cutblur_prob=0.20 if use_cutblur else 0.0,
        hard_negative_prob=0.10 if not use_cutblur else 0.0,
    )
    val_ds = BlurDetectionDataset([(img, lbl) for img, lbl, _ in val_records], val_tf)

    sampler = get_balanced_sampler([(img, lbl) for img, lbl, _ in train_records]) if balanced else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    criterion = ClarityLoss(focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, epochs // 8) * len(train_loader)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup, epochs * len(train_loader))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(imgs)
            loss, _ = criterion(out["logits"], out["blur_score"], labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        model.eval()
        probs_all, preds_all, labels_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["image"].to(device))
                p = F.softmax(out["logits"], dim=1)[:, 1]
                probs_all.extend(p.cpu().tolist())
                preds_all.extend((p >= 0.5).long().cpu().tolist())
                labels_all.extend(batch["label"].tolist())
        acc = accuracy_score(labels_all, preds_all)
        f1  = f1_score(labels_all, preds_all, zero_division=0)
        auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.5
        history.append({"epoch": epoch, "acc": acc, "f1": f1, "auc": auc})
    return history


def epochs_to_threshold(history, threshold):
    for h in history:
        if h["acc"] >= threshold:
            return h["epoch"]
    return None  # never reached


def make_model(version, device):
    return ClarityNet(
        backbone="convnextv2_tiny", pretrained=False,
        num_classes=2, dropout=0.2, freq_branch=True,
        model_version=version,
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scenario(name, scenario, n_train_per_class, n_val, n_test, epochs, device, batch_size=32):
    print(f"\n{'━'*62}")
    print(f"  Scenario: {name}")
    print(f"  n_train={n_train_per_class*2} | n_test={n_test*2} | epochs={epochs}")
    print(f"{'━'*62}")

    train_rec = build_hard_dataset(n_train_per_class, 128, seed_offset=0, scenario=scenario)
    val_rec   = build_hard_dataset(n_val,             128, seed_offset=88888, scenario=scenario)
    test_rec  = build_hard_dataset(n_test,            128, seed_offset=77777, scenario=scenario)

    results = {}
    for version, use_cutblur, balanced in [("v1", False, False), ("v2", True, True)]:
        model = make_model(version, device)
        t0 = time.time()
        hist = train_and_track(model, train_rec, val_rec, epochs, batch_size,
                               lr=3e-4, device=device, use_cutblur=use_cutblur, balanced=balanced)
        elapsed = time.time() - t0

        # Final test
        val_tf = get_val_transform(128)
        test_ds = BlurDetectionDataset([(img, lbl) for img, lbl, _ in test_rec], val_tf)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        model.eval()
        probs_all, preds_all, labels_all = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch["image"].to(device))
                p = F.softmax(out["logits"], dim=1)[:, 1]
                probs_all.extend(p.cpu().tolist())
                preds_all.extend((p >= 0.5).long().cpu().tolist())
                labels_all.extend(batch["label"].tolist())

        results[version] = {
            "history": hist,
            "test_acc": accuracy_score(labels_all, preds_all),
            "test_f1":  f1_score(labels_all, preds_all, zero_division=0),
            "test_auc": roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.5,
            "time_s":   elapsed,
        }
        best = max(h["acc"] for h in hist)
        e90 = epochs_to_threshold(hist, 0.90)
        e95 = epochs_to_threshold(hist, 0.95)
        e99 = epochs_to_threshold(hist, 0.99)
        print(f"  {version}: test_acc={results[version]['test_acc']:.4f}  f1={results[version]['test_f1']:.4f}  "
              f"auc={results[version]['test_auc']:.4f}  "
              f"→90%:{e90 or 'n/a'}  →95%:{e95 or 'n/a'}  →99%:{e99 or 'n/a'}  "
              f"({elapsed:.0f}s)")

    v1, v2 = results["v1"], results["v2"]
    print(f"\n  {'Metric':<30} {'v1':>8} {'v2':>8} {'Δ':>8}")
    print(f"  {'─'*54}")
    for metric, key in [("Test Accuracy", "test_acc"), ("Test F1", "test_f1"), ("Test AUC-ROC", "test_auc")]:
        d = v2[key] - v1[key]
        print(f"  {metric:<30} {v1[key]:>8.4f} {v2[key]:>8.4f} {'+' if d>=0 else ''}{d:>7.4f}")

    for thresh, label in [(0.90, "Epochs to 90% acc"), (0.95, "Epochs to 95% acc"), (0.99, "Epochs to 99% acc")]:
        e1 = epochs_to_threshold(v1["history"], thresh)
        e2 = epochs_to_threshold(v2["history"], thresh)
        s1 = str(e1) if e1 else "never"
        s2 = str(e2) if e2 else "never"
        gain = f"{e1 - e2} faster" if e1 and e2 and e1 > e2 else ("tie" if e1 == e2 else "v1 faster" if e1 and e2 else "")
        print(f"  {label:<30} {s1:>8} {s2:>8}  {gain}")

    return results


def main():
    device = get_device("mps")
    print(f"\n{'='*62}")
    print(f"  ClarityNet Hard-Regime Benchmark  |  {device}")
    print(f"{'='*62}")

    all_results = {}

    # Scenario 1: Near-threshold blur — barely visible
    all_results["near_threshold"] = run_scenario(
        "Near-threshold blur (sigma 0.6-1.8)",
        scenario="near_threshold",
        n_train_per_class=600, n_val=150, n_test=200,
        epochs=30, device=device,
    )

    # Scenario 2: Small data — 300 total samples
    all_results["small_data"] = run_scenario(
        "Small data (300 train)",
        scenario="small_data",
        n_train_per_class=150, n_val=100, n_test=200,
        epochs=30, device=device,
    )

    # Scenario 3: Texture-confounded — soft textures + sensor noise + near-threshold blur
    all_results["texture_confounded"] = run_scenario(
        "Texture-confounded (soft texture + noise)",
        scenario="texture_confounded",
        n_train_per_class=600, n_val=150, n_test=200,
        epochs=30, device=device,
    )

    # Summary
    print(f"\n\n{'='*62}")
    print(f"  FINAL SUMMARY")
    print(f"  {'Scenario':<32} {'Δ Acc':>8} {'Δ F1':>8} {'Δ AUC':>8}")
    print(f"  {'─'*58}")
    for sname, sresults in all_results.items():
        v1, v2 = sresults["v1"], sresults["v2"]
        da = v2["test_acc"] - v1["test_acc"]
        df = v2["test_f1"]  - v1["test_f1"]
        du = v2["test_auc"] - v1["test_auc"]
        label = {"near_threshold": "Near-threshold blur",
                 "small_data":     "Small data (300)",
                 "texture_confounded": "Soft texture + noise"}[sname]
        print(f"  {label:<32} {'+' if da>=0 else ''}{da:>7.4f} {'+' if df>=0 else ''}{df:>7.4f} {'+' if du>=0 else ''}{du:>7.4f}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
