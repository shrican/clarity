"""
Blur detection dataset pipeline — two-stage loading strategy:

Stage 1 (pretraining): wtcherr/unsplash_10k_blur_rand_KS
  - 'guide' column → PIL image (SHARP, label=0)
  - 'image' column → PIL image (BLURRY, label=1)
  - ~20k synthetic paired examples for robust base features

Stage 2 (fine-tuning, optional): chitradrishti/cuhk-blur + chitradrishti/Flickr-Blur
  - Real camera blur/sharp images with folder-based labels
  - Loaded if available; gracefully skipped on error (license/access issues)

Note: Mixup is NOT applied — it corrupts the high-frequency edge statistics
that are the primary discriminative signal for sharpness detection.
"""
from __future__ import annotations
import random
from typing import Literal
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import torch

from clarity.config import Config
from clarity.augmentations import (
    get_train_transform, get_val_transform, get_tta_transforms,
    apply_synthetic_blur,
)

LABEL_SHARP = 0
LABEL_BLURRY = 1


class BlurDetectionDataset(Dataset):
    """Wraps a list of (image_array, label) pairs with transforms."""

    def __init__(
        self,
        records: list[tuple[np.ndarray, int]],
        transform,
        hard_negative_prob: float = 0.0,
    ):
        self.records = records
        self.transform = transform
        self.hard_negative_prob = hard_negative_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        img_np, label = self.records[idx]

        # Hard-negative: take a sharp image and apply extra blur → still labeled blurry
        if label == LABEL_SHARP and random.random() < self.hard_negative_prob:
            img_np, _ = apply_synthetic_blur(img_np)
            label = LABEL_BLURRY

        augmented = self.transform(image=img_np)["image"]
        return {"image": augmented, "label": torch.tensor(label, dtype=torch.long)}


def _pil_to_np(pil: Image.Image) -> np.ndarray:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return np.array(pil, dtype=np.uint8)


def _load_pretrain_records(cfg: Config) -> list[tuple[np.ndarray, int]]:
    """
    Load Stage 1 dataset: wtcherr/unsplash paired images.
    guide=sharp (label 0), image=blurry (label 1).
    """
    print(f"[Stage 1] Loading pretrain dataset: {cfg.dataset.pretrain_dataset}")
    ds = load_dataset(cfg.dataset.pretrain_dataset, cache_dir=cfg.dataset.cache_dir, split="train")

    records: list[tuple[np.ndarray, int]] = []
    for sample in ds:
        if "guide" in sample and sample["guide"] is not None:
            records.append((_pil_to_np(sample["guide"]), LABEL_SHARP))
        if "image" in sample and sample["image"] is not None:
            records.append((_pil_to_np(sample["image"]), LABEL_BLURRY))

    print(f"  Loaded {len(records)} records (sharp: {sum(1 for _,l in records if l==0)}, "
          f"blurry: {sum(1 for _,l in records if l==1)})")
    return records


def _load_finetune_records(cfg: Config) -> list[tuple[np.ndarray, int]]:
    """
    Load Stage 2 datasets: CUHK-Blur + Flickr-Blur (real labeled images).
    These use imagefolder format where the folder name encodes the label.
    Silently skips datasets that fail (license/access/format issues).
    """
    records: list[tuple[np.ndarray, int]] = []
    for ds_name in cfg.dataset.finetune_datasets:
        try:
            print(f"[Stage 2] Loading fine-tune dataset: {ds_name}")
            ds = load_dataset(ds_name, cache_dir=cfg.dataset.cache_dir, split="train")
            sample = next(iter(ds))
            keys = set(sample.keys())

            if "label" in keys and "image" in keys:
                # Standard imagefolder format with integer labels
                label_names = ds.features["label"].names if hasattr(ds.features.get("label"), "names") else []
                for item in ds:
                    img = item["image"]
                    lbl = item["label"]
                    # Map label names to SHARP=0 / BLURRY=1
                    if label_names:
                        name = label_names[lbl].lower()
                        mapped = LABEL_BLURRY if any(w in name for w in ("blur", "blurry", "defocus", "motion")) else LABEL_SHARP
                    else:
                        mapped = int(lbl)
                    records.append((_pil_to_np(img), mapped))

            elif "image" in keys and len(keys) == 1:
                # Images only (Flickr-Blur): no labels available, skip
                print(f"  Skipping {ds_name}: no labels found (image-only dataset)")
                continue

            print(f"  Loaded {len(records)} total fine-tune records from {ds_name}")

        except Exception as exc:
            print(f"  Skipping {ds_name}: {exc}")

    return records


def _load_hf_records(cfg: Config) -> tuple[list, list, list]:
    """
    Load all records and split into (train, val, test).
    Stage 1 (synthetic) + Stage 2 (real, if available) are combined.
    """
    records = _load_pretrain_records(cfg)

    finetune = _load_finetune_records(cfg)
    if finetune:
        print(f"Adding {len(finetune)} real-label fine-tune records")
        records.extend(finetune)

    print(f"Total samples: {len(records)} | "
          f"sharp: {sum(1 for _,l in records if l==0)} | "
          f"blurry: {sum(1 for _,l in records if l==1)}")

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(records))
    records = [records[i] for i in indices]

    n_test = int(len(records) * cfg.dataset.test_split)
    n_val = int(len(records) * cfg.dataset.val_split)
    test = records[:n_test]
    val = records[n_test: n_test + n_val]
    train = records[n_test + n_val:]

    print(f"Split → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


class BlurDataModule:
    """Manages dataset loading and DataLoader creation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.train_records, self.val_records, self.test_records = _load_hf_records(cfg)

        self._train_transform = get_train_transform(cfg.dataset.image_size)
        self._val_transform = get_val_transform(cfg.dataset.image_size)
        self._tta_transforms = get_tta_transforms(cfg.dataset.image_size)

    def set_image_size(self, size: int) -> None:
        """Progressive resizing: swap transforms mid-training."""
        self._train_transform = get_train_transform(size)

    def train_loader(self) -> DataLoader:
        ds = BlurDetectionDataset(
            self.train_records,
            self._train_transform,
            hard_negative_prob=0.15,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=False,  # MPS doesn't benefit from pinned memory
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        ds = BlurDetectionDataset(self.val_records, self._val_transform)
        return DataLoader(
            ds,
            batch_size=self.cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
        )

    def test_loader(self) -> DataLoader:
        ds = BlurDetectionDataset(self.test_records, self._val_transform)
        return DataLoader(
            ds,
            batch_size=self.cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
        )

    def tta_loaders(self) -> list[DataLoader]:
        """One DataLoader per TTA transform variant."""
        return [
            DataLoader(
                BlurDetectionDataset(self.test_records, t),
                batch_size=self.cfg.training.batch_size * 2,
                shuffle=False,
                num_workers=self.cfg.dataset.num_workers,
            )
            for t in self._tta_transforms
        ]
