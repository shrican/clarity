"""
Blur detection dataset pipeline.

Primary source: wtcherr/unsplash_10k_blur_rand_KS
  - 'guide' column → PIL image (SHARP, label=0)
  - 'image' column → PIL image (BLURRY, label=1)

Augmentation:
  - Hard-negative mining: apply extra synthetic blur to some sharp images
  - Mixup on the fly in the collate_fn
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


def _load_hf_records(cfg: Config) -> tuple[list, list, list]:
    """
    Download the HuggingFace dataset and return (train_records, val_records, test_records).
    Each record is (np.ndarray H×W×3, label).
    """
    print(f"Loading dataset: {cfg.dataset.name}")
    ds = load_dataset(cfg.dataset.name, cache_dir=cfg.dataset.cache_dir, split="train")

    records: list[tuple[np.ndarray, int]] = []

    for sample in ds:
        # 'guide' = original sharp, 'image' = blurred version
        if "guide" in sample and sample["guide"] is not None:
            records.append((_pil_to_np(sample["guide"]), LABEL_SHARP))
        if "image" in sample and sample["image"] is not None:
            records.append((_pil_to_np(sample["image"]), LABEL_BLURRY))

    print(f"Total samples loaded: {len(records)}")
    print(f"  Sharp: {sum(1 for _, l in records if l == LABEL_SHARP)}")
    print(f"  Blurry: {sum(1 for _, l in records if l == LABEL_BLURRY)}")

    # Deterministic shuffle before split
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
