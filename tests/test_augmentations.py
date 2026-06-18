"""Tests for blur augmentation pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from clarity.augmentations import (
    apply_synthetic_blur, apply_cutblur, get_train_transform,
    get_val_transform, get_tta_transforms,
)


def _rand_image(H=256, W=256):
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


class TestSyntheticBlur:
    def test_gaussian_blur(self):
        img = _rand_image()
        result, btype = apply_synthetic_blur(img, "gaussian")
        assert result.shape == img.shape
        assert btype == "gaussian"
        assert result.dtype == np.uint8

    def test_motion_blur(self):
        img = _rand_image()
        result, btype = apply_synthetic_blur(img, "motion")
        assert result.shape == img.shape

    def test_defocus_blur(self):
        img = _rand_image()
        result, btype = apply_synthetic_blur(img, "defocus")
        assert result.shape == img.shape

    def test_cutblur_output(self):
        img = _rand_image()
        result, label = apply_cutblur(img, blur_prob=1.0)
        assert result.shape == img.shape
        assert label == 1  # forced blurry

    def test_cutblur_passthrough(self):
        img = _rand_image()
        result, label = apply_cutblur(img, blur_prob=0.0)
        assert label == 0  # kept sharp
        assert (result == img).all()

    def test_random_blur_type(self):
        img = _rand_image()
        result, btype = apply_synthetic_blur(img)
        assert btype in ("gaussian", "motion", "defocus")
        assert result.shape == img.shape


class TestTransforms:
    def test_train_transform_output(self):
        tf = get_train_transform(224)
        img = _rand_image()
        out = tf(image=img)["image"]
        assert out.shape == (3, 224, 224)

    def test_val_transform_output(self):
        tf = get_val_transform(224)
        img = _rand_image(300, 300)
        out = tf(image=img)["image"]
        assert out.shape == (3, 224, 224)

    def test_tta_transforms_count(self):
        transforms = get_tta_transforms(224)
        assert len(transforms) == 5
        img = _rand_image(300, 300)
        for tf in transforms:
            out = tf(image=img)["image"]
            assert out.shape == (3, 224, 224)

    def test_train_transform_different_sizes(self):
        for size in [128, 192, 224]:
            tf = get_train_transform(size)
            img = _rand_image(size * 2, size * 2)
            out = tf(image=img)["image"]
            assert out.shape == (3, size, size)
