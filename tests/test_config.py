"""Tests for configuration loading."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.config import Config, ModelConfig, TrainingConfig


def test_default_config():
    cfg = Config()
    assert cfg.model.backbone == "convnextv2_tiny"
    assert cfg.training.epochs == 30
    assert cfg.dataset.image_size == 224
    assert cfg.training.mixup_alpha == 0.0  # disabled for blur detection


def test_config_from_yaml():
    cfg = Config.from_yaml("configs/default.yaml")
    assert cfg.model.backbone == "convnextv2_tiny"
    assert cfg.training.focal_gamma == 2.0
    assert cfg.dataset.val_split == 0.15
    assert cfg.dataset.pretrain_dataset == "wtcherr/unsplash_10k_blur_rand_KS"
    assert "chitradrishti/cuhk-blur" in cfg.dataset.finetune_datasets


def test_config_to_dict():
    cfg = Config()
    d = cfg.to_dict()
    assert "model" in d and "training" in d and "dataset" in d
    assert d["model"]["backbone"] == "convnextv2_tiny"
