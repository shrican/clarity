#!/usr/bin/env python3
"""
Train ClarityNet.

Usage:
  python scripts/train.py
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --epochs 10 --batch-size 16
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.config import Config
from clarity.trainer import Trainer
from clarity.utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="Train ClarityNet blur detector")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None, choices=["mps", "cuda", "cpu"])
    p.add_argument("--no-freq-branch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging()

    cfg = Config.from_yaml(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.device is not None:
        cfg.inference.device = args.device
    if args.no_freq_branch:
        cfg.model.freq_branch = False

    log.info("Config:\n" + "\n".join(f"  {k}: {v}" for k, v in cfg.to_dict().items()))

    trainer = Trainer(cfg)
    results = trainer.train()
    log.info(f"Training complete. Best val F1: {results['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
