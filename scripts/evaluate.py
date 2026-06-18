#!/usr/bin/env python3
"""
Evaluate a trained ClarityNet checkpoint on the held-out test set.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint checkpoints/best.pt --tta
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.config import Config
from clarity.dataset import BlurDataModule
from clarity.evaluator import Evaluator
from clarity.utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging()

    cfg = Config.from_yaml(args.config)
    if args.checkpoint:
        cfg.inference.checkpoint = args.checkpoint
    if args.device:
        cfg.inference.device = args.device

    data = BlurDataModule(cfg)
    evaluator = Evaluator(cfg)
    evaluator.load_best()

    use_tta = args.tta or cfg.eval.tta
    metrics = evaluator.evaluate(
        data.test_loader(),
        use_tta=use_tta,
        tta_loaders=data.tta_loaders() if use_tta else None,
    )

    print("\n=== Test Results ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")


if __name__ == "__main__":
    main()
