#!/usr/bin/env python3
"""
Run blur detection on one or more image files.

Usage:
  python scripts/predict.py photo.jpg
  python scripts/predict.py *.jpg --threshold 0.6
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.config import Config
from api.predictor import Predictor
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="Image file paths")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--json", action="store_true", help="Output JSON")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.threshold is not None:
        cfg.eval.threshold = args.threshold

    predictor = Predictor(cfg)
    results = []

    for path_str in args.images:
        path = Path(path_str)
        if not path.exists():
            print(f"  [SKIP] {path} not found", file=sys.stderr)
            continue
        pil = Image.open(path)
        result = predictor.predict_pil(pil)
        result["file"] = str(path)
        results.append(result)

        if not args.json:
            status = "✓ SHARP" if result["is_sharp"] else "✗ BLURRY"
            print(
                f"{status}  {path.name}  "
                f"blur_score={result['blur_score']:.3f}  "
                f"confidence={result['confidence']:.3f}  "
                f"type={result['blur_type']}"
            )

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
