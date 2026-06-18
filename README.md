# Clarity — Production CNN for Image Blur / Focus Detection

ClarityNet detects whether a photo subject is **in focus (sharp)** or has **blur / focus issues** — powered by a dual-branch CNN with ConvNeXt-Tiny backbone + frequency analysis.

## Architecture

```
Input Image
    │
    ├── Semantic Branch (ConvNeXt-Tiny, pretrained ImageNet-1k)
    │       └── Global Average Pool → 768-d features
    │
    ├── Frequency Branch (Laplacian CNN)
    │       ├── Fixed Laplacian kernel → gradient map
    │       └── 4-stage lightweight CNN → 128-d features
    │
    └── Laplacian Variance feature (1-d scalar, log-scale)
           ↓
    Fusion: concat (897-d) → CBAM channel attention → Dropout
           │
           ├── Classification head → logits (2 classes: sharp / blurry)
           └── Regression head → blur score ∈ [0, 1]
```

**Key design decisions:**
- **Frequency branch** preserves high-frequency sharpness cues that deep CNNs typically discard in early downsampling
- **Laplacian variance** provides a deterministic, no-reference sharpness signal as an inductive bias
- **CBAM attention** re-weights fused features, learning which channels carry discriminative sharpness info
- **Dual head** (classification + regression) allows fine-grained blur scoring, not just binary labels
- **Focal loss** handles class imbalance and down-weights easy examples
- **Progressive resizing** (128 → 192 → 224px) improves generalization and training stability

## Dataset

Primary: [`wtcherr/unsplash_10k_blur_rand_KS`](https://huggingface.co/datasets/wtcherr/unsplash_10k_blur_rand_KS)
- 10k paired images (sharp `guide` + blurry `image`) from Unsplash
- ~20k total examples after pairing
- Augmented with synthetic Gaussian, motion, and defocus blur at training time
- Hard-negative mining: 15% of sharp images receive extra blur → labeled blurry

Split: 75% train / 15% val / 10% test (fixed seed)

## Training

```bash
# Full training (30 epochs, MPS by default)
python scripts/train.py

# Quick run
python scripts/train.py --epochs 5 --batch-size 16

# CPU fallback
python scripts/train.py --device cpu --epochs 5
```

Training features:
- MPS (Apple Silicon) / CUDA / CPU support via auto-detect
- Progressive resizing: 128 → 192 → 224px over epochs
- Mixup augmentation (α=0.4)
- AdamW + cosine LR schedule with linear warmup
- Focal loss (α=0.25, γ=2.0) + auxiliary MSE regression (λ=0.3)
- Gradient clipping (max norm=1.0)
- Best model saved to `checkpoints/best.pt` by val F1

## Evaluation

```bash
python scripts/evaluate.py                   # standard eval
python scripts/evaluate.py --tta             # with 5-crop TTA
```

Metrics: Accuracy, F1, Precision, Recall, AUC-ROC, confusion matrix, optimal threshold search.

## Inference

**Single image:**
```bash
python scripts/predict.py photo.jpg
python scripts/predict.py *.jpg --json
```

**REST API:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# or
python api/main.py
```

```
POST /predict           — single image upload
POST /predict/batch     — up to 32 images
GET  /health
GET  /model/info
```

Response:
```json
{
  "is_sharp": true,
  "label": "sharp",
  "confidence": 0.94,
  "sharp_probability": 0.94,
  "blurry_probability": 0.06,
  "blur_score": 0.12,
  "blur_type": "sharp",
  "latency_ms": 18.3
}
```

**Docker:**
```bash
docker build -t clarity .
docker run -p 8000:8000 clarity
```

## Project Structure

```
clarity/
├── src/clarity/
│   ├── config.py         # Dataclass-based config
│   ├── dataset.py        # HuggingFace data pipeline
│   ├── augmentations.py  # Blur augmentation + transforms
│   ├── model.py          # ClarityNet architecture
│   ├── losses.py         # Focal loss + combined loss
│   ├── trainer.py        # Training loop (MPS/CUDA/CPU)
│   ├── evaluator.py      # Metrics + TTA + threshold search
│   └── utils.py          # Device, mixup, LR schedule, checkpointing
├── api/
│   ├── main.py           # FastAPI server
│   └── predictor.py      # Inference wrapper
├── scripts/
│   ├── train.py          # Training entrypoint
│   ├── evaluate.py       # Evaluation entrypoint
│   └── predict.py        # Single/batch image prediction
├── tests/                # Pytest suite (33 tests)
├── configs/default.yaml  # All hyperparameters
├── Dockerfile
└── pyproject.toml
```

## Tests

```bash
pytest tests/ -v
# 33 tests covering model, augmentations, config, and API
```

## Requirements

- Python 3.11+
- PyTorch 2.2+ (MPS or CUDA recommended)
- timm 1.0+
- albumentations 1.4+
- fastapi, uvicorn (for API)

```bash
pip install -e ".[dev]"
```
