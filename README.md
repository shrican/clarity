# Clarity — Production CNN for Image Blur / Focus Detection

ClarityNet detects whether a photo subject is **in focus (sharp)** or has **blur / focus issues** — powered by a dual-branch CNN with ConvNeXt V2 Tiny backbone + multi-scale frequency analysis.

## Architecture (v2)

```
Input Image
    │
    ├── Semantic Branch (ConvNeXt V2 Tiny, pretrained ImageNet-1k)
    │       └── Global Average Pool → 768-d features
    │
    ├── Multi-Scale Frequency Branch
    │       ├── LoG kernels at σ=1, 3, 6 → edge maps
    │       ├── Sobel XY → gradient maps
    │       └── 3-stage lightweight CNN → 256-d features
    │
    ├── FFT Sharpness Feature (zero learnable params)
    │       └── 2D FFT power spectrum → high-freq energy ratio → 3-d
    │
    └── Multi-Scale Laplacian Scalars
            └── Laplacian variance at 3 scales → 3-d
                       ↓
    Fusion: concat (1030-d) → CoordinateAttention → Dropout
                       │
                       ├── Classification head → logits (2 classes: sharp / blurry)
                       └── Regression head → blur score ∈ [0, 1]
```

**Key design decisions:**
- **ConvNeXt V2 Tiny** (vs V1): GRN (Global Response Normalization) reduces feature collapse, better decorrelation
- **Multi-scale frequency branch** (σ=1,3,6 LoG + Sobel): captures fine, mid, and coarse blur signatures simultaneously
- **FFT sharpness feature**: physics-based high-frequency energy ratio — zero parameters, direct signal
- **CoordinateAttention**: factorized H/W spatial attention (LayerNorm, not BatchNorm — safe at batch size 1)
- **Dual head** (classification + regression): fine-grained blur scoring, not just binary labels
- **Focal loss + auxiliary MSE regression**: handles class imbalance, down-weights easy examples
- **CutBlur augmentation**: cuts blurry patch into sharp image — forces regional blur detection vs global cues
- **Mixup disabled**: Mixup corrupts high-frequency edge statistics — the primary blur discriminative signal

## Benchmark Results

### Standard benchmark (25 epochs, 2400 samples, 128px)

| Metric        |   v1 (Baseline)  |   v2 (Improved)  |      Δ     |
|---------------|:----------------:|:----------------:|:----------:|
| Test Accuracy |      0.9975      |      1.0000      |  **+0.0025** |
| Test F1       |      0.9975      |      1.0000      |  **+0.0025** |
| Test AUC-ROC  |      1.0000      |      1.0000      |    0.0000  |
| Convergence   |  99.75% @ ep 5   |  100% @ ep 5     |  v2 faster |
| Parameters    |      28.84M      |      28.62M      | -0.22M     |

### Hard-regime benchmark (three challenging scenarios, 30 epochs)

| Scenario | v1 Acc | v2 Acc | Δ Acc | v1 AUC | v2 AUC | Δ AUC |
|----------|:------:|:------:|:-----:|:------:|:------:|:-----:|
| Near-threshold blur (σ=0.6–1.8) | 0.9925 | 0.9925 | 0.0000 | 0.9999 | 0.9994 | -0.0005 |
| Small data (300 samples) | **0.9875** | 0.9750 | -0.0125 | 1.0000 | 0.9976 | -0.0024 |
| Texture-confounded (soft + noise) | 0.9775 | **0.9925** | **+0.0150** | 0.9985 | **0.9989** | +0.0004 |

**Convergence speed (texture-confounded scenario — most realistic):**
| Milestone | v1 | v2 | Gain |
|-----------|:--:|:--:|:----:|
| 90% accuracy | epoch 11 | epoch 6 | **5 epochs faster** |
| 95% accuracy | epoch 16 | epoch 6 | **10 epochs faster** |
| 99% accuracy | never | epoch 6 | **v2 only** |

**Key findings:**
- **Near-threshold**: both models tie at 99.25%; v2 reaches 99% (v1 never does)
- **Small data**: v1 wins — simpler model generalizes better with only 300 samples
- **Texture-confounded** (the hard, real-world scenario): v2 wins by +1.5% accuracy and converges 10 epochs faster to 95%. This is the scenario that matters most in production — soft-textured subjects, sensor noise, and subtle blur.

## Dataset

Two-stage training pipeline:

**Stage 1 (synthetic pretraining):** [`wtcherr/unsplash_10k_blur_rand_KS`](https://huggingface.co/datasets/wtcherr/unsplash_10k_blur_rand_KS)
- 10k paired images (sharp `guide` + blurry `image`) from Unsplash
- ~20k total examples after pairing

**Stage 2 (real-label fine-tuning, optional):**
- [`chitradrishti/cuhk-blur`](https://huggingface.co/datasets/chitradrishti/cuhk-blur)
- [`chitradrishti/Flickr-Blur`](https://huggingface.co/datasets/chitradrishti/Flickr-Blur)

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
- AdamW + cosine LR schedule with linear warmup
- Focal loss (α=0.25, γ=2.0) + auxiliary MSE regression (λ=0.3)
- CutBlur augmentation + balanced class sampling (WeightedRandomSampler)
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
│   ├── augmentations.py  # Blur augmentation + CutBlur + transforms
│   ├── model.py          # ClarityNet v1/v2 architecture
│   ├── losses.py         # Focal loss + combined loss
│   ├── trainer.py        # Training loop (MPS/CUDA/CPU)
│   ├── evaluator.py      # Metrics + TTA + threshold search
│   └── utils.py          # Device, LR schedule, checkpointing
├── api/
│   ├── main.py           # FastAPI server
│   └── predictor.py      # Inference wrapper
├── scripts/
│   ├── train.py          # Training entrypoint
│   ├── evaluate.py       # Evaluation entrypoint
│   ├── predict.py        # Single/batch image prediction
│   ├── benchmark.py      # v1 vs v2 standard benchmark
│   └── benchmark_hard.py # Hard-regime 3-scenario benchmark
├── tests/                # Pytest suite (41 tests)
├── configs/default.yaml  # All hyperparameters
├── Dockerfile
└── pyproject.toml
```

## Tests

```bash
pytest tests/ -v
# 41 tests covering model, augmentations, config, and API
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
