"""
Stateless predictor for production inference.
Loaded once at API startup; shared across requests.
"""
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from clarity.config import Config
from clarity.model import ClarityNet
from clarity.augmentations import get_val_transform
from clarity.utils import get_device, load_checkpoint


BLUR_TYPE_LABELS = ["sharp", "mild_blur", "heavy_blur"]


class Predictor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.inference.device)
        self.transform = get_val_transform(cfg.dataset.image_size)

        self.model = ClarityNet(
            backbone=cfg.model.backbone,
            pretrained=False,
            num_classes=cfg.model.num_classes,
            dropout=0.0,
            freq_branch=cfg.model.freq_branch,
            cbam_reduction=cfg.model.cbam_reduction,
        ).to(self.device)

        ckpt_path = Path(cfg.inference.checkpoint)
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, self.model, self.device)
        else:
            # Allow import without checkpoint for testing
            import warnings
            warnings.warn(f"Checkpoint not found at {ckpt_path}. Model weights are random.")

        self.model.eval()

    @torch.no_grad()
    def predict_pil(self, pil_image: Image.Image) -> dict:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        img_np = np.array(pil_image, dtype=np.uint8)

        tensor = self.transform(image=img_np)["image"].unsqueeze(0).to(self.device)

        out = self.model(tensor)
        probs = F.softmax(out["logits"], dim=1)[0]
        blur_score = float(out["blur_score"][0, 0])

        is_sharp = bool(probs[0] >= self.cfg.eval.threshold)
        label = "sharp" if is_sharp else "blurry"

        # Qualitative blur severity
        if is_sharp:
            blur_type = "sharp"
        elif blur_score < 0.6:
            blur_type = "mild_blur"
        else:
            blur_type = "heavy_blur"

        return {
            "is_sharp": is_sharp,
            "label": label,
            "confidence": float(probs[0] if is_sharp else probs[1]),
            "sharp_probability": float(probs[0]),
            "blurry_probability": float(probs[1]),
            "blur_score": blur_score,
            "blur_type": blur_type,
        }

    @torch.no_grad()
    def predict_bytes(self, image_bytes: bytes) -> dict:
        pil = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(pil)
