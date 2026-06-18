"""Unit tests for ClarityNet v1 and v2 architectures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from clarity.model import (
    ClarityNet, LaplacianLayer, FrequencyBranch, MultiScaleFrequencyBranch,
    FFTSharpnessFeature, CoordinateAttention, laplacian_variance_feature,
    multi_scale_sharpness_scalars,
)
from clarity.losses import FocalLoss, ClarityLoss


@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def model(device):
    return ClarityNet(backbone="convnextv2_tiny", pretrained=False, model_version="v2").to(device)


@pytest.fixture
def model_v1(device):
    return ClarityNet(backbone="convnextv2_tiny", pretrained=False, model_version="v1").to(device)


def _rand_batch(B=2, C=3, H=224, W=224, device="cpu"):
    return torch.rand(B, C, H, W, device=device)


class TestModelForward:
    def test_output_keys(self, model, device):
        x = _rand_batch(device=device)
        out = model(x)
        assert "logits" in out and "blur_score" in out

    def test_logits_shape(self, model, device):
        x = _rand_batch(B=4, device=device)
        out = model(x)
        assert out["logits"].shape == (4, 2)

    def test_blur_score_range(self, model, device):
        x = _rand_batch(B=4, device=device)
        out = model(x)
        assert out["blur_score"].min() >= 0.0
        assert out["blur_score"].max() <= 1.0

    def test_predict_interface(self, model, device):
        x = _rand_batch(B=2, device=device)
        result = model.predict(x)
        assert result["label"].shape == (2,)
        assert result["confidence"].shape == (2,)
        assert all(c >= 0 and c <= 1 for c in result["confidence"].tolist())

    def test_no_freq_branch(self, device):
        m = ClarityNet(backbone="convnextv2_tiny", pretrained=False, freq_branch=False).to(device)
        x = _rand_batch(device=device)
        out = m(x)
        assert out["logits"].shape == (2, 2)

    def test_v1_model(self, model_v1, device):
        x = _rand_batch(device=device)
        out = model_v1(x)
        assert out["logits"].shape == (2, 2)
        assert out["blur_score"].min() >= 0.0

    def test_small_input_size(self, model, device):
        x = _rand_batch(B=1, H=128, W=128, device=device)
        out = model(x)
        assert out["logits"].shape == (1, 2)

    def test_single_image(self, model, device):
        x = _rand_batch(B=1, device=device)
        out = model(x)
        assert out["logits"].shape == (1, 2)


class TestFrequencyBranch:
    def test_laplacian_output_shape(self, device):
        lap = LaplacianLayer().to(device)
        x = _rand_batch(device=device)
        out = lap(x)
        assert out.shape == x.shape

    def test_laplacian_range(self, device):
        lap = LaplacianLayer().to(device)
        x = _rand_batch(device=device)
        out = lap(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_freq_branch_v1_output(self, device):
        fb = FrequencyBranch(out_dim=128).to(device)
        x = _rand_batch(device=device)
        out = fb(x)
        assert out.shape == (2, 128)

    def test_multiscale_freq_branch_output(self, device):
        fb = MultiScaleFrequencyBranch(out_dim=256).to(device)
        x = _rand_batch(device=device)
        out = fb(x)
        assert out.shape == (2, 256)

    def test_fft_sharpness_feature(self, device):
        fft = FFTSharpnessFeature().to(device)
        x = _rand_batch(B=4, device=device)
        out = fft(x)
        assert out.shape == (4, 3)

    def test_fft_sharp_vs_blurry(self, device):
        """FFT should give higher values for sharp images (more high-freq energy)."""
        import numpy as np, cv2
        fft = FFTSharpnessFeature().to(device)
        sharp_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        blurry_np = cv2.GaussianBlur(sharp_np, (21, 21), 5.0)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        def to_tensor(img):
            t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
            for c in range(3):
                t[c] = (t[c] - mean[c]) / std[c]
            return t.unsqueeze(0).to(device)
        s_feats = fft(to_tensor(sharp_np))
        b_feats = fft(to_tensor(blurry_np))
        # High-freq ratio (index 2) should be higher for sharp image
        assert s_feats[0, 2] > b_feats[0, 2]

    def test_laplacian_variance_feature(self, device):
        x = _rand_batch(B=3, device=device)
        var = laplacian_variance_feature(x)
        assert var.shape == (3, 1)
        blurry = torch.zeros_like(x)
        sharp = torch.rand_like(x)
        assert laplacian_variance_feature(sharp).mean() > laplacian_variance_feature(blurry).mean()

    def test_multi_scale_sharpness_scalars(self, device):
        x = _rand_batch(B=3, device=device)
        feats = multi_scale_sharpness_scalars(x)
        assert feats.shape == (3, 3)

    def test_coordinate_attention(self, device):
        ca = CoordinateAttention(dim=256).to(device)
        x = torch.randn(4, 256, device=device)
        out = ca(x)
        assert out.shape == (4, 256)


class TestLosses:
    def test_focal_loss_shape(self):
        loss_fn = FocalLoss()
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_clarity_loss_returns_dict(self):
        criterion = ClarityLoss()
        logits = torch.randn(4, 2)
        blur_score = torch.sigmoid(torch.randn(4, 1))
        labels = torch.randint(0, 2, (4,))
        total, info = criterion(logits, blur_score, labels)
        assert "loss" in info and "cls_loss" in info and "reg_loss" in info
        assert total.item() > 0

    def test_focal_loss_with_all_easy(self):
        """Focal loss should be small when predictions are confident and correct."""
        loss_fn = FocalLoss(gamma=2.0)
        # Perfect predictions (very high logit for correct class)
        logits = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
        labels = torch.tensor([0, 0])
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.1
