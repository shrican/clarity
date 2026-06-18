"""Integration tests for the FastAPI inference server."""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient


def _make_jpeg_bytes(H=224, W=224) -> bytes:
    """Create a random in-memory JPEG."""
    arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "backbone" in data
        assert "classes" in data


class TestPredictEndpoint:
    def test_predict_returns_fields(self, client):
        jpeg = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "is_sharp" in data
        assert "label" in data
        assert "confidence" in data
        assert "blur_score" in data
        assert "blur_type" in data
        assert "latency_ms" in data

    def test_predict_label_values(self, client):
        jpeg = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        data = resp.json()
        assert data["label"] in ("sharp", "blurry")
        assert data["blur_type"] in ("sharp", "mild_blur", "heavy_blur")
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["blur_score"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, client):
        jpeg = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        data = resp.json()
        total = data["sharp_probability"] + data["blurry_probability"]
        assert abs(total - 1.0) < 1e-4

    def test_predict_rejects_non_image(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 422

    def test_predict_png(self, client):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        resp = client.post(
            "/predict",
            files={"file": ("test.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200


class TestBatchEndpoint:
    def test_batch_predict(self, client):
        files = [
            ("files", ("img1.jpg", _make_jpeg_bytes(), "image/jpeg")),
            ("files", ("img2.jpg", _make_jpeg_bytes(), "image/jpeg")),
            ("files", ("img3.jpg", _make_jpeg_bytes(), "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        assert "total_latency_ms" in data
