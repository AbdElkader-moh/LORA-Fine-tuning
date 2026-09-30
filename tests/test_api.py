import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from tests.conftest import CLASS_NAMES


@pytest.fixture(scope="module")
def client(artifact_dir):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(artifact_dir))
    with TestClient(app) as test_client:  # the context manager runs the startup model load
        yield test_client
    mp.undo()


def jpeg_bytes(size=(64, 64), color=(30, 120, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_labels(client):
    response = client.get("/labels")
    assert response.status_code == 200
    assert response.json() == {"labels": CLASS_NAMES}


def test_predict_returns_ranked_labels(client):
    response = client.post("/predict", files={"file": ("tile.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()

    assert body["filename"] == "tile.jpg"
    assert body["label"] in CLASS_NAMES
    assert body["label"] == body["top_k"][0]["label"]
    confidences = [p["confidence"] for p in body["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert abs(sum(confidences) - 1.0) < 1e-2  # top_k=3 covers all 3 test classes
    assert body["inference_ms"] >= 0


def test_predict_respects_top_k(client):
    response = client.post("/predict?top_k=1", files={"file": ("tile.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert len(response.json()["top_k"]) == 1


def test_predict_accepts_non_rgb_images(client):
    buffer = io.BytesIO()
    Image.new("L", (100, 40), 128).save(buffer, format="PNG")
    response = client.post("/predict", files={"file": ("gray.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_predict_requires_a_file(client):
    assert client.post("/predict").status_code == 422
