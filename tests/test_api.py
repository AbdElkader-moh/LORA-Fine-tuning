import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from tests.conftest import CLASS_NAMES


@pytest.fixture(scope="module")
def client(artifact_dir, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(artifact_dir))
    mp.setenv("MONITOR_DIR", str(tmp_path_factory.mktemp("monitoring")))
    mp.setenv("DRIFT_MIN_SAMPLES", "3")
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
    assert response.json() == {"status": "ok", "model_loaded": True, "version": app.version}


def test_labels(client):
    response = client.get("/labels")
    assert response.status_code == 200
    assert response.json() == {"labels": CLASS_NAMES}


def test_predict_returns_ranked_labels(client):
    response = client.post("/predict", files={"file": ("tile.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()

    assert body["filename"] == "tile.jpg"
    assert len(body["request_id"]) == 32
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


def predict_id(client):
    response = client.post("/predict", files={"file": ("tile.jpg", jpeg_bytes(), "image/jpeg")})
    return response.json()["request_id"]


def test_feedback_is_joined_to_the_prediction(client):
    request_id = predict_id(client)
    response = client.post("/feedback", json={"request_id": request_id, "label": "River"})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == request_id and body["true_label"] == "River"
    assert body["correct"] == (body["predicted"] == "River")

    perf = client.get("/monitoring/performance").json()
    assert perf["window_size"] >= 1 and 0 <= perf["accuracy"] <= 1
    assert perf["baseline_accuracy"] == 0.9


def test_feedback_rejects_unknown_ids_and_labels(client):
    assert client.post("/feedback", json={"request_id": "nope", "label": "River"}).status_code == 404
    assert client.post("/feedback", json={"request_id": predict_id(client), "label": "Desert"}).status_code == 422


def test_drift_report(client):
    for _ in range(3):
        predict_id(client)
    body = client.get("/monitoring/drift").json()
    assert body["reference_loaded"] is True
    assert body["status"] in {"ok", "drift"}
    assert "brightness" in body["features"]
    assert set(body["predictions"]["label_distribution"]) == set(CLASS_NAMES)


def test_metrics_exposes_model_drift_and_infrastructure(client):
    predict_id(client)
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    for name in ["vit_http_requests_total", "vit_predictions_total", "vit_prediction_confidence_bucket",
                 "vit_inference_duration_seconds_bucket", "vit_feature_drift_psi", "vit_window_accuracy",
                 "vit_baseline_accuracy", "vit_system_cpu_percent", "vit_process_resident_memory_bytes"]:
        assert name in text, name
    assert 'path="/predict"' in text
