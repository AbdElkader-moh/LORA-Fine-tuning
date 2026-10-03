import numpy as np
import pytest

from app.monitoring import ModelMonitor, PredictionStore
from app.monitoring.drift import feature_drift, ks_test, psi
from app.monitoring.features import FEATURE_NAMES, image_features, prediction_entropy
from app.monitoring.performance import classification_metrics
from tests.conftest import CLASS_NAMES, make_reference_profile, textured_image


def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0)
    assert psi([0.1] * 10, [0.55, 0.45] + [0.0] * 8) > 0.25


def test_ks_test_separates_same_and_shifted_samples():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    _, p_same = ks_test(a, b)
    d_shift, p_shift = ks_test(a, b + 1.0)
    assert p_same > 0.01
    assert d_shift > 0.3 and p_shift < 1e-6


def test_image_features_track_brightness():
    rng = np.random.default_rng(1)
    bright, dark = image_features(textured_image(rng)), image_features(textured_image(rng, scale=0.4))
    assert set(bright) == set(FEATURE_NAMES)
    assert dark["brightness"] < bright["brightness"]


def test_prediction_entropy_bounds():
    assert prediction_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert prediction_entropy(np.full(4, 0.25)) == pytest.approx(1.0)


def test_feature_drift_flags_darkened_images_only():
    reference = make_reference_profile()
    rng = np.random.default_rng(42)
    same = feature_drift(reference, [image_features(textured_image(rng)) for _ in range(150)])
    dark = feature_drift(reference, [image_features(textured_image(rng, scale=0.4)) for _ in range(150)])
    assert not same["brightness"]["drifted"]
    assert dark["brightness"]["drifted"] and dark["brightness"]["status"] == "drift"


def test_classification_metrics():
    pairs = [("Forest", "Forest"), ("River", "River"), ("River", "SeaLake"), ("SeaLake", "SeaLake")]
    result = classification_metrics(pairs, CLASS_NAMES)
    assert result["accuracy"] == 0.75
    assert result["per_class_recall"] == {"Forest": 1.0, "River": 1.0, "SeaLake": 0.5}
    assert classification_metrics([], CLASS_NAMES)["accuracy"] is None


def test_monitor_logs_predictions_and_joins_feedback(tmp_path):
    monitor = ModelMonitor(PredictionStore(tmp_path / "p.db"), CLASS_NAMES, "test",
                           reference=make_reference_profile(), min_drift_samples=5)
    rng = np.random.default_rng(3)
    ids = [monitor.record_prediction(textured_image(rng), np.array([0.7, 0.2, 0.1]), 5.0) for _ in range(10)]

    assert monitor.record_feedback("does-not-exist", "Forest") is None
    assert monitor.record_feedback(ids[0], "Forest")["correct"] is True
    assert monitor.record_feedback(ids[1], "River")["correct"] is False

    perf = monitor.performance_report()
    assert perf["window_size"] == 2 and perf["accuracy"] == 0.5
    assert perf["predictions"] == 10 and perf["baseline_accuracy"] == 0.9

    drift = monitor.drift_report()
    assert drift["window_size"] == 10
    # Every prediction is 'Forest' at 0.7 confidence: the output mix has drifted.
    assert drift["predictions"]["label_status"] == "drift"
    assert drift["status"] == "drift"


def test_monitor_without_reference_reports_it(tmp_path):
    monitor = ModelMonitor(PredictionStore(tmp_path / "p.db"), CLASS_NAMES, "test")
    assert monitor.drift_report()["status"] == "no_reference"
