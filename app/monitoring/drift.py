import json
import math
from pathlib import Path

import numpy as np

REFERENCE_FILE = "reference_profile.json"

# Rule-of-thumb PSI bands used across the industry. Only PSI decides "drifted":
# with 7 features checked on every scrape, a KS p-value cut-off fires on clean
# 200-image windows ~13% of the time, while clean PSI stays under 0.14 (p99).
# KS is still reported as supporting evidence.
PSI_WARNING = 0.1
PSI_DRIFT = 0.25
EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two discrete distributions
    (proportions over the same bins)."""
    expected = np.clip(np.asarray(expected, dtype=np.float64), EPS, None)
    actual = np.clip(np.asarray(actual, dtype=np.float64), EPS, None)
    expected, actual = expected / expected.sum(), actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def quantile_edges(values, n_bins: int = 10) -> np.ndarray:
    """Bin edges at the reference's deciles, so each bin holds ~10% of the
    reference data; the outer edges are open to catch out-of-range values."""
    inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def binned_proportions(values, edges) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(len(values), 1)


def ks_test(reference, current):
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value) with the
    asymptotic p-value, so the service does not need scipy."""
    reference, current = np.sort(reference), np.sort(current)
    grid = np.concatenate([reference, current])
    cdf_ref = np.searchsorted(reference, grid, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, grid, side="right") / len(current)
    d = float(np.max(np.abs(cdf_ref - cdf_cur)))

    n_eff = len(reference) * len(current) / (len(reference) + len(current))
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def severity(psi_value: float) -> str:
    if psi_value >= PSI_DRIFT:
        return "drift"
    if psi_value >= PSI_WARNING:
        return "warning"
    return "ok"


class ReferenceProfile:
    """What 'normal' looks like: input features from the training split and
    the model's outputs on the validation split, saved by build_reference.py."""

    def __init__(self, data: dict):
        self.data = data
        self.features = {k: np.asarray(v, dtype=np.float64) for k, v in data["features"].items()}
        self.confidence = np.asarray(data["confidence"], dtype=np.float64)
        self.label_counts = data["label_counts"]
        self.accuracy = data.get("accuracy")
        self.macro_f1 = data.get("macro_f1")
        self.edges = {k: quantile_edges(v) for k, v in self.features.items()}
        self.confidence_edges = quantile_edges(self.confidence)

    @classmethod
    def load(cls, model_dir):
        path = Path(model_dir) / REFERENCE_FILE
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, model_dir):
        path = Path(model_dir) / REFERENCE_FILE
        path.write_text(json.dumps(self.data), encoding="utf-8")
        return path

    def label_proportions(self, class_names) -> np.ndarray:
        counts = np.array([self.label_counts.get(c, 0) for c in class_names], dtype=np.float64)
        return counts / max(counts.sum(), 1)


def feature_drift(reference: ReferenceProfile, feature_rows: list) -> dict:
    """PSI + KS for every input feature over the current window."""
    report = {}
    for name, ref_values in reference.features.items():
        current = np.array([row[name] for row in feature_rows if name in row], dtype=np.float64)
        if len(current) == 0:
            continue
        edges = reference.edges[name]
        value = psi(binned_proportions(ref_values, edges), binned_proportions(current, edges))
        ks_stat, ks_p = ks_test(ref_values, current)
        report[name] = {
            "psi": round(value, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 6),
            "reference_mean": round(float(ref_values.mean()), 4),
            "current_mean": round(float(current.mean()), 4),
            "status": severity(value),
            "drifted": value >= PSI_DRIFT,
        }
    return report


def prediction_drift(reference: ReferenceProfile, labels: list, confidences: list, class_names) -> dict:
    """Shift in what the model outputs: the predicted-class mix and its confidence."""
    current_props = np.array([labels.count(c) for c in class_names], dtype=np.float64) / max(len(labels), 1)
    label_psi = psi(reference.label_proportions(class_names), current_props)

    conf = np.asarray(confidences, dtype=np.float64)
    edges = reference.confidence_edges
    conf_psi = psi(binned_proportions(reference.confidence, edges), binned_proportions(conf, edges))
    return {
        "label_psi": round(label_psi, 4),
        "label_status": severity(label_psi),
        "label_distribution": {c: round(float(p), 4) for c, p in zip(class_names, current_props)},
        "confidence_psi": round(conf_psi, 4),
        "confidence_status": severity(conf_psi),
        "reference_mean_confidence": round(float(reference.confidence.mean()), 4),
        "current_mean_confidence": round(float(conf.mean()), 4),
    }
