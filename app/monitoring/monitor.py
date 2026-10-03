import threading
import uuid

from . import metrics as m
from .drift import ReferenceProfile, feature_drift, prediction_drift
from .features import image_features, prediction_entropy
from .performance import classification_metrics
from .store import PredictionStore


class ModelMonitor:
    """Everything the API needs for AI observability:

    * logs each prediction with its input features (store.py),
    * accepts ground-truth labels that arrive later (feedback),
    * compares the latest window of traffic with the training reference
      (data drift + prediction drift) and the labelled window with the
      release baseline (performance), and publishes all of it as
      Prometheus metrics.
    """

    def __init__(self, store: PredictionStore, class_names, model_version, reference: ReferenceProfile = None,
                 drift_window=200, min_drift_samples=30, performance_window=200,
                 low_confidence_threshold=0.5):
        self.store = store
        self.class_names = list(class_names)
        self.model_version = model_version
        self.reference = reference
        self.drift_window = drift_window
        self.min_drift_samples = min_drift_samples
        self.performance_window = performance_window
        self.low_confidence_threshold = low_confidence_threshold
        self._lock = threading.Lock()

        m.REFERENCE_LOADED.set(1 if reference else 0)
        if reference:
            m.REFERENCE_MEAN_CONFIDENCE.set(float(reference.confidence.mean()))
        if reference and reference.accuracy is not None:
            m.BASELINE_ACCURACY.set(reference.accuracy)
        if reference and reference.macro_f1 is not None:
            m.BASELINE_MACRO_F1.set(reference.macro_f1)

    # ---- per request ---------------------------------------------------
    def record_prediction(self, image, probs, inference_ms):
        """Logs one prediction and returns its request id (the key for feedback)."""
        request_id = uuid.uuid4().hex
        top = int(probs.argmax())
        label, confidence = self.class_names[top], float(probs[top])
        self.store.log_prediction(
            request_id, self.model_version, label, confidence, prediction_entropy(probs),
            inference_ms, image_features(image),
        )
        m.INFERENCE_LATENCY.observe(inference_ms / 1000)
        m.PREDICTIONS.labels(label=label).inc()
        m.CONFIDENCE.observe(confidence)
        if confidence < self.low_confidence_threshold:
            m.LOW_CONFIDENCE.inc()
        return request_id

    def record_feedback(self, request_id, true_label):
        """Returns {"correct": bool} or None if the request id is unknown."""
        predicted = self.store.add_feedback(request_id, true_label)
        if predicted is None:
            return None
        correct = predicted == true_label
        m.FEEDBACK.labels(correct=str(correct).lower()).inc()
        return {"request_id": request_id, "predicted": predicted, "true_label": true_label, "correct": correct}

    # ---- window reports (also refresh the gauges) ------------------------
    def drift_report(self):
        rows = self.store.recent_predictions(self.drift_window)
        report = {"window_size": len(rows), "reference_loaded": self.reference is not None,
                  "min_samples": self.min_drift_samples}
        m.DRIFT_WINDOW.set(len(rows))
        if rows:
            m.WINDOW_MEAN_CONFIDENCE.set(sum(r["confidence"] for r in rows) / len(rows))
            labels = [r["label"] for r in rows]
            for c in self.class_names:
                m.WINDOW_PREDICTION_SHARE.labels(label=c).set(labels.count(c) / len(rows))

        if self.reference is None:
            report["status"] = "no_reference"
            return report
        if len(rows) < self.min_drift_samples:
            report["status"] = "insufficient_data"
            return report

        features = feature_drift(self.reference, [r["features"] for r in rows])
        predictions = prediction_drift(
            self.reference, [r["label"] for r in rows], [r["confidence"] for r in rows], self.class_names)
        drifted = [name for name, f in features.items() if f["drifted"]]

        with self._lock:
            for name, f in features.items():
                m.FEATURE_PSI.labels(feature=name).set(f["psi"])
                m.FEATURE_KS.labels(feature=name).set(f["ks_statistic"])
                m.FEATURE_MEAN.labels(feature=name, window="current").set(f["current_mean"])
                m.FEATURE_MEAN.labels(feature=name, window="reference").set(f["reference_mean"])
                m.FEATURE_DRIFTED.labels(feature=name).set(1 if f["drifted"] else 0)
            m.DRIFTED_SHARE.set(len(drifted) / max(len(features), 1))
            m.LABEL_PSI.set(predictions["label_psi"])
            m.CONFIDENCE_PSI.set(predictions["confidence_psi"])

        report.update({
            "status": "drift" if drifted or predictions["label_status"] == "drift" else "ok",
            "drifted_features": drifted,
            "features": features,
            "predictions": predictions,
        })
        return report

    def performance_report(self):
        pairs = self.store.recent_labelled(self.performance_window)
        perf = classification_metrics(pairs, self.class_names)
        m.WINDOW_LABELLED.set(len(pairs))
        if perf["accuracy"] is not None:
            m.WINDOW_ACCURACY.set(perf["accuracy"])
            m.WINDOW_MACRO_F1.set(perf["macro_f1"])
            for c, r in perf["per_class_recall"].items():
                m.CLASS_RECALL.labels(label=c).set(r)
        return {
            "window_size": len(pairs),
            **perf,
            "baseline_accuracy": self.reference.accuracy if self.reference else None,
            "baseline_macro_f1": self.reference.macro_f1 if self.reference else None,
            **self.store.counts(),
        }

    def refresh(self):
        """Recompute every window metric; called on each Prometheus scrape."""
        self.drift_report()
        self.performance_report()
