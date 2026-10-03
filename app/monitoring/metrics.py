import os

import psutil
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info
from prometheus_client.core import GaugeMetricFamily

# ---- service ---------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "vit_http_requests_total", "HTTP requests handled", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "vit_http_request_duration_seconds", "End-to-end request latency", ["path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
HTTP_IN_PROGRESS = Gauge("vit_http_requests_in_progress", "Requests currently being served")

# ---- model outputs ---------------------------------------------------------
MODEL_INFO = Info("vit_model", "Model currently being served")
INFERENCE_LATENCY = Histogram(
    "vit_inference_duration_seconds", "Model forward-pass time",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2))
PREDICTIONS = Counter("vit_predictions_total", "Predictions by predicted class", ["label"])
CONFIDENCE = Histogram(
    "vit_prediction_confidence", "Top-1 softmax probability",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
LOW_CONFIDENCE = Counter("vit_low_confidence_predictions_total", "Predictions below the confidence threshold")

# ---- ground truth / performance -------------------------------------------
FEEDBACK = Counter("vit_feedback_total", "Ground-truth labels received", ["correct"])
WINDOW_ACCURACY = Gauge("vit_window_accuracy", "Accuracy over the latest labelled predictions")
WINDOW_MACRO_F1 = Gauge("vit_window_macro_f1", "Macro-F1 over the latest labelled predictions")
WINDOW_LABELLED = Gauge("vit_window_labelled_samples", "Labelled predictions in the performance window")
CLASS_RECALL = Gauge("vit_window_class_recall", "Per-class recall in the performance window", ["label"])
BASELINE_ACCURACY = Gauge("vit_baseline_accuracy", "Accuracy of this model on the held-out set at release")
BASELINE_MACRO_F1 = Gauge("vit_baseline_macro_f1", "Macro-F1 of this model on the held-out set at release")

# ---- drift -----------------------------------------------------------------
FEATURE_PSI = Gauge("vit_feature_drift_psi", "PSI of an input feature vs the training reference", ["feature"])
FEATURE_KS = Gauge("vit_feature_drift_ks_statistic", "KS statistic of an input feature vs the reference", ["feature"])
FEATURE_MEAN = Gauge("vit_feature_mean", "Mean of an input feature in the drift window", ["feature", "window"])
FEATURE_DRIFTED = Gauge("vit_feature_drift_detected", "1 if the feature has drifted", ["feature"])
DRIFTED_SHARE = Gauge("vit_drifted_features_share", "Share of input features flagged as drifted")
LABEL_PSI = Gauge("vit_prediction_drift_psi", "PSI of the predicted-class mix vs the validation reference")
CONFIDENCE_PSI = Gauge("vit_confidence_drift_psi", "PSI of top-1 confidence vs the validation reference")
WINDOW_MEAN_CONFIDENCE = Gauge("vit_window_mean_confidence", "Mean top-1 confidence in the drift window")
REFERENCE_MEAN_CONFIDENCE = Gauge("vit_reference_mean_confidence", "Mean top-1 confidence on the validation reference")
WINDOW_PREDICTION_SHARE = Gauge("vit_window_prediction_share", "Share of each class in the drift window", ["label"])
DRIFT_WINDOW = Gauge("vit_drift_window_samples", "Predictions in the drift window")
REFERENCE_LOADED = Gauge("vit_reference_profile_loaded", "1 if a drift reference profile is available")


class SystemCollector:
    """Infrastructure metrics read at scrape time: host CPU / memory / disk and
    this process's own footprint. Works on Linux, macOS and Windows alike
    (Prometheus' built-in process_* metrics are Linux-only)."""

    def __init__(self, disk_path="."):
        self.disk_path = disk_path
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)  # prime the counters; first reading is always 0
        psutil.cpu_percent(None)

    def collect(self):
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(self.disk_path)
        with self.process.oneshot():
            rss = self.process.memory_info().rss
            proc_cpu = self.process.cpu_percent(None)
            threads = self.process.num_threads()

        yield _gauge("vit_system_cpu_percent", "Host CPU utilisation", psutil.cpu_percent(None))
        yield _gauge("vit_system_cpu_count", "Logical CPUs", psutil.cpu_count())
        yield _gauge("vit_system_load1", "1-minute load average", psutil.getloadavg()[0])
        yield _gauge("vit_system_memory_total_bytes", "Host memory", mem.total)
        yield _gauge("vit_system_memory_used_percent", "Host memory in use", mem.percent)
        yield _gauge("vit_system_disk_used_percent", "Disk usage of the monitoring volume", disk.percent)
        yield _gauge("vit_process_cpu_percent", "CPU used by the API process (100 = one core)", proc_cpu)
        yield _gauge("vit_process_resident_memory_bytes", "Resident memory of the API process", rss)
        yield _gauge("vit_process_threads", "Threads in the API process", threads)


def _gauge(name, doc, value):
    family = GaugeMetricFamily(name, doc)
    family.add_metric([], value)
    return family


_system_collector = None


def register_system_collector(disk_path="."):
    """Idempotent: the app can start several times in one process (tests)."""
    global _system_collector
    if _system_collector is None:
        _system_collector = SystemCollector(disk_path)
        REGISTRY.register(_system_collector)
    else:
        _system_collector.disk_path = disk_path
