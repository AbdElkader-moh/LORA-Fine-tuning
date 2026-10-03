"""Generates dashboards/vit-observability.json. Edit this file, not the JSON:

    py observability/grafana/make_dashboard.py
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
PSI = [(None, "green"), (0.1, "orange"), (0.25, "red")]
GOOD_HIGH = [(None, "red"), (0.85, "orange"), (0.92, "green")]
PREDICT = 'path="/predict"'
REAL_FS = 'fstype!~"tmpfs|rootfs|fuse.*|overlay"'


class Layout:
    def __init__(self):
        self.panels, self.y = [], 0

    def row(self, title):
        self.panels.append({"type": "row", "title": title, "collapsed": False, "panels": [],
                            "gridPos": {"h": 1, "w": 24, "x": 0, "y": self.y}})
        self.y += 1

    def add(self, kind, title, targets, x, w, h=8, dy=0, unit=None, thresholds=None, min=None, max=None,
            description=None, options=None, defaults=None, overrides=None, instant=False):
        field = dict(defaults or {})
        if unit:
            field["unit"] = unit
        if min is not None:
            field["min"] = min
        if max is not None:
            field["max"] = max
        if thresholds:
            field["thresholds"] = {"mode": "absolute",
                                   "steps": [{"color": c, "value": v} for v, c in thresholds]}
            if kind == "timeseries":
                field.setdefault("custom", {})["thresholdsStyle"] = {"mode": "dashed"}
        options = dict(options or {})
        if kind == "timeseries":
            options.setdefault("legend", {"displayMode": "list", "placement": "bottom"})
            options.setdefault("tooltip", {"mode": "multi"})
        panel = {
            "type": kind, "title": title, "datasource": DS,
            "gridPos": {"h": h, "w": w, "x": x, "y": self.y + dy},
            "fieldConfig": {"defaults": field, "overrides": overrides or []},
            "options": options,
            "targets": [{"datasource": DS, "refId": chr(65 + i), "expr": expr, "legendFormat": legend,
                         **({"instant": True, "format": "table"} if instant else {})}
                        for i, (expr, legend) in enumerate(targets)],
        }
        if description:
            panel["description"] = description
        self.panels.append(panel)
        return panel

    def next_row(self, h):
        self.y += h


def build():
    d = Layout()
    bars = {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True}

    d.row("Service health")
    d.add("stat", "API", [('up{job="vit-api"}', "")], 0, 3, 4, thresholds=[(None, "red"), (1, "green")],
          defaults={"mappings": [{"type": "value", "options": {"0": {"text": "DOWN"}, "1": {"text": "UP"}}}]})
    d.add("stat", "Predict requests / s", [(f"sum(rate(vit_http_requests_total{{{PREDICT}}}[1m]))", "")],
          3, 3, 4, unit="reqps")
    d.add("stat", "p95 /predict latency",
          [(f"histogram_quantile(0.95, sum by (le) (rate(vit_http_request_duration_seconds_bucket{{{PREDICT}}}[5m])))", "")],
          6, 3, 4, unit="s", thresholds=[(None, "green"), (0.5, "orange"), (1, "red")])
    d.add("stat", "5xx share",
          [('(sum(rate(vit_http_requests_total{status=~"5.."}[5m])) or vector(0))'
            ' / clamp_min(sum(rate(vit_http_requests_total[5m])), 1e-9)', "")],
          9, 3, 4, unit="percentunit", thresholds=[(None, "green"), (0.01, "orange"), (0.05, "red")])
    d.add("stat", "Predictions served", [("sum(vit_predictions_total)", "")], 12, 3, 4)
    d.add("stat", "Labels received", [("sum(vit_feedback_total)", "")], 15, 3, 4)
    alerts = d.add("table", "Firing alerts", [('ALERTS{alertstate="firing"}', "")], 18, 6, 4, instant=True)
    alerts["transformations"] = [{"id": "organize", "options": {"excludeByName": {
        "Time": True, "Value": True, "__name__": True, "alertstate": True, "instance": True, "job": True}}}]
    d.next_row(4)
    d.add("timeseries", "Request rate by endpoint and status",
          [("sum by (path, status) (rate(vit_http_requests_total[1m]))", "{{path}} {{status}}")], 0, 12, unit="reqps")
    d.add("timeseries", "Latency: /predict request vs model forward pass",
          [(f"histogram_quantile(0.5, sum by (le) (rate(vit_http_request_duration_seconds_bucket{{{PREDICT}}}[5m])))", "request p50"),
           (f"histogram_quantile(0.95, sum by (le) (rate(vit_http_request_duration_seconds_bucket{{{PREDICT}}}[5m])))", "request p95"),
           ("histogram_quantile(0.95, sum by (le) (rate(vit_inference_duration_seconds_bucket[5m])))", "model p95")],
          12, 12, unit="s")
    d.next_row(8)

    d.row("Model outputs (no labels needed)")
    d.add("timeseries", "Predicted class mix (drift window)", [("vit_window_prediction_share", "{{label}}")],
          0, 12, unit="percentunit", min=0, max=1,
          defaults={"custom": {"stacking": {"mode": "normal"}, "fillOpacity": 60, "lineWidth": 0}})
    d.add("timeseries", "Mean top-1 confidence", [("vit_window_mean_confidence", "live window"),
                                                  ("vit_reference_mean_confidence", "validation reference")],
          12, 6, unit="percentunit", max=1)
    d.add("timeseries", "Low-confidence share (< 0.5)",
          [("sum(rate(vit_low_confidence_predictions_total[5m])) / clamp_min(sum(rate(vit_predictions_total[5m])), 1e-9)", "share")],
          18, 6, unit="percentunit", min=0)
    d.next_row(8)

    d.row("Data drift (live inputs vs training data)")
    d.add("stat", "Drifted input features", [("vit_drifted_features_share", "")], 0, 4, 5, unit="percentunit",
          thresholds=[(None, "green"), (0.01, "orange"), (0.3, "red")],
          description="Share of input features with PSI >= 0.25 over the last DRIFT_WINDOW predictions.")
    d.add("stat", "Prediction drift (PSI)", [("vit_prediction_drift_psi", "")], 0, 4, 5, dy=5, thresholds=PSI,
          description="PSI of the predicted-class mix vs the validation split.")
    d.add("bargauge", "PSI per input feature", [("vit_feature_drift_psi", "{{feature}}")], 4, 8, 10,
          thresholds=PSI, min=0, max=1, options=bars,
          description="Population Stability Index: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 drift.")
    d.add("timeseries", "PSI over time", [("vit_feature_drift_psi", "{{feature}}"),
                                          ("vit_prediction_drift_psi", "predicted class"),
                                          ("vit_confidence_drift_psi", "confidence")],
          12, 12, 10, thresholds=PSI, min=0)
    d.next_row(10)
    d.add("timeseries", "Input feature means: live window vs training reference",
          [('vit_feature_mean{window="current"}', "{{feature}} live"),
           ('vit_feature_mean{window="reference"}', "{{feature}} reference")], 0, 24, 8,
          options={"legend": {"displayMode": "table", "placement": "right"}},
          overrides=[{"matcher": {"id": "byRegexp", "options": ".*reference"},
                      "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [6, 6]}}]}])
    d.next_row(8)

    d.row("Model performance (from ground-truth feedback)")
    d.add("stat", "Live accuracy", [("vit_window_accuracy", "")], 0, 4, 5, unit="percentunit", thresholds=GOOD_HIGH)
    d.add("stat", "Live macro-F1", [("vit_window_macro_f1", "")], 4, 4, 5, unit="percentunit", thresholds=GOOD_HIGH)
    d.add("stat", "Baseline accuracy", [("vit_baseline_accuracy", "")], 8, 4, 5, unit="percentunit",
          description="Accuracy of the served model on the validation split when the reference was built.")
    d.add("stat", "Labelled samples in window", [("vit_window_labelled_samples", "")], 12, 4, 5)
    d.add("timeseries", "Live accuracy vs baseline", [("vit_window_accuracy", "live accuracy"),
                                                      ("vit_window_macro_f1", "live macro-F1"),
                                                      ("vit_baseline_accuracy", "baseline accuracy")],
          0, 8, 5, dy=5, unit="percentunit", max=1)
    d.add("timeseries", "Feedback per minute", [("sum by (correct) (rate(vit_feedback_total[1m]) * 60)", "correct={{correct}}")],
          8, 8, 5, dy=5)
    d.add("bargauge", "Recall per class (window)", [("vit_window_class_recall", "{{label}}")], 16, 8, 10,
          unit="percentunit", thresholds=GOOD_HIGH, min=0, max=1, options=bars)
    d.next_row(10)

    d.row("Infrastructure")
    d.add("timeseries", "Host CPU", [('100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])))', "node-exporter"),
                                     ("vit_system_cpu_percent", "seen by API")], 0, 8, unit="percent", min=0, max=100)
    d.add("timeseries", "Host memory used",
          [("100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)", "node-exporter"),
           ("vit_system_memory_used_percent", "seen by API")], 8, 8, unit="percent", min=0, max=100)
    d.add("timeseries", "Disk used",
          [(f"max(100 * (1 - node_filesystem_avail_bytes{{{REAL_FS}}} / node_filesystem_size_bytes{{{REAL_FS}}}))",
            "fullest disk (node)"),
           ("vit_system_disk_used_percent", "prediction-log volume")], 16, 8, unit="percent", min=0, max=100)
    d.next_row(8)
    d.add("timeseries", "Container CPU (cores)",
          [('sum by (name) (rate(container_cpu_usage_seconds_total{name=~".+"}[1m]))', "{{name}}"),
           ("vit_process_cpu_percent / 100", "api process")], 0, 8, min=0)
    d.add("timeseries", "Container memory",
          [('sum by (name) (container_memory_working_set_bytes{name=~".+"})', "{{name}}"),
           ("vit_process_resident_memory_bytes", "api process RSS")], 8, 8, unit="bytes", min=0)
    d.add("timeseries", "Host network I/O",
          [('sum(rate(node_network_receive_bytes_total{device!="lo"}[1m]))', "receive"),
           ('-sum(rate(node_network_transmit_bytes_total{device!="lo"}[1m]))', "transmit")], 16, 8, unit="Bps")
    d.next_row(8)

    for i, panel in enumerate(d.panels, start=1):
        panel["id"] = i
    return {
        "uid": "vit-observability", "title": "ViT-LoRA AI observability", "tags": ["ml", "observability"],
        "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"}, "panels": d.panels,
        "annotations": {"list": []}, "templating": {"list": []},
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "dashboards" / "vit-observability.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
