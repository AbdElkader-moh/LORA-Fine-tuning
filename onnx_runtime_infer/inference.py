import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


class ONNXRuntimeInference:
    """Loads an exported ONNX model and runs inference with ONNX Runtime."""

    def __init__(self, onnx_path, providers=None):
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_logits(self, pixel_values: np.ndarray) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values.astype(np.float32))
        return self.session.run([self.output_name], {self.input_name: pixel_values})[0]

    def evaluate(self, loader, criterion=None):
        criterion = criterion or nn.CrossEntropyLoss()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_labels = [], []

        for images, labels in loader:
            logits = self.predict_logits(images.numpy())
            logits_t = torch.from_numpy(logits)
            loss = criterion(logits_t, labels)
            running_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            all_preds.extend(logits_t.argmax(dim=1).numpy())
            all_labels.extend(labels.numpy())

        avg_loss = running_loss / n_samples
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, acc, f1
