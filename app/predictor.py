import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vit_lora.quantization import WEIGHTS_FILE, load_artifact


class Predictor:
    """Loads the quantized ViT once and classifies PIL images with it."""

    def __init__(self, model_dir):
        self.model, meta = load_artifact(model_dir)
        self.class_names = meta["class_names"]
        self.image_size = meta["image_size"]
        self.mean = np.array(meta["image_mean"], dtype=np.float32)
        self.std = np.array(meta["image_std"], dtype=np.float32)
        # Content hash of the weights: ties every logged prediction to the exact model.
        self.version = hashlib.sha256((Path(model_dir) / WEIGHTS_FILE).read_bytes()).hexdigest()[:12]

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Same steps as the eval transform used in training (resize ->
        scale to [0, 1] -> normalize), without needing torchvision."""
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)

    def predict_proba(self, image: Image.Image):
        """Returns (class probabilities as a numpy array, forward-pass time in ms)."""
        pixel_values = self.preprocess(image)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(pixel_values=pixel_values).logits[0]
        inference_ms = 1000 * (time.perf_counter() - start)
        return torch.softmax(logits, dim=0).numpy(), inference_ms

    def format_result(self, probs: np.ndarray, inference_ms: float, top_k: int = 3) -> dict:
        order = np.argsort(probs)[::-1][:min(top_k, len(self.class_names))]
        predictions = [{"label": self.class_names[i], "confidence": round(float(probs[i]), 4)}
                       for i in order]
        return {
            "label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "inference_ms": round(inference_ms, 1),
        }

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        probs, inference_ms = self.predict_proba(image)
        return self.format_result(probs, inference_ms, top_k)
