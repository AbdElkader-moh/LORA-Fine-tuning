import time

import numpy as np
import torch
from PIL import Image

from vit_lora.quantization import load_artifact


class Predictor:
    """Loads the quantized ViT once and classifies PIL images with it."""

    def __init__(self, model_dir):
        self.model, meta = load_artifact(model_dir)
        self.class_names = meta["class_names"]
        self.image_size = meta["image_size"]
        self.mean = np.array(meta["image_mean"], dtype=np.float32)
        self.std = np.array(meta["image_std"], dtype=np.float32)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Same steps as the eval transform used in training (resize ->
        scale to [0, 1] -> normalize), without needing torchvision."""
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        pixel_values = self.preprocess(image)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(pixel_values=pixel_values).logits[0]
        inference_ms = 1000 * (time.perf_counter() - start)

        probs = torch.softmax(logits, dim=0)
        top = torch.topk(probs, k=min(top_k, len(self.class_names)))
        predictions = [{"label": self.class_names[i], "confidence": round(p, 4)}
                       for p, i in zip(top.values.tolist(), top.indices.tolist())]
        return {
            "label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "inference_ms": round(inference_ms, 1),
        }
