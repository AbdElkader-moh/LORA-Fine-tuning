import torch
import numpy as np
import pytest
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification

from app.monitoring.drift import ReferenceProfile
from app.monitoring.features import FEATURE_NAMES, image_features
from vit_lora.quantization import quantize_model, save_artifact

CLASS_NAMES = ["Forest", "River", "SeaLake"]
IMAGE_SIZE = 32


def make_tiny_vit():
    """A randomly initialised 2-layer ViT: same module layout as the real
    model, but builds in milliseconds and needs no download."""
    torch.manual_seed(0)
    config = ViTConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        image_size=IMAGE_SIZE, patch_size=16, num_labels=len(CLASS_NAMES),
    )
    return ViTForImageClassification(config).eval()


@pytest.fixture
def tiny_vit():
    return make_tiny_vit()


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    """A saved quantized artifact in the exact format the API loads."""
    out_dir = tmp_path_factory.mktemp("artifacts")
    quantized = quantize_model(make_tiny_vit())
    save_artifact(quantized, CLASS_NAMES, IMAGE_SIZE, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], out_dir)
    make_reference_profile().save(out_dir)
    return out_dir


def textured_image(rng, scale=1.0):
    """A random mid-grey textured RGB image; `scale` < 1 darkens it."""
    pixels = np.clip(rng.normal(0.45, 0.12, (48, 48, 3)) * scale, 0, 1)
    return Image.fromarray((pixels * 255).astype(np.uint8))


def make_reference_profile(n=200):
    rng = np.random.default_rng(0)
    rows = [image_features(textured_image(rng)) for _ in range(n)]
    return ReferenceProfile({
        "features": {name: [r[name] for r in rows] for name in FEATURE_NAMES},
        "confidence": rng.uniform(0.33, 0.6, n).tolist(),
        "label_counts": {c: n // len(CLASS_NAMES) for c in CLASS_NAMES},
        "accuracy": 0.9,
        "macro_f1": 0.88,
    })
