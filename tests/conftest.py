import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

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
    return out_dir
