import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

WEIGHTS_FILE = "model_int8.pt"
META_FILE = "model_meta.json"


def quantize_model(model):
    """Post-training dynamic quantization: every nn.Linear gets int8 weights
    (activations are quantized on the fly at inference time). Linear layers
    hold ~99% of a ViT's parameters, so this shrinks the model roughly 4x.
    Returns a new CPU-only model; the input model is left untouched."""
    model.eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def state_dict_size_mb(model) -> float:
    """Size of the serialized weights, i.e. what the model costs on disk and in RAM."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024 ** 2


def save_artifact(quantized_model, class_names, image_size, image_mean, image_std, out_dir, extra=None):
    """Writes everything the inference service needs, and nothing it doesn't:
    the int8 weights plus a JSON file with the architecture, labels and
    preprocessing stats. No pretrained download or LoRA code is needed to load it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), out_dir / WEIGHTS_FILE)
    meta = {
        "vit_config": quantized_model.config.to_dict(),
        "class_names": list(class_names),
        "image_size": image_size,
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        **(extra or {}),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_artifact(model_dir):
    """Rebuilds the quantized model saved by `save_artifact`. Returns (model, meta)."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    model = ViTForImageClassification(ViTConfig.from_dict(meta["vit_config"]))
    model = quantize_model(model)
    # Quantized layers keep their weights in packed params, which the
    # weights_only unpickler rejects; this file is our own build output.
    state_dict = torch.load(model_dir / WEIGHTS_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta
