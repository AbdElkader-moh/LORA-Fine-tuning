import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from .engine import evaluate_model
from .lora import apply_lora


def load_best_model(config, checkpoint_path, class_names, class_to_idx, idx_to_class, device):
    """Rebuilds the model from the original pretrained ViT, re-creates LoRA
    layers using the config saved inside the checkpoint, and loads only the
    saved adapter (+ classification head) weights. Returns the ready-to-use
    model and the checkpoint dict (for its `epoch`/metrics metadata)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ViTForImageClassification.from_pretrained(
        config.model.name,
        num_labels=len(class_names),
        id2label=idx_to_class,
        label2id=class_to_idx,
        ignore_mismatched_sizes=True,
    )
    model = apply_lora(model, **checkpoint["lora_config"])

    missing, unexpected = model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
    # `missing` should only ever be the frozen pretrained weights (never
    # trained, never saved); `unexpected` should be empty.
    non_lora_missing = [k for k in missing if "lora_" in k or "classifier" in k]
    if non_lora_missing:
        raise RuntimeError(f"Failed to restore trainable params: {non_lora_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(config, checkpoint_path, class_names, class_to_idx, idx_to_class,
                         test_loader, device):
    """Loads the best saved LoRA model and evaluates it on `test_loader`."""
    model, checkpoint = load_best_model(
        config, checkpoint_path, class_names, class_to_idx, idx_to_class, device
    )
    criterion = nn.CrossEntropyLoss()
    loss, acc, f1 = evaluate_model(model, test_loader, criterion, device)
    return {"loss": loss, "acc": acc, "f1": f1, "epoch": checkpoint["epoch"]}
