from dataclasses import asdict

from transformers import ViTForImageClassification

from .lora import apply_lora, freeze_for_lora


def build_lora_model(config, class_names, class_to_idx, idx_to_class):
    """Loads the pretrained ViT specified in `config.model.name`, injects LoRA
    into the layers/projections specified in `config.lora`, and freezes
    everything except the LoRA matrices and the classification head."""
    model = ViTForImageClassification.from_pretrained(
        config.model.name,
        num_labels=len(class_names),
        id2label=idx_to_class,
        label2id=class_to_idx,
        ignore_mismatched_sizes=True,
    )
    model = apply_lora(model, **asdict(config.lora))
    model = freeze_for_lora(model)
    return model


def trainable_state_dict(model):
    """The pieces we actually train and the only ones worth persisting: the
    LoRA A/B matrices of the targeted blocks and the classification head. The
    frozen pretrained ViT weights are deliberately excluded."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_" in k or "classifier" in k}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
