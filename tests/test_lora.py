import torch
import torch.nn as nn

from vit_lora.lora import LoRALinear, apply_lora, freeze_for_lora, merge_lora

LORA_KWARGS = dict(layer_indices=[0, 1], target_projections=["query", "value"], r=4, alpha=8)


def randomize_lora_b(model):
    """lora_B starts at zero (a no-op update); give it values so the adapter actually does something."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_B, std=0.1)


def test_lora_is_a_noop_at_init(tiny_vit):
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = tiny_vit(pixel_values=x).logits
        after = apply_lora(tiny_vit, **LORA_KWARGS)(pixel_values=x).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_only_lora_and_classifier_are_trainable(tiny_vit):
    model = freeze_for_lora(apply_lora(tiny_vit, **LORA_KWARGS))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name or "classifier" in name for name in trainable)
    # 2 layers x 2 projections x (A, B) + classifier weight and bias
    assert len(trainable) == 2 * 2 * 2 + 2


def test_merge_lora_preserves_outputs_and_removes_wrappers(tiny_vit):
    model = apply_lora(tiny_vit, **LORA_KWARGS)
    randomize_lora_b(model)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        with_adapters = model(pixel_values=x).logits
        merged = merge_lora(model)
        after_merge = merged(pixel_values=x).logits

    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(with_adapters, after_merge, atol=1e-5)
