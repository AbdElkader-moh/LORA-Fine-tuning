import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer and adds a trainable low-rank update
    on top of it, while freezing the original layer's weight and bias.

    forward(x) = base_linear(x) + scaling * (x @ A^T @ B^T)
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_update

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def get_vit_layers(model):
    """Returns the sequence of ViT transformer blocks, regardless of whether the
    installed transformers version nests them under `.encoder.layer` (< 5.0) or
    exposes them directly as `.layers` (>= 5.0)."""
    vit = model.vit
    if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        return vit.encoder.layer
    if hasattr(vit, "layers"):
        return vit.layers
    raise AttributeError(
        "Could not locate the ViT transformer blocks on this model. "
        "Run `print(model.vit)` to inspect its structure and adjust get_vit_layers()."
    )


def get_projection(layer, proj_name):
    """Given one transformer block and a logical projection name
    ("query", "key", "value", or "attn_output"), returns (parent_module, attr_name)."""
    attn = layer.attention
    new_names = {"query": "q_proj", "key": "k_proj", "value": "v_proj", "attn_output": "o_proj"}
    if hasattr(attn, new_names[proj_name]):
        return attn, new_names[proj_name]
    if proj_name in ("query", "key", "value"):
        return attn.attention, proj_name
    return attn.output, "dense"


def apply_lora(model, layer_indices, target_projections, r, alpha):
    """Replaces the chosen projections of the chosen transformer blocks with
    LoRALinear wrappers."""
    layers = get_vit_layers(model)
    for idx in layer_indices:
        layer = layers[idx]
        for proj_name in target_projections:
            parent, attr = get_projection(layer, proj_name)
            original_linear = getattr(parent, attr)
            if isinstance(original_linear, LoRALinear):
                original_linear = original_linear.base_linear
            wrapped = LoRALinear(original_linear, r=r, alpha=alpha)
            setattr(parent, attr, wrapped)
    return model


def freeze_for_lora(model):
    """Freezes everything except LoRA's A/B matrices and the classification head."""
    for name, param in model.named_parameters():
        if "lora_" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def merge_lora(model):
    """Folds every LoRALinear's low-rank update into its frozen base weight
    (W' = W + scaling * B @ A) and swaps the wrapper back for the plain
    nn.Linear. The merged model computes the same function with no LoRA
    modules left, so it can be quantized/exported like a stock ViT."""
    wrapped = [(parent, name, child)
               for parent in model.modules()
               for name, child in parent.named_children()
               if isinstance(child, LoRALinear)]
    for parent, name, lora_layer in wrapped:
        with torch.no_grad():
            lora_layer.base_linear.weight += lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)
        setattr(parent, name, lora_layer.base_linear)
    return model
