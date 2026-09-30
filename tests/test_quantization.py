import torch
import torch.nn as nn

from vit_lora.quantization import load_artifact, quantize_model, save_artifact, state_dict_size_mb


def test_quantization_replaces_linear_layers_and_shrinks_model(tiny_vit):
    quantized = quantize_model(tiny_vit)

    assert not any(type(m) is nn.Linear for m in quantized.modules())
    assert any(type(m) is nn.Linear for m in tiny_vit.modules()), "original model must be left untouched"
    assert state_dict_size_mb(quantized) < state_dict_size_mb(tiny_vit)


def test_quantized_model_stays_close_to_fp32(tiny_vit):
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        fp32_logits = tiny_vit(pixel_values=x).logits
        int8_logits = quantize_model(tiny_vit)(pixel_values=x).logits
    assert int8_logits.shape == fp32_logits.shape
    assert torch.allclose(fp32_logits, int8_logits, atol=0.1)


def test_artifact_roundtrip_gives_identical_predictions(tiny_vit, tmp_path):
    quantized = quantize_model(tiny_vit)
    save_artifact(quantized, ["a", "b", "c"], 32, [0.5] * 3, [0.5] * 3, tmp_path)
    loaded, meta = load_artifact(tmp_path)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(quantized(pixel_values=x).logits, loaded(pixel_values=x).logits)
    assert meta["class_names"] == ["a", "b", "c"]
    assert meta["image_size"] == 32
