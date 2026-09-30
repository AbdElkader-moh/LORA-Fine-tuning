import argparse
from pathlib import Path

import torch
import torch.nn as nn

from vit_lora.config import load_config
from vit_lora.data import EuroSATData
from vit_lora.evaluation import load_best_model
from vit_lora.processor import ImageProcessor


class LogitsOnly(nn.Module):
    """Wraps the HF model so the exported graph takes/returns plain tensors
    (pixel_values -> logits) instead of a ModelOutput object."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <checkpoint.dir>/best_lora_weights.pt")
    parser.add_argument("--output", default="onnx_models/vit_lora_best.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or str(Path(config.checkpoint.dir) / "best_lora_weights.pt")

    device = torch.device("cpu")  # export on CPU for portability

    processor = ImageProcessor(config.model.name)
    data = EuroSATData(config, processor).prepare()

    model, checkpoint = load_best_model(
        config, checkpoint_path, data.class_names, data.class_to_idx, data.idx_to_class, device
    )
    print(f"Loaded best LoRA model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")

    wrapped = LogitsOnly(model)
    wrapped.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, processor.image_size, processor.image_size, device=device)
    torch.onnx.export(
        wrapped,
        dummy_input,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
