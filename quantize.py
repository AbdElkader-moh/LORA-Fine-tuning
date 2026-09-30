import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from vit_lora.config import load_config
from vit_lora.data import EuroSATData
from vit_lora.engine import evaluate_model
from vit_lora.evaluation import load_best_model
from vit_lora.lora import merge_lora
from vit_lora.processor import ImageProcessor
from vit_lora.quantization import quantize_model, save_artifact, state_dict_size_mb


def timed_evaluate(model, loader, device):
    start = time.perf_counter()
    loss, acc, f1 = evaluate_model(model, loader, nn.CrossEntropyLoss(), device)
    ms_per_image = 1000 * (time.perf_counter() - start) / len(loader.dataset)
    return {"loss": loss, "accuracy": acc, "macro_f1": f1, "ms_per_image": ms_per_image}


def export_samples(data, samples_dir):
    """Copies one held-out test image per class into the repo so the API can
    be tried (and smoke-tested in CI) without downloading the dataset."""
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for path, label in zip(data.test_files, data.test_labels):
        if label not in seen:
            seen.add(label)
            shutil.copy(path, samples_dir / f"{data.idx_to_class[label]}.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <checkpoint.dir>/best_lora_weights.pt")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--samples-dir", default="samples")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or str(Path(config.checkpoint.dir) / "best_lora_weights.pt")
    device = torch.device("cpu")  # dynamic quantization is CPU-only

    processor = ImageProcessor(config.model.name)
    data = EuroSATData(config, processor).prepare()
    _, _, test_loader = data.get_loaders(config.training.batch_size, config.training.num_workers)

    model, checkpoint = load_best_model(
        config, checkpoint_path, data.class_names, data.class_to_idx, data.idx_to_class, device
    )
    print(f"Loaded best LoRA model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")

    model = merge_lora(model)
    fp32 = timed_evaluate(model, test_loader, device)
    fp32["size_mb"] = state_dict_size_mb(model)

    quantized = quantize_model(model)
    int8 = timed_evaluate(quantized, test_loader, device)
    int8["size_mb"] = state_dict_size_mb(quantized)

    report = {
        "base_model": config.model.name,
        "best_epoch": checkpoint["epoch"],
        "test_images": len(data.test_files),
        "fp32": fp32,
        "int8": int8,
        "size_reduction": fp32["size_mb"] / int8["size_mb"],
    }
    save_artifact(quantized, data.class_names, processor.image_size,
                  processor.image_mean, processor.image_std, args.output_dir)
    (Path(args.output_dir) / "quantization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    export_samples(data, args.samples_dir)

    print(f"\n{'':10}{'size (MB)':>12}{'accuracy':>12}{'macro-F1':>12}{'ms/image':>12}")
    for name, row in (("FP32", fp32), ("INT8", int8)):
        print(f"{name:10}{row['size_mb']:>12.1f}{row['accuracy']:>12.4f}"
              f"{row['macro_f1']:>12.4f}{row['ms_per_image']:>12.1f}")
    print(f"\nSize reduction: {report['size_reduction']:.2f}x")
    print(f"Saved quantized model to {args.output_dir}/")


if __name__ == "__main__":
    main()
