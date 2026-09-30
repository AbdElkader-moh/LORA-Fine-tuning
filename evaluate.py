import argparse
from pathlib import Path

import torch

from vit_lora.config import load_config
from vit_lora.data import EuroSATData
from vit_lora.evaluation import evaluate_checkpoint
from vit_lora.processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <checkpoint.dir>/best_lora_weights.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or str(Path(config.checkpoint.dir) / "best_lora_weights.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    processor = ImageProcessor(config.model.name)
    data = EuroSATData(config, processor).prepare()
    _, _, test_loader = data.get_loaders(config.training.batch_size, config.training.num_workers)

    result = evaluate_checkpoint(
        config, checkpoint_path,
        data.class_names, data.class_to_idx, data.idx_to_class,
        test_loader, device,
    )
    print(f"\nLoaded adapter from epoch {result['epoch']}")
    print(f"Test set -> loss: {result['loss']:.4f}  accuracy: {result['acc']:.4f}  macro-F1: {result['f1']:.4f}")


if __name__ == "__main__":
    main()
