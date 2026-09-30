import argparse
import random

import numpy as np
import torch

from vit_lora.config import load_config
from vit_lora.data import EuroSATData
from vit_lora.model import build_lora_model, count_parameters
from vit_lora.processor import ImageProcessor
from vit_lora.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    random.seed(config.data.seed)
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    processor = ImageProcessor(config.model.name)
    data = EuroSATData(config, processor).prepare()
    train_loader, val_loader, _ = data.get_loaders(config.training.batch_size, config.training.num_workers)
    print(f"train: {len(data.train_files)}  val: {len(data.val_files)}  test: {len(data.test_files)}")

    model = build_lora_model(config, data.class_names, data.class_to_idx, data.idx_to_class)
    total, trainable = count_parameters(model)
    print(f"total params: {total:,} | trainable params: {trainable:,} ({100 * trainable / total:.3f}%)")

    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()

    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best LoRA weights saved to: {trainer.best_lora_weights_path}")


if __name__ == "__main__":
    main()
