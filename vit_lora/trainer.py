from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .engine import evaluate_model, train_one_epoch
from .model import trainable_state_dict


class Trainer:
    """Owns the training process: optimizer/scheduler steps, per-epoch
    checkpointing, and best-model tracking by validation loss.

    After each epoch it writes `checkpoint_epoch{N}.pt` containing the
    trainable parameters, optimizer state, scheduler state, epoch number,
    and validation loss/accuracy. Whenever validation loss improves it also
    writes `best_checkpoint.pt` (full, resumable) and `best_lora_weights.pt`
    (just the LoRA + classifier-head weights, for deployment/evaluation).
    """

    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.checkpoint_dir = Path(config.checkpoint.dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        self.best_lora_weights_path = self.checkpoint_dir / "best_lora_weights.pt"

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.training.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    def train(self):
        epochs = self.config.training.epochs
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.device, desc=f"epoch {epoch}/{epochs}",
            )
            val_loss, val_acc, val_f1 = evaluate_model(
                self.model, self.val_loader, self.criterion, self.device
            )
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            self._save_epoch_checkpoint(epoch, val_loss, val_acc)

        return self.history

    def _save_epoch_checkpoint(self, epoch, val_loss, val_acc):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainable_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lora_config": asdict(self.config.lora),
        }
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch{epoch}.pt")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"  -> new best val_loss ({val_loss:.4f}); saving best checkpoint")
            torch.save(checkpoint, self.best_checkpoint_path)
            torch.save({
                "epoch": epoch,
                "lora_state_dict": checkpoint["model_state_dict"],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lora_config": checkpoint["lora_config"],
            }, self.best_lora_weights_path)
