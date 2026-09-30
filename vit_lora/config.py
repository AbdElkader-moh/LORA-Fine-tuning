from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    data_root: str
    data_url: str
    val_size: float
    test_size: float
    seed: int
    max_per_class: Optional[int] = None


@dataclass
class ModelConfig:
    name: str


@dataclass
class LoRAConfig:
    layer_indices: List[int]
    target_projections: List[str]
    r: int
    alpha: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    num_workers: int


@dataclass
class CheckpointConfig:
    dir: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        lora=LoRAConfig(**raw["lora"]),
        training=TrainingConfig(**raw["training"]),
        checkpoint=CheckpointConfig(**raw["checkpoint"]),
    )
