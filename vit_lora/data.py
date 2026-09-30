import random
import zipfile
from pathlib import Path

import requests
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class EuroSATDataset(Dataset):
    """Given a list of file paths and labels, loads + preprocesses one image at a time."""

    def __init__(self, file_paths, labels, transform):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


class EuroSATData:
    """Downloads/extracts the EuroSAT RGB dataset, builds the file/label lists,
    and splits them into train/val/test. Call `prepare()` once before
    `get_loaders()`."""

    def __init__(self, config, processor: "ImageProcessor"):
        self.config = config.data
        self.processor = processor

        self.class_names = None
        self.class_to_idx = None
        self.idx_to_class = None

        self.train_files = self.train_labels = None
        self.val_files = self.val_labels = None
        self.test_files = self.test_labels = None

    def prepare(self) -> "EuroSATData":
        data_root = Path(self.config.data_root)
        zip_path = data_root / "EuroSAT_RGB.zip"
        extract_dir = data_root / "EuroSAT_RGB"
        data_root.mkdir(parents=True, exist_ok=True)

        self._download(self.config.data_url, zip_path)
        self._extract(zip_path, extract_dir, data_root)

        self.class_names = sorted(d.name for d in extract_dir.iterdir() if d.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        rng = random.Random(self.config.seed)
        all_files, all_labels = [], []
        for class_name in self.class_names:
            class_files = sorted((extract_dir / class_name).glob("*.jpg"))
            if self.config.max_per_class and len(class_files) > self.config.max_per_class:
                class_files = rng.sample(class_files, self.config.max_per_class)
            for img_path in class_files:
                all_files.append(str(img_path))
                all_labels.append(self.class_to_idx[class_name])

        holdout_size = self.config.val_size + self.config.test_size
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, all_labels,
            test_size=holdout_size,
            random_state=self.config.seed,
            stratify=all_labels,
        )
        relative_test_size = self.config.test_size / holdout_size
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels,
            test_size=relative_test_size,
            random_state=self.config.seed,
            stratify=temp_labels,
        )

        self.train_files, self.train_labels = train_files, train_labels
        self.val_files, self.val_labels = val_files, val_labels
        self.test_files, self.test_labels = test_files, test_labels
        return self

    def get_loaders(self, batch_size: int, num_workers: int):
        train_ds = EuroSATDataset(self.train_files, self.train_labels, self.processor.train_transform)
        val_ds = EuroSATDataset(self.val_files, self.val_labels, self.processor.eval_transform)
        test_ds = EuroSATDataset(self.test_files, self.test_labels, self.processor.eval_transform)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader

    @staticmethod
    def _download(url: str, dest_path: Path):
        if dest_path.exists():
            return
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest_path.name}"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

    @staticmethod
    def _extract(zip_path: Path, extract_dir: Path, data_root: Path):
        if extract_dir.exists():
            return
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
