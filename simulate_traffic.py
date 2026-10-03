"""Sends production-like traffic to the API so the monitoring has something to watch.

Images come from the held-out test split, so their true labels are known and
can be sent back through /feedback. Each phase can corrupt the images to mimic
a real-world shift (a darker season, a blurrier or noisier sensor, haze).

    py simulate_traffic.py --phases clean:200 dark:200 --label-rate 0.5
"""
import argparse
import io
import random

import numpy as np
import requests
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

from vit_lora.config import load_config
from vit_lora.data import EuroSATData


def _array_op(fn):
    def apply(image):
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray((np.clip(fn(pixels), 0, 1) * 255).astype(np.uint8))
    return apply


_rng = np.random.default_rng(0)
CORRUPTIONS = {
    "clean": None,
    "dark": _array_op(lambda x: x * 0.45),
    "bright": _array_op(lambda x: x * 0.6 + 0.4),
    "haze": _array_op(lambda x: 0.55 * x + 0.45 * 0.85),
    "noise": _array_op(lambda x: x + _rng.normal(0, 0.12, x.shape)),
    "blur": lambda image: image.filter(ImageFilter.GaussianBlur(2.5)),
    "grayscale": lambda image: image.convert("L").convert("RGB"),
}


def encode(path, corruption):
    if corruption is None:
        with open(path, "rb") as f:
            return f.read()
    with Image.open(path) as image:
        buffer = io.BytesIO()
        corruption(image.convert("RGB")).save(buffer, format="PNG")
        return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phases", nargs="+", default=["clean:200"],
                        help=f"<corruption>:<n_requests>, corruption in {sorted(CORRUPTIONS)}")
    parser.add_argument("--label-rate", type=float, default=0.5,
                        help="Fraction of predictions that get a ground-truth label back")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = EuroSATData(load_config(args.config), processor=None).prepare()
    pool = [(path, data.idx_to_class[label]) for path, label in zip(data.test_files, data.test_labels)]
    rng = random.Random(args.seed)
    session = requests.Session()

    for phase in args.phases:
        name, n = phase.split(":")
        corruption = CORRUPTIONS[name]
        correct = labelled = 0
        for _ in tqdm(range(int(n)), desc=f"{name:>9}"):
            path, true_label = rng.choice(pool)
            response = session.post(f"{args.url}/predict",
                                    files={"file": ("tile.png", encode(path, corruption))})
            response.raise_for_status()
            body = response.json()
            if rng.random() < args.label_rate:
                session.post(f"{args.url}/feedback",
                             json={"request_id": body["request_id"], "label": true_label}).raise_for_status()
                labelled += 1
                correct += body["label"] == true_label
        accuracy = f"{correct / labelled:.3f}" if labelled else "n/a"
        print(f"{name}: {n} requests, {labelled} labelled, accuracy on labelled {accuracy}")

    drift = session.get(f"{args.url}/monitoring/drift").json()
    perf = session.get(f"{args.url}/monitoring/performance").json()
    print(f"\nDrift status: {drift['status']}  drifted features: {drift.get('drifted_features')}")
    print(f"Window accuracy: {perf['accuracy']}  (baseline {perf['baseline_accuracy']})")


if __name__ == "__main__":
    main()
