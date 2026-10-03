"""Builds the drift reference profile the API compares live traffic against.

* Input features come from the training split: that is the data the model learned from.
* Model outputs (predicted-class mix, confidence) and the baseline accuracy come
  from the validation split, because on training images the model is
  over-confident and would make every production window look drifted.

Writes <model-dir>/reference_profile.json next to the quantized model.
"""
import argparse
import time
from collections import Counter

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from app.monitoring.drift import ReferenceProfile
from app.monitoring.features import FEATURE_NAMES, image_features
from app.monitoring.performance import classification_metrics
from app.predictor import Predictor
from vit_lora.config import load_config
from vit_lora.data import EuroSATData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--max-val", type=int, default=None, help="Cap on validation images to run the model on")
    args = parser.parse_args()

    config = load_config(args.config)
    data = EuroSATData(config, processor=None).prepare()  # only the split is needed, not the transforms

    features = {name: [] for name in FEATURE_NAMES}
    for path in tqdm(data.train_files, desc="Train features"):
        with Image.open(path) as image:
            for name, value in image_features(image).items():
                features[name].append(round(value, 5))

    predictor = Predictor(args.model_dir)
    val = list(zip(data.val_files, data.val_labels))[:args.max_val]
    confidences, pairs = [], []
    start = time.perf_counter()
    for path, label in tqdm(val, desc="Validation predictions"):
        with Image.open(path) as image:
            probs, _ = predictor.predict_proba(image)
        top = int(probs.argmax())
        confidences.append(round(float(probs[top]), 5))
        pairs.append((predictor.class_names[top], data.idx_to_class[label]))
    perf = classification_metrics(pairs, predictor.class_names)

    profile = ReferenceProfile({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": predictor.version,
        "n_train": len(data.train_files),
        "n_val": len(val),
        "features": features,
        "confidence": confidences,
        "label_counts": dict(Counter(p for p, _ in pairs)),
        "accuracy": perf["accuracy"],
        "macro_f1": perf["macro_f1"],
    })
    path = profile.save(args.model_dir)

    print(f"Reference profile -> {path}")
    print(f"  {len(data.train_files)} training images, {len(val)} validation images "
          f"({1000 * (time.perf_counter() - start) / len(val):.0f} ms/image)")
    print(f"  validation accuracy {perf['accuracy']:.4f}, macro-F1 {perf['macro_f1']:.4f}, "
          f"mean confidence {np.mean(confidences):.4f}")
    for name in FEATURE_NAMES:
        values = np.array(features[name])
        print(f"  {name:<11} mean {values.mean():.4f}  std {values.std():.4f}")


if __name__ == "__main__":
    main()
