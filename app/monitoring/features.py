import numpy as np
from PIL import Image

# Image properties that describe *what the model is being fed*. If any of
# these move away from the training data (darker scenes, blur, sensor noise,
# a new region with different colours), the model's inputs have drifted even
# when nobody has labelled a single prediction yet.
FEATURE_NAMES = [
    "brightness",   # mean grey level, 0-1
    "contrast",     # std of the grey level
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation",   # mean HSV-style saturation
    "sharpness",    # mean |Laplacian|: drops with blur, rises with noise
]

_GREY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def image_features(image: Image.Image, size: int = 64) -> dict:
    """Cheap per-image statistics, computed on a small fixed-size copy so the
    numbers do not depend on the upload's resolution."""
    rgb = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    grey = rgb @ _GREY

    cmax, cmin = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.where(cmax > 0, (cmax - cmin) / np.maximum(cmax, 1e-6), 0.0)

    laplacian = (grey[1:-1, :-2] + grey[1:-1, 2:] + grey[:-2, 1:-1] + grey[2:, 1:-1]
                 - 4 * grey[1:-1, 1:-1])

    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "red_mean": float(rgb[..., 0].mean()),
        "green_mean": float(rgb[..., 1].mean()),
        "blue_mean": float(rgb[..., 2].mean()),
        "saturation": float(saturation.mean()),
        "sharpness": float(np.abs(laplacian).mean()),
    }


def prediction_entropy(probs: np.ndarray) -> float:
    """Normalised entropy of the softmax output: 0 = certain, 1 = uniform."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))
