import torchvision.transforms as T
from transformers import ViTImageProcessor


class ImageProcessor:
    """Builds the preprocessing pipeline for a given pretrained ViT checkpoint:
    resize/normalize using the checkpoint's own stats, with augmentation for
    training and a clean pass-through for validation/test."""

    def __init__(self, model_name: str):
        hf_processor = ViTImageProcessor.from_pretrained(model_name)
        self.image_mean = hf_processor.image_mean
        self.image_std = hf_processor.image_std
        size = hf_processor.size["height"]
        self.image_size = size
        normalize = T.Normalize(mean=hf_processor.image_mean, std=hf_processor.image_std)

        self.train_transform = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            normalize,
        ])

        self.eval_transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            normalize,
        ])
