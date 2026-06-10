import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ["get_eval_transforms", "get_train_transforms"]


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Augmentations applied during training."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic transforms for validation during training."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
