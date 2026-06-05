from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from training.transforms import get_eval_transforms, get_train_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _collect_samples(
    root: Path,
    class_names: list[str] | None = None,
) -> tuple[list[tuple[Path, int]], list[str]]:
    root = Path(root)
    if not root.exists():
        return [], class_names or []

    if class_names is None:
        class_names = sorted(d.name for d in root.iterdir() if d.is_dir())

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []

    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        label = class_to_idx[class_name]
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    return samples, class_names


class ImageFolderDataset(Dataset):
    """Image classification dataset with Albumentations transforms.

    Supports flat or split layouts::

        data_dir/
            cats/
            dogs/

        data_dir/
            train/
                cats/
                dogs/
            test/
                cats/
                dogs/
    """

    def __init__(
        self,
        data_dir: Path,
        transform: A.Compose | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples, self.class_names = _collect_samples(self.data_dir, class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        image_path, label = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]


def _split_train_val(
    train_dir: Path,
    image_size: int,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset, list[str]]:
    train_source = ImageFolderDataset(
        train_dir,
        transform=get_train_transforms(image_size),
    )
    val_source = ImageFolderDataset(
        train_dir,
        transform=get_eval_transforms(image_size),
        class_names=train_source.class_names,
    )

    if len(train_source) == 0:
        raise ValueError(
            f"No training images found in '{train_dir}'. "
            "Expected subdirectories per class (e.g. cats/, dogs/)."
        )

    val_size = max(1, int(len(train_source) * val_ratio))
    train_size = len(train_source) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, _ = random_split(train_source, [train_size, val_size], generator=generator)
    _, val_dataset = random_split(val_source, [train_size, val_size], generator=generator)

    return train_dataset, val_dataset, train_source.class_names


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoaders:
    """Create reproducible train, validation and test DataLoaders.

    Uses ``data_dir/train`` for train/val split and ``data_dir/test`` for testing.
    Falls back to a flat layout when split folders are not present.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if train_dir.exists() and test_dir.exists():
        train_dataset, val_dataset, class_names = _split_train_val(
            train_dir,
            image_size=image_size,
            val_ratio=val_ratio,
            seed=seed,
        )
        test_dataset = ImageFolderDataset(
            test_dir,
            transform=get_eval_transforms(image_size),
            class_names=class_names,
        )
    else:
        train_dataset, val_dataset, class_names = _split_train_val(
            data_dir,
            image_size=image_size,
            val_ratio=val_ratio,
            seed=seed,
        )
        test_dataset = ImageFolderDataset(
            data_dir,
            transform=get_eval_transforms(image_size),
            class_names=class_names,
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    return DataLoaders(
        train=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        class_names=class_names,
    )
