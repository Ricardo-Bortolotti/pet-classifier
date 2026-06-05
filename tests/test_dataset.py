from pathlib import Path

import pytest
from PIL import Image

from training.dataset import ImageFolderDataset, create_dataloaders


def _create_sample_images(base: Path, class_name: str, count: int) -> None:
    class_dir = base / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (64, 64), color=(i * 10, 50, 100)).save(class_dir / f"{i}.jpg")


@pytest.fixture
def split_dataset(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _create_sample_images(train_dir, "cats", 8)
    _create_sample_images(train_dir, "dogs", 8)
    _create_sample_images(test_dir, "cats", 4)
    _create_sample_images(test_dir, "dogs", 4)
    return tmp_path


def test_image_folder_dataset_loads_samples(split_dataset: Path):
    dataset = ImageFolderDataset(split_dataset / "train")
    assert len(dataset) == 16
    assert dataset.class_names == ["cats", "dogs"]


def test_create_dataloaders_split_layout(split_dataset: Path):
    loaders = create_dataloaders(
        split_dataset,
        batch_size=4,
        image_size=64,
        val_ratio=0.25,
        seed=42,
        num_workers=0,
    )

    assert loaders.class_names == ["cats", "dogs"]
    assert len(loaders.train.dataset) == 12
    assert len(loaders.val.dataset) == 4
    assert len(loaders.test.dataset) == 8

    images, labels = next(iter(loaders.train))
    assert images.shape[0] <= 4
    assert labels.shape[0] == images.shape[0]


def test_create_dataloaders_reproducible_split(split_dataset: Path):
    loaders_a = create_dataloaders(
        split_dataset,
        batch_size=2,
        image_size=64,
        val_ratio=0.25,
        seed=42,
        num_workers=0,
    )
    loaders_b = create_dataloaders(
        split_dataset,
        batch_size=2,
        image_size=64,
        val_ratio=0.25,
        seed=42,
        num_workers=0,
    )
    loaders_c = create_dataloaders(
        split_dataset,
        batch_size=2,
        image_size=64,
        val_ratio=0.25,
        seed=99,
        num_workers=0,
    )

    val_indices_a = loaders_a.val.dataset.indices
    val_indices_b = loaders_b.val.dataset.indices
    val_indices_c = loaders_c.val.dataset.indices

    assert val_indices_a == val_indices_b
    assert val_indices_a != val_indices_c
