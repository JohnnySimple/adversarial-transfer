"""MNIST loading, the fixed train/validation split, and the fixed evaluation set.

Two invariants from CLAUDE.md are enforced here, because every downstream measurement
depends on them.

**Pixel space.** The only transform is ``ToTensor``, so tensors carry raw [0, 1] pixels.
Normalization is a layer inside each model (``models.normalization``), never a dataset
transform. This is what makes ``epsilon`` mean the same thing in pixel units for every
architecture; moving normalization into the transform would silently rescale the
threat model per model and invalidate the whole transfer matrix.

**Fixed partitions.** The 55k/5k split is drawn from ``config.SPLIT_SEED`` alone, never
from a model seed, so all 15 training runs see the identical partition. The evaluation
set is the first ``config.EVAL_SET_SIZE`` test images in canonical order -- no shuffling,
no resampling -- so a given ``image_idx`` refers to the same image in every record of
every run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from advtransfer import config, seeds

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: The sole transform: PIL image -> float32 tensor of shape (1, 28, 28) in [0, 1].
#: Deliberately contains no Normalize step. See the module docstring.
PIXEL_TRANSFORM = transforms.ToTensor()

#: One MNIST example as yielded by the dataset: a (1, 28, 28) image and an integer label.
MnistItem = tuple[torch.Tensor, int]
MnistSubset = Subset[MnistItem]


def _mnist(*, train: bool, download: bool = True) -> MNIST:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return MNIST(
        root=str(config.DATA_DIR),
        train=train,
        download=download,
        transform=PIXEL_TRANSFORM,
    )


def split_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return the (train, validation) index arrays for the MNIST training set.

    Derived from an explicit permutation under ``config.SPLIT_SEED`` rather than
    ``random_split``, so the partition is auditable and independent of torch's RNG
    semantics across versions.
    """
    rng = np.random.default_rng(config.SPLIT_SEED)
    permutation = rng.permutation(config.MNIST_TRAIN_TOTAL)
    train_idx = permutation[: config.TRAIN_SIZE]
    val_idx = permutation[config.TRAIN_SIZE : config.TRAIN_SIZE + config.VAL_SIZE]
    if len(train_idx) != config.TRAIN_SIZE or len(val_idx) != config.VAL_SIZE:
        raise RuntimeError("split produced unexpected sizes; check config split constants")
    return train_idx, val_idx


def train_val_datasets(*, download: bool = True) -> tuple[MnistSubset, MnistSubset]:
    """Return the fixed 55k training subset and 5k validation subset."""
    full = _mnist(train=True, download=download)
    train_idx, val_idx = split_indices()
    return Subset(full, train_idx.tolist()), Subset(full, val_idx.tolist())


def eval_dataset(*, download: bool = True) -> MnistSubset:
    """Return the fixed evaluation set: the first N test images in canonical order."""
    test = _mnist(train=False, download=download)
    return Subset(test, list(range(config.EVAL_SET_SIZE)))


def train_dataloader(seed: int, *, download: bool = True) -> DataLoader[MnistItem]:
    """Return a shuffled training DataLoader whose shuffle order is fixed by ``seed``."""
    train, _ = train_val_datasets(download=download)
    return DataLoader(
        train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=seeds.torch_generator(seed),
        worker_init_fn=seeds.dataloader_worker_init,
        num_workers=config.DATALOADER_WORKERS,
        drop_last=False,
    )


def eval_dataloader(*, download: bool = True) -> DataLoader[MnistItem]:
    """Return an unshuffled DataLoader over the fixed evaluation set."""
    return DataLoader(
        eval_dataset(download=download),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER_WORKERS,
    )


def as_tensors(
    dataset: Dataset[MnistItem], device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a dataset as ``(images, labels)`` tensors in canonical order.

    The evaluation set is 2,000 images, so holding it as one batch keeps the attack and
    gradient code simple and keeps per-image record indices aligned with dataset order.
    """
    loader: DataLoader[MnistItem] = DataLoader(
        dataset,
        batch_size=len(dataset),  # type: ignore[arg-type]
        shuffle=False,
    )
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


def eval_tensors(
    device: str | torch.device = "cpu", *, download: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed evaluation set as ``(images, labels)`` tensors.

    ``images`` has shape ``(EVAL_SET_SIZE, 1, 28, 28)`` in [0, 1]; ``labels`` has shape
    ``(EVAL_SET_SIZE,)``. Row ``i`` is test image ``i`` for every model and every run.
    """
    images, labels = as_tensors(eval_dataset(download=download), device)
    assert_pixel_space(images)
    return images, labels


def assert_pixel_space(images: torch.Tensor) -> None:
    """Fail loudly if ``images`` is not in [0, 1] pixel space.

    A normalized tensor reaching the attack code would mean epsilon is being applied in
    the wrong units, which is a threat-model violation rather than a numerical nuisance,
    so this is checked rather than assumed.
    """
    low = float(images.min())
    high = float(images.max())
    if low < config.PIXEL_MIN - config.EPS_BALL_TOLERANCE:
        raise ValueError(f"images below pixel space: min={low}; normalization leaked into data")
    if high > config.PIXEL_MAX + config.EPS_BALL_TOLERANCE:
        raise ValueError(f"images above pixel space: max={high}; normalization leaked into data")
