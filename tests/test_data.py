"""Data-layer guards.

The pixel-space test is the important one. If normalization ever leaks into the dataset
transform, every epsilon in the study silently means a different perturbation size per
model, and the transfer matrix becomes uninterpretable -- with no error raised anywhere.
"""

import numpy as np
import pytest
import torch

from advtransfer import config, data

pytestmark = pytest.mark.slow  # requires the MNIST download


def test_split_is_a_partition_of_the_training_set():
    train_idx, val_idx = data.split_indices()
    assert len(train_idx) == config.TRAIN_SIZE
    assert len(val_idx) == config.VAL_SIZE
    assert not set(train_idx.tolist()) & set(val_idx.tolist())
    assert len(set(train_idx.tolist()) | set(val_idx.tolist())) == (
        config.TRAIN_SIZE + config.VAL_SIZE
    )


def test_split_is_identical_on_every_call():
    a_train, a_val = data.split_indices()
    b_train, b_val = data.split_indices()
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_val, b_val)


def test_split_is_not_the_identity_permutation():
    train_idx, _ = data.split_indices()
    assert not np.array_equal(train_idx, np.arange(config.TRAIN_SIZE))


def test_eval_set_is_the_first_test_images_in_canonical_order():
    subset = data.eval_dataset()
    assert len(subset) == config.EVAL_SET_SIZE
    assert subset.indices == list(range(config.EVAL_SET_SIZE))


def test_eval_tensors_are_in_pixel_space_with_expected_shape():
    images, labels = data.eval_tensors()
    assert images.shape == (config.EVAL_SET_SIZE, 1, 28, 28)
    assert labels.shape == (config.EVAL_SET_SIZE,)
    assert images.dtype == torch.float32
    assert float(images.min()) >= config.PIXEL_MIN
    assert float(images.max()) <= config.PIXEL_MAX
    # A genuinely un-normalized MNIST batch reaches both extremes of the range.
    assert float(images.min()) == pytest.approx(0.0)
    assert float(images.max()) == pytest.approx(1.0)


def test_eval_tensors_are_stable_across_calls():
    first, first_labels = data.eval_tensors()
    second, second_labels = data.eval_tensors()
    assert torch.equal(first, second)
    assert torch.equal(first_labels, second_labels)


def test_transform_contains_no_normalization():
    # Asserted structurally as well as numerically: a Normalize step here would be a
    # threat-model violation, so it must be impossible to add without a test failing.
    from torchvision import transforms

    assert isinstance(data.PIXEL_TRANSFORM, transforms.ToTensor)


def test_assert_pixel_space_rejects_normalized_input():
    normalized = (torch.rand(4, 1, 28, 28) - config.MNIST_MEAN) / config.MNIST_STD
    with pytest.raises(ValueError, match="pixel space"):
        data.assert_pixel_space(normalized)


def test_assert_pixel_space_accepts_valid_input():
    data.assert_pixel_space(torch.rand(4, 1, 28, 28))


def test_train_dataloader_shuffle_order_is_seed_determined():
    def first_labels(seed: int) -> torch.Tensor:
        loader = data.train_dataloader(seed)
        _, labels = next(iter(loader))
        return labels

    assert torch.equal(first_labels(11), first_labels(11))
    assert not torch.equal(first_labels(11), first_labels(12))
