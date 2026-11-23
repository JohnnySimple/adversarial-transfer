"""Deterministic seeding.

``seed_for`` is the only place a model's training seed is decided (CLAUDE.md section 4).
Two properties matter and are both covered by tests:

1. ``cnn_a`` and ``cnn_a_s2`` receive different seeds within every replicate. They share
   one architecture definition, so the seed is the *only* difference between them and the
   entire content of H5.
2. Every (model, replicate) seed is distinct across the whole study, so no two training
   runs can accidentally share an initialisation.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from advtransfer import config

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def seed_for(model_name: str, replicate: int) -> int:
    """Return the training seed for ``model_name`` in ``replicate``.

    Raises:
        ValueError: if the model name is not one of ``config.MODEL_NAMES`` or the
            replicate index is outside ``range(config.N_REPLICATES)``. Failing loudly
            here prevents a typo from silently producing an off-study seed.
    """
    if model_name not in config.MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}; expected one of {config.MODEL_NAMES}")
    if not 0 <= replicate < config.N_REPLICATES:
        raise ValueError(
            f"replicate {replicate} out of range; expected 0..{config.N_REPLICATES - 1}"
        )
    model_index = config.MODEL_NAMES.index(model_name)
    return (
        config.MODEL_SEED_BASE
        + replicate * config.MODEL_SEED_REPLICATE_STRIDE
        + model_index * config.MODEL_SEED_MODEL_STRIDE
    )


def all_seeds() -> dict[tuple[str, int], int]:
    """Return every (model, replicate) -> seed in the study, for manifests and tests."""
    return {
        (name, replicate): seed_for(name, replicate)
        for replicate in range(config.N_REPLICATES)
        for name in config.MODEL_NAMES
    }


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` and request deterministic kernels.

    Call once at the start of any run whose output is reported. Note that determinism is
    only guaranteed on CPU; ``config.DEFAULT_DEVICE`` is ``"cpu"`` for exactly this
    reason.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug("seeded all RNGs with %d", seed)


def torch_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with ``seed``, for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def dataloader_worker_init(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically from torch's initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def iter_model_replicates() -> Iterator[tuple[str, int]]:
    """Yield every (model_name, replicate) in a fixed, reproducible order."""
    for replicate in range(config.N_REPLICATES):
        for name in config.MODEL_NAMES:
            yield name, replicate
