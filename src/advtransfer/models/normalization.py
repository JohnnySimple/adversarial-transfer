"""Input normalization as a model layer.

CLAUDE.md section 3 requires that normalization live inside each model rather than in the
dataset transform. The reason is the threat model: epsilon is defined in [0, 1] pixel
space, so the attack must operate on raw pixels. If normalization happened in the
transform, an attack would be perturbing standardised units and the same nominal epsilon
would mean a different physical perturbation for every model, making the transfer matrix
uninterpretable.

Putting it here instead means every model is a function of raw pixels, so ``epsilon``
means exactly one thing across the whole study.
"""

from __future__ import annotations

import torch
from torch import nn

from advtransfer import config


class InputNormalization(nn.Module):
    """Standardise [0, 1] pixel input using fixed MNIST statistics.

    Mean and standard deviation are registered as persistent buffers, so a checkpoint
    fully determines the model's input handling and cannot be reloaded against different
    statistics by accident.
    """

    # Declared for the type checker: register_buffer otherwise types these as
    # ``Tensor | Module``, which blocks arithmetic on them.
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: float = config.MNIST_MEAN, std: float = config.MNIST_STD) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        return f"mean={float(self.mean):.4f}, std={float(self.std):.4f}"
