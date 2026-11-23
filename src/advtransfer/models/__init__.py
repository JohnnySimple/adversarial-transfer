"""Model registry and mode guards.

The registry is the only place a model name is bound to an architecture. ``cnn_a`` and
``cnn_a_s2`` map to the *same* class object, which is what makes H5 ("same architecture,
different seed") a controlled comparison rather than two unrelated models.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from advtransfer import config
from advtransfer.models.architectures import CnnA, CnnB, LeNet, Mlp
from advtransfer.models.normalization import InputNormalization

__all__ = [
    "ARCHITECTURES",
    "CnnA",
    "CnnB",
    "InputNormalization",
    "LeNet",
    "Mlp",
    "architecture_class",
    "assert_eval_mode",
    "build_model",
    "n_parameters",
    "shares_architecture",
]

logger = logging.getLogger(__name__)

#: Model name -> architecture class. ``cnn_a`` and ``cnn_a_s2`` deliberately share one
#: entry: duplicating the definition would make H5 untestable (CLAUDE.md section 4).
ARCHITECTURES: dict[str, type[nn.Module]] = {
    "cnn_a": CnnA,
    "cnn_a_s2": CnnA,
    "cnn_b": CnnB,
    "lenet": LeNet,
    "mlp": Mlp,
}


def architecture_class(name: str) -> type[nn.Module]:
    """Return the architecture class registered for ``name``."""
    try:
        return ARCHITECTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; expected one of {tuple(ARCHITECTURES)}"
        ) from None


def build_model(name: str) -> nn.Module:
    """Construct an untrained model by registry name.

    Deliberately does not seed: the caller seeds first (``seeds.seed_everything``) so that
    the seeding step is explicit and visible at the call site rather than hidden inside
    construction. This is what lets ``cnn_a`` and ``cnn_a_s2`` differ by seed alone.
    """
    return architecture_class(name)()


def shares_architecture(first: str, second: str) -> bool:
    """Return whether two model names resolve to the same architecture class."""
    return architecture_class(first) is architecture_class(second)


def n_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_eval_mode(model: nn.Module) -> None:
    """Fail if ``model`` is in training mode.

    This matters more than it looks. ``cnn_b`` contains batch normalization, and in
    training mode BatchNorm normalises using *batch* statistics. Every sample's output --
    and therefore its input gradient -- would then depend on the other samples sharing its
    batch. That would silently break three things the study depends on:

    1. per-image gradient cosine would no longer be a property of that image,
    2. a per-image adversarial perturbation would depend on its batch-mates, so results
       would change with batch size and ordering,
    3. the measurements would stop being reproducible across any change in batching.

    Attack generation, gradient computation and evaluation must therefore all run under
    ``model.eval()``. Because the failure is silent and only affects one of five models,
    it is asserted rather than assumed.
    """
    if model.training:
        raise RuntimeError(
            f"{type(model).__name__} is in training mode; attacks, gradients and "
            "evaluation require eval() so batch-norm uses running statistics and "
            "per-image quantities stay independent of batch composition"
        )


def assert_registry_matches_config() -> None:
    """Fail if the registry and ``config.MODEL_NAMES`` have drifted apart."""
    if tuple(ARCHITECTURES) != config.MODEL_NAMES:
        raise RuntimeError(
            f"registry {tuple(ARCHITECTURES)} does not match "
            f"config.MODEL_NAMES {config.MODEL_NAMES}"
        )


def input_gradient(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-image gradients of cross-entropy loss with respect to the input.

    Computed with a single backward pass on the *summed* loss. This is valid because
    cross-entropy is separable across samples: d(sum_i L_i)/dx_j = dL_j/dx_j, since L_i
    for i != j does not depend on x_j. So one backward pass yields every per-image
    gradient exactly, with no loop (CLAUDE.md section 5).

    The model must be in eval mode, which is enforced: under training-mode batch norm the
    separability argument above fails, because samples become coupled through the batch
    statistics.
    """
    assert_eval_mode(model)
    x = images.detach().clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), labels, reduction="sum")
    (grad,) = torch.autograd.grad(loss, x)
    return grad
