"""The five model architectures.

Roles, from the proposal's experimental design -- each model exists to vary exactly one
thing relative to ``cnn_a``:

==========  ==========================================  =====================================
Name        Architecture                                What it varies
==========  ==========================================  =====================================
cnn_a       2x conv3x3 + max-pool + dense head          reference
cnn_a_s2    *the same class as* cnn_a                    training seed only (H5)
cnn_b       3x conv5x5, batch norm, global avg pool     architecture + normalization + pooling
lenet       LeNet-5 style, tanh + average pooling       activation smoothness
mlp         784-512-256-10, ReLU                        convolution vs. none
==========  ==========================================  =====================================

Every model begins with :class:`InputNormalization`, so all five are functions of raw
[0, 1] pixels (see ``normalization``). ``cnn_a_s2`` has no class of its own by design:
section 4 forbids duplicating the definition, because H5's entire content is that the two
differ *only* in seed.
"""

from __future__ import annotations

import torch
from torch import nn

from advtransfer.models.normalization import InputNormalization

N_CLASSES = 10


class CnnA(nn.Module):
    """Reference CNN: two 3x3 convolution blocks with max-pooling, then a dense head.

    Used for both ``cnn_a`` and ``cnn_a_s2``; the two differ only in training seed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.normalize = InputNormalization()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.features(self.normalize(x)))
        return logits


class CnnB(nn.Module):
    """Controlled architecture change: three 5x5 convolutions, batch norm, global avg pool.

    Note for interpretation: this model differs from ``cnn_a`` in three ways at once
    (kernel size, batch normalization, and a GAP head instead of a dense head). The
    stretch-goal batch-norm ablation in section 2 exists to separate those effects, so
    results here should not be read as isolating "kernel size" or "batch norm" alone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.normalize = InputNormalization()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling
            nn.Flatten(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.features(self.normalize(x)))
        return logits


class LeNet(nn.Module):
    """LeNet-5 style network: tanh activations and average pooling.

    The smooth-activation comparison. ReLU networks are piecewise linear, so their input
    gradient is locally constant; tanh networks are not. If gradient alignment behaves
    differently here, activation smoothness is a candidate explanation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.normalize = InputNormalization()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),  # 28 -> 14
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),  # 10 -> 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.features(self.normalize(x)))
        return logits


class Mlp(nn.Module):
    """Non-convolutional baseline: 784-512-256-10 with ReLU.

    Has no spatial inductive bias at all, so it is the most architecturally distant
    model in the study and the natural lower bound for cross-architecture transfer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.normalize = InputNormalization()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.normalize(x))
        return logits
