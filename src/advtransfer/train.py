"""Training, the accuracy gate, and checkpoints.

Every model trains under the identical budget from ``config`` (Adam, lr 1e-3, batch 128,
5 epochs), so no model is advantaged by tuning. The only thing that differs between
``cnn_a`` and ``cnn_a_s2`` is the seed, which is what makes H5 a controlled comparison.

**Operation order is load-bearing.** Seeding, model construction, optimiser construction
and dataloader construction happen in a fixed order, because model initialisation draws
from the global torch RNG. Reordering them would change every trained weight and silently
break the reproducibility fixture recorded in CLAUDE.md section 4.

**The gate is a validation decision.** A model is admitted on validation accuracy, never
on evaluation-set accuracy. Gating on the evaluation set would select models by their
performance on the very set all transfer numbers are later measured on, and at n=2000 the
binomial standard error (0.38 pp) is large enough that a 97% threshold there would partly
be gating on sampling noise.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from advtransfer import config, data, models, provenance, seeds

logger = logging.getLogger(__name__)


class GateError(RuntimeError):
    """Raised when a trained model falls below the validation accuracy gate.

    This halts the pipeline by design (CLAUDE.md section 4). It must not be caught and
    worked around by retraining with a different seed: re-seeding until a model passes is
    selection on the outcome and would bias the study. The remedy is to change the
    architecture or the training budget and rerun *every* model under the new budget.
    """


@dataclass(frozen=True)
class TrainingResult:
    """Everything recorded about one training run."""

    model_name: str
    replicate: int
    seed: int
    val_accuracy: float
    clean_test_accuracy: float
    final_train_loss: float
    epochs: int
    n_parameters: int
    wall_clock_s: float
    gate_threshold: float
    gate_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def checkpoint_path(model_name: str, replicate: int) -> Path:
    """Return the weights path for one (model, replicate)."""
    return config.CHECKPOINT_DIR / f"{model_name}_r{replicate}.pt"


def metadata_path(model_name: str, replicate: int) -> Path:
    """Return the metadata sidecar path for one (model, replicate)."""
    return config.CHECKPOINT_DIR / f"{model_name}_r{replicate}.json"


@torch.no_grad()
def accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = config.BATCH_SIZE,
    device: str | torch.device = config.DEFAULT_DEVICE,
) -> float:
    """Return classification accuracy, evaluated in batches.

    Requires eval mode, which is enforced: with ``cnn_b``'s batch normalization in
    training mode, predictions would depend on batch composition and the result would
    change with ``batch_size``.
    """
    models.assert_eval_mode(model)
    correct = 0
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        targets = labels[start : start + batch_size].to(device)
        correct += int((model(batch).argmax(dim=1) == targets).sum())
    return correct / len(images)


def train_one(
    model_name: str,
    replicate: int,
    *,
    device: str | torch.device = config.DEFAULT_DEVICE,
    download: bool = True,
    enforce_gate: bool = True,
) -> tuple[nn.Module, TrainingResult]:
    """Train one (model, replicate) and evaluate it.

    Returns the trained model in eval mode, plus its :class:`TrainingResult`.

    Raises:
        GateError: if validation accuracy is below ``config.MIN_CLEAN_ACCURACY`` and
            ``enforce_gate`` is set. ``enforce_gate=False`` exists only for tests that
            deliberately train an under-budget model; production runs never set it.
    """
    seed = seeds.seed_for(model_name, replicate)

    # --- Order below is fixed; see module docstring. ---------------------------------
    seeds.seed_everything(seed)
    model = models.build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loader = data.train_dataloader(seed, download=download)
    # ---------------------------------------------------------------------------------

    logger.info(
        "training %s replicate %d (seed %d, %s params) on %s",
        model_name,
        replicate,
        seed,
        f"{models.n_parameters(model):,}",
        device,
    )

    started = time.perf_counter()
    model.train()
    final_train_loss = float("nan")
    for epoch in range(config.EPOCHS):
        running, batches = 0.0, 0
        for batch, targets in loader:
            batch, targets = batch.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch), targets)
            # Tensor.backward is untyped in the torch 2.7 stubs. Suppressed rather than
            # routed through autograd.grad: the training path must stay byte-identical to
            # the one that produced the reproducibility fixture, and a type checker is
            # not a reason to change training numerics.
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            running += float(loss)
            batches += 1
        final_train_loss = running / batches
        logger.info(
            "  %s r%d epoch %d/%d: train loss %.4f",
            model_name,
            replicate,
            epoch + 1,
            config.EPOCHS,
            final_train_loss,
        )
    wall_clock = time.perf_counter() - started

    model.eval()
    _, val = data.train_val_datasets(download=download)
    val_images, val_labels = data.as_tensors(val)
    test_images, test_labels = data.eval_tensors(download=download)

    val_accuracy = accuracy(model, val_images, val_labels, device=device)
    clean_test_accuracy = accuracy(model, test_images, test_labels, device=device)
    gate_passed = val_accuracy >= config.MIN_CLEAN_ACCURACY

    result = TrainingResult(
        model_name=model_name,
        replicate=replicate,
        seed=seed,
        val_accuracy=val_accuracy,
        clean_test_accuracy=clean_test_accuracy,
        final_train_loss=final_train_loss,
        epochs=config.EPOCHS,
        n_parameters=models.n_parameters(model),
        wall_clock_s=wall_clock,
        gate_threshold=config.MIN_CLEAN_ACCURACY,
        gate_passed=gate_passed,
    )

    logger.info(
        "  %s r%d done in %.0fs: val %.2f%%, clean test %.2f%% (gate %s)",
        model_name,
        replicate,
        wall_clock,
        val_accuracy * 100,
        clean_test_accuracy * 100,
        "pass" if gate_passed else "FAIL",
    )

    if enforce_gate and not gate_passed:
        raise GateError(
            f"{model_name} replicate {replicate} (seed {seed}) reached "
            f"{val_accuracy:.2%} validation accuracy, below the "
            f"{config.MIN_CLEAN_ACCURACY:.0%} gate. Do not re-seed until it passes: "
            f"that is selection on the outcome. Change the architecture or the training "
            f"budget in config.py and rerun every model under the new budget."
        )
    return model, result


def save_checkpoint(model: nn.Module, result: TrainingResult) -> Path:
    """Write weights and a metadata sidecar; return the weights path."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    weights = checkpoint_path(result.model_name, result.replicate)
    torch.save(model.state_dict(), weights)

    metadata = {
        **result.as_dict(),
        **provenance.environment_info(),
        "architecture": type(model).__name__,
    }
    metadata_path(result.model_name, result.replicate).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    logger.debug("saved %s", weights)
    return weights


def load_checkpoint(
    model_name: str,
    replicate: int,
    *,
    device: str | torch.device = config.DEFAULT_DEVICE,
) -> nn.Module:
    """Load a trained model in **eval mode**.

    Eval mode is the default rather than an option because every downstream use -- attack
    generation, gradient measurement, evaluation -- requires it, and training mode would
    silently couple samples through ``cnn_b``'s batch-norm statistics.

    A checkpoint written under a different ``config_hash`` is loaded but warned about: the
    constants defining the experiment have changed since it was trained.
    """
    weights = checkpoint_path(model_name, replicate)
    if not weights.exists():
        raise FileNotFoundError(
            f"no checkpoint at {weights}; train {model_name} replicate {replicate} first"
        )

    sidecar = metadata_path(model_name, replicate)
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        stored_hash = metadata.get("config_hash")
        if stored_hash != config.config_hash():
            logger.warning(
                "%s r%d was trained under config_hash %s but the current config hash is "
                "%s; experiment constants have changed since training",
                model_name,
                replicate,
                stored_hash,
                config.config_hash(),
            )

    model = models.build_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model


def is_checkpoint_current(model_name: str, replicate: int) -> bool:
    """Return whether a checkpoint exists and was trained under the current config."""
    if not checkpoint_path(model_name, replicate).exists():
        return False
    sidecar = metadata_path(model_name, replicate)
    if not sidecar.exists():
        return False
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("config_hash") == config.config_hash()
        and metadata.get("seed") == seeds.seed_for(model_name, replicate)
        and metadata.get("gate_passed") is True
    )


def train_all(
    *,
    device: str | torch.device = config.DEFAULT_DEVICE,
    download: bool = True,
    reuse_existing: bool = False,
) -> list[TrainingResult]:
    """Train all 15 (model, replicate) runs in a fixed order.

    Halts on the first gate failure rather than continuing, so a budget problem surfaces
    immediately instead of after another fifteen minutes of training.

    Args:
        reuse_existing: skip runs whose checkpoint already matches the current config
            hash and seed. Off by default so a full run always retrains from scratch and
            cannot silently mix stale weights with fresh ones.
    """
    provenance.warn_if_dirty()
    results: list[TrainingResult] = []
    for model_name, replicate in seeds.iter_model_replicates():
        if reuse_existing and is_checkpoint_current(model_name, replicate):
            metadata = json.loads(metadata_path(model_name, replicate).read_text())
            logger.info(
                "reusing %s r%d (val %.2f%%)",
                model_name,
                replicate,
                metadata["val_accuracy"] * 100,
            )
            results.append(
                TrainingResult(
                    **{
                        field.name: metadata[field.name]
                        for field in dataclasses.fields(TrainingResult)
                    }
                )
            )
            continue
        model, result = train_one(model_name, replicate, device=device, download=download)
        save_checkpoint(model, result)
        results.append(result)

    logger.info(
        "trained %d models; all cleared the %.0f%% validation gate",
        len(results),
        config.MIN_CLEAN_ACCURACY * 100,
    )
    return results
