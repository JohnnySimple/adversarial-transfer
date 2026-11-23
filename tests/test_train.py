"""Training-layer guards.

The expensive tests are marked ``slow``. The fast ones cover the gate logic, checkpoint
round-trips and eval-mode enforcement without training anything.
"""

import json

import pytest
import torch

from advtransfer import config, models, seeds, train

# ----------------------------------------------------------------------------------
# Accuracy helper
# ----------------------------------------------------------------------------------


def test_accuracy_counts_correctly_on_a_known_case():
    class Fixed(torch.nn.Module):
        """Always predicts class 3."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(len(x), 10)
            logits[:, 3] = 1.0
            return logits

    model = Fixed().eval()
    images = torch.rand(10, 1, 28, 28)
    labels = torch.tensor([3, 3, 3, 3, 0, 1, 2, 4, 5, 6])
    assert train.accuracy(model, images, labels) == pytest.approx(0.4)


def test_accuracy_is_independent_of_batch_size():
    # Must hold so that a reported accuracy is a property of the model, not of how it
    # happened to be batched. Only true in eval mode for the batch-norm model.
    seeds.seed_everything(0)
    model = models.build_model("cnn_b").eval()
    images, labels = torch.rand(64, 1, 28, 28), torch.randint(0, 10, (64,))
    reference = train.accuracy(model, images, labels, batch_size=64)
    for batch_size in (1, 7, 16, 128):
        assert train.accuracy(model, images, labels, batch_size=batch_size) == reference


def test_accuracy_refuses_training_mode():
    model = models.build_model("cnn_b").train()
    with pytest.raises(RuntimeError, match="training mode"):
        train.accuracy(model, torch.rand(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))


# ----------------------------------------------------------------------------------
# Paths and checkpoint round-trip
# ----------------------------------------------------------------------------------


def test_checkpoint_paths_are_distinct_per_model_and_replicate():
    paths = {
        train.checkpoint_path(name, replicate) for name, replicate in seeds.iter_model_replicates()
    }
    assert len(paths) == config.N_MODELS * config.N_REPLICATES


def test_load_checkpoint_errors_clearly_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        train.load_checkpoint("cnn_a", 0)


def test_checkpoint_round_trip_preserves_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    seeds.seed_everything(seeds.seed_for("lenet", 0))
    model = models.build_model("lenet").eval()
    result = train.TrainingResult(
        model_name="lenet",
        replicate=0,
        seed=seeds.seed_for("lenet", 0),
        val_accuracy=0.98,
        clean_test_accuracy=0.979,
        final_train_loss=0.05,
        epochs=config.EPOCHS,
        n_parameters=models.n_parameters(model),
        wall_clock_s=26.0,
        gate_threshold=config.MIN_CLEAN_ACCURACY,
        gate_passed=True,
    )
    train.save_checkpoint(model, result)

    images = torch.rand(8, 1, 28, 28)
    with torch.no_grad():
        before = model(images)
    restored = train.load_checkpoint("lenet", 0)
    with torch.no_grad():
        after = restored(images)
    assert torch.equal(before, after)


def test_loaded_checkpoint_is_in_eval_mode(tmp_path, monkeypatch):
    # Every downstream use requires eval mode, so loading must not hand back a model in
    # training mode that would silently couple samples through batch norm.
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    seeds.seed_everything(0)
    model = models.build_model("cnn_b")
    result = train.TrainingResult(
        model_name="cnn_b",
        replicate=1,
        seed=seeds.seed_for("cnn_b", 1),
        val_accuracy=0.985,
        clean_test_accuracy=0.982,
        final_train_loss=0.04,
        epochs=config.EPOCHS,
        n_parameters=models.n_parameters(model),
        wall_clock_s=167.0,
        gate_threshold=config.MIN_CLEAN_ACCURACY,
        gate_passed=True,
    )
    train.save_checkpoint(model.eval(), result)
    restored = train.load_checkpoint("cnn_b", 1)
    assert not restored.training
    models.assert_eval_mode(restored)


def test_metadata_sidecar_records_provenance_and_result(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    seeds.seed_everything(0)
    model = models.build_model("mlp").eval()
    result = train.TrainingResult(
        model_name="mlp",
        replicate=2,
        seed=seeds.seed_for("mlp", 2),
        val_accuracy=0.9736,
        clean_test_accuracy=0.965,
        final_train_loss=0.03,
        epochs=config.EPOCHS,
        n_parameters=models.n_parameters(model),
        wall_clock_s=6.0,
        gate_threshold=config.MIN_CLEAN_ACCURACY,
        gate_passed=True,
    )
    train.save_checkpoint(model, result)
    metadata = json.loads(train.metadata_path("mlp", 2).read_text())

    assert metadata["model_name"] == "mlp"
    assert metadata["seed"] == seeds.seed_for("mlp", 2)
    assert metadata["config_hash"] == config.config_hash()
    assert metadata["architecture"] == "Mlp"
    assert "git_sha" in metadata and "git_dirty" in metadata


def test_is_checkpoint_current_detects_config_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    seeds.seed_everything(0)
    model = models.build_model("lenet").eval()
    result = train.TrainingResult(
        model_name="lenet",
        replicate=0,
        seed=seeds.seed_for("lenet", 0),
        val_accuracy=0.98,
        clean_test_accuracy=0.979,
        final_train_loss=0.05,
        epochs=config.EPOCHS,
        n_parameters=models.n_parameters(model),
        wall_clock_s=26.0,
        gate_threshold=config.MIN_CLEAN_ACCURACY,
        gate_passed=True,
    )
    train.save_checkpoint(model, result)
    assert train.is_checkpoint_current("lenet", 0)

    sidecar = train.metadata_path("lenet", 0)
    metadata = json.loads(sidecar.read_text())
    metadata["config_hash"] = "deadbeef0000"
    sidecar.write_text(json.dumps(metadata))
    assert not train.is_checkpoint_current("lenet", 0)


def test_is_checkpoint_current_rejects_a_failed_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    seeds.seed_everything(0)
    model = models.build_model("lenet").eval()
    result = train.TrainingResult(
        model_name="lenet",
        replicate=0,
        seed=seeds.seed_for("lenet", 0),
        val_accuracy=0.5,
        clean_test_accuracy=0.5,
        final_train_loss=2.0,
        epochs=config.EPOCHS,
        n_parameters=models.n_parameters(model),
        wall_clock_s=26.0,
        gate_threshold=config.MIN_CLEAN_ACCURACY,
        gate_passed=False,
    )
    train.save_checkpoint(model, result)
    assert not train.is_checkpoint_current("lenet", 0)


# ----------------------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------------------


def test_gate_error_is_not_silently_catchable_as_a_value_error():
    # GateError must halt the pipeline. Deriving from RuntimeError keeps it outside the
    # ValueError/KeyError classes that argument-validation handlers tend to catch.
    assert issubclass(train.GateError, RuntimeError)
    assert not issubclass(train.GateError, ValueError)


@pytest.mark.slow
def test_gate_halts_on_an_under_budget_model(monkeypatch):
    # Train with one epoch on a deliberately crippled budget so the gate must trip. This
    # exercises the real halt path rather than a mocked one.
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "LEARNING_RATE", 1e-9)  # learns essentially nothing
    with pytest.raises(train.GateError, match="below the 97% gate"):
        train.train_one("mlp", 0)


@pytest.mark.slow
def test_gate_message_names_model_replicate_seed_and_accuracy(monkeypatch):
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "LEARNING_RATE", 1e-9)
    with pytest.raises(train.GateError) as excinfo:
        train.train_one("mlp", 2)
    message = str(excinfo.value)
    assert "mlp" in message
    assert "replicate 2" in message
    assert str(seeds.seed_for("mlp", 2)) in message
    assert "%" in message
    # Must tell the reader what not to do.
    assert "re-seed" in message


@pytest.mark.slow
def test_enforce_gate_false_returns_the_failing_result(monkeypatch):
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "LEARNING_RATE", 1e-9)
    _, result = train.train_one("mlp", 0, enforce_gate=False)
    assert result.gate_passed is False
    assert result.val_accuracy < config.MIN_CLEAN_ACCURACY


# ----------------------------------------------------------------------------------
# Real training
# ----------------------------------------------------------------------------------


@pytest.mark.slow
def test_training_lenet_clears_the_gate_and_is_reproducible():
    # lenet is the cheapest real model (~26s). Training it twice must give identical
    # accuracy: the pipeline's reproducibility claim depends on this.
    _, first = train.train_one("lenet", 0)
    _, second = train.train_one("lenet", 0)
    assert first.gate_passed
    assert first.val_accuracy == second.val_accuracy
    assert first.clean_test_accuracy == second.clean_test_accuracy
    assert first.final_train_loss == pytest.approx(second.final_train_loss, abs=1e-9)


@pytest.mark.slow
def test_result_records_the_configured_budget():
    _, result = train.train_one("mlp", 0)
    assert result.epochs == config.EPOCHS
    assert result.gate_threshold == config.MIN_CLEAN_ACCURACY
    assert result.seed == seeds.seed_for("mlp", 0)
    assert result.n_parameters > 0
    assert result.wall_clock_s > 0
