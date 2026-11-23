"""The reproducibility fixture.

These assert the exact validation accuracies recorded in CLAUDE.md section 4, which were
measured before ``train.py`` existed. They are the project's strongest guard: anything
that perturbs the training path -- a reordered seeding call, a changed optimiser
construction, a different dataloader generator, an altered architecture, a torch upgrade
that shifts kernel selection -- moves these numbers and fails here.

A failure is a signal to find what changed, not to update the expected values. Updating
them to match new output would discard the only evidence that the pipeline is
reproducible at all.

The ``mlp`` cases are cheap (~6 s each) and run in the ``slow`` tier. The ``cnn_b`` cases
cost ~167 s each and are marked ``expensive``, excluded from the default run; invoke them
with ``pytest -m expensive``.
"""

import pytest

from advtransfer import config, train

# (model, replicate) -> validation accuracy, from CLAUDE.md section 4.
FIXTURE_CASES = [
    pytest.param("mlp", 0, 0.9754, marks=pytest.mark.slow),
    pytest.param("mlp", 1, 0.9754, marks=pytest.mark.slow),
    pytest.param("mlp", 2, 0.9736, marks=pytest.mark.slow),
    pytest.param("cnn_b", 0, 0.9746, marks=pytest.mark.expensive),
    pytest.param("cnn_b", 1, 0.9852, marks=pytest.mark.expensive),
    pytest.param("cnn_b", 2, 0.9740, marks=pytest.mark.expensive),
]


@pytest.mark.parametrize(("model_name", "replicate", "expected"), FIXTURE_CASES)
def test_training_reproduces_recorded_validation_accuracy(model_name, replicate, expected):
    _, result = train.train_one(model_name, replicate)
    # Exact to the recorded 4 decimal places: 2 of 5000 validation images would be a
    # 0.04 pp shift, so this tolerance still detects a single flipped prediction.
    assert result.val_accuracy == pytest.approx(expected, abs=5e-5), (
        f"{model_name} r{replicate} gave {result.val_accuracy:.4%}, fixture says "
        f"{expected:.4%}. The training path has changed -- find what changed rather "
        f"than updating this expected value."
    )
    assert result.gate_passed


@pytest.mark.parametrize(("model_name", "replicate", "expected"), FIXTURE_CASES)
def test_fixture_models_clear_the_validation_gate(model_name, replicate, expected):
    # Documents that the recorded values are all above the gate, so the pipeline is known
    # not to halt on these six runs rather than merely expected not to.
    assert expected >= config.MIN_CLEAN_ACCURACY
