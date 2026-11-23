"""Model-layer guards.

Three of these encode failures that would otherwise be silent: the twin pair sharing a
class (H5's validity), normalization living inside the model (epsilon's meaning), and
eval mode under batch norm (per-image independence).
"""

import pytest
import torch
from torch import nn

from advtransfer import config, models, seeds
from advtransfer.models.normalization import InputNormalization

BATCH = 4


def pixel_batch(n: int = BATCH) -> torch.Tensor:
    return torch.rand(n, 1, 28, 28)


# ----------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------


def test_registry_covers_exactly_the_configured_models():
    assert tuple(models.ARCHITECTURES) == config.MODEL_NAMES
    models.assert_registry_matches_config()


def test_architecture_class_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        models.architecture_class("resnet50")


def test_twin_pair_shares_one_architecture_class():
    # H5 is the claim that cnn_a and cnn_a_s2 differ *only* in training seed. If they
    # ever resolved to different classes the hypothesis would be untestable, so this is
    # asserted on class identity rather than on parameter counts matching by luck.
    first, second = config.SEED_TWIN_PAIR
    assert models.shares_architecture(first, second)
    assert models.architecture_class(first) is models.architecture_class(second)


def test_non_twin_models_have_distinct_architectures():
    distinct = {"cnn_a", "cnn_b", "lenet", "mlp"}
    classes = {models.architecture_class(name) for name in distinct}
    assert len(classes) == len(distinct)


def test_twin_pair_has_identical_state_dict_structure():
    first, second = config.SEED_TWIN_PAIR
    a, b = models.build_model(first), models.build_model(second)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert models.n_parameters(a) == models.n_parameters(b)


# ----------------------------------------------------------------------------------
# Forward pass
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_forward_shape_and_finiteness(name):
    model = models.build_model(name).eval()
    logits = model(pixel_batch())
    assert logits.shape == (BATCH, 10)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_models_accept_full_pixel_range(name):
    # Models are functions of raw [0, 1] pixels, including the extremes that clipping
    # produces during an attack.
    model = models.build_model(name).eval()
    for value in (config.PIXEL_MIN, config.PIXEL_MAX):
        logits = model(torch.full((BATCH, 1, 28, 28), value))
        assert torch.isfinite(logits).all()


# ----------------------------------------------------------------------------------
# Normalization lives inside the model
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_every_model_normalizes_internally(name):
    model = models.build_model(name)
    normalizers = [m for m in model.modules() if isinstance(m, InputNormalization)]
    assert len(normalizers) == 1, "exactly one input-normalization layer expected"


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_normalization_is_the_first_operation(name):
    # Verified behaviourally: feeding the model pre-normalized input must change the
    # output, which can only be true if the model normalizes internally.
    seeds.seed_everything(0)
    model = models.build_model(name).eval()
    x = pixel_batch()
    pre_normalized = (x - config.MNIST_MEAN) / config.MNIST_STD
    assert not torch.allclose(model(x), model(pre_normalized))


def test_normalization_matches_configured_statistics():
    layer = InputNormalization()
    x = pixel_batch()
    expected = (x - config.MNIST_MEAN) / config.MNIST_STD
    assert torch.allclose(layer(x), expected)


def test_normalization_statistics_survive_a_checkpoint_round_trip():
    layer = InputNormalization()
    state = layer.state_dict()
    assert "mean" in state and "std" in state
    restored = InputNormalization(mean=0.0, std=1.0)
    restored.load_state_dict(state)
    assert float(restored.mean) == pytest.approx(config.MNIST_MEAN)


def test_normalization_rejects_non_positive_std():
    with pytest.raises(ValueError, match="std must be positive"):
        InputNormalization(std=0.0)


# ----------------------------------------------------------------------------------
# Seed determinism
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_same_seed_gives_identical_initialisation(name):
    def weights(seed: int) -> list[torch.Tensor]:
        seeds.seed_everything(seed)
        return [p.detach().clone() for p in models.build_model(name).parameters()]

    for a, b in zip(weights(42), weights(42), strict=True):
        assert torch.equal(a, b)


def test_twin_pair_initialises_differently_under_its_own_seeds():
    # The two share a class, so only the seed separates them. Confirm the seed scheme
    # actually produces different weights rather than merely different seed integers.
    first, second = config.SEED_TWIN_PAIR
    for replicate in range(config.N_REPLICATES):
        seeds.seed_everything(seeds.seed_for(first, replicate))
        a = [p.detach().clone() for p in models.build_model(first).parameters()]
        seeds.seed_everything(seeds.seed_for(second, replicate))
        b = [p.detach().clone() for p in models.build_model(second).parameters()]
        assert not all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


# ----------------------------------------------------------------------------------
# Eval mode and batch-norm coupling
# ----------------------------------------------------------------------------------


def test_assert_eval_mode_rejects_training_mode():
    model = models.build_model("cnn_b").train()
    with pytest.raises(RuntimeError, match="training mode"):
        models.assert_eval_mode(model)
    models.assert_eval_mode(model.eval())


def test_batch_norm_in_training_mode_couples_samples():
    # Demonstrates that the eval-mode guard protects against a real effect, not a
    # hypothetical one: in training mode an image's logits change when its batch-mates
    # change. In eval mode they do not.
    seeds.seed_everything(0)
    model = models.build_model("cnn_b")
    target = pixel_batch(1)

    model.train()
    with torch.no_grad():
        a = model(torch.cat([target, pixel_batch(7)]))[0]
        b = model(torch.cat([target, torch.zeros(7, 1, 28, 28)]))[0]
    assert not torch.allclose(a, b, atol=1e-5), "expected batch coupling in train mode"

    model.eval()
    with torch.no_grad():
        c = model(torch.cat([target, pixel_batch(7)]))[0]
        d = model(torch.cat([target, torch.zeros(7, 1, 28, 28)]))[0]
    assert torch.allclose(c, d, atol=1e-6), "eval mode must decouple samples"


# ----------------------------------------------------------------------------------
# Input gradients
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_input_gradient_shape_and_non_triviality(name):
    seeds.seed_everything(0)
    model = models.build_model(name).eval()
    x, y = pixel_batch(), torch.randint(0, 10, (BATCH,))
    grad = models.input_gradient(model, x, y)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_batched_input_gradient_equals_per_image_gradient(name):
    # The batched backward pass on a summed loss must reproduce exactly what a
    # one-image-at-a-time loop would produce. This is the separability argument that
    # lets the whole study avoid a 2,000-iteration Python loop per pair.
    seeds.seed_everything(0)
    model = models.build_model(name).eval()
    x, y = pixel_batch(6), torch.randint(0, 10, (6,))

    batched = models.input_gradient(model, x, y)
    for i in range(len(x)):
        single = models.input_gradient(model, x[i : i + 1], y[i : i + 1])
        assert torch.allclose(batched[i], single[0], atol=1e-6)


def test_input_gradient_refuses_training_mode():
    model = models.build_model("cnn_b").train()
    with pytest.raises(RuntimeError, match="training mode"):
        models.input_gradient(model, pixel_batch(), torch.zeros(BATCH, dtype=torch.long))


def test_input_gradient_does_not_leave_parameter_gradients_behind():
    # Attack generation must not pollute optimiser state; autograd.grad is used precisely
    # so that parameter .grad attributes stay untouched.
    seeds.seed_everything(0)
    model = models.build_model("cnn_a").eval()
    models.input_gradient(model, pixel_batch(), torch.zeros(BATCH, dtype=torch.long))
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_parameter_counts_are_reported(name):
    model = models.build_model(name)
    assert models.n_parameters(model) > 0
    assert isinstance(model, nn.Module)
