"""Seed-scheme guards.

The seed scheme carries real scientific weight: H5 is entirely the claim that two models
differing *only* in seed behave differently from two models differing in architecture. If
the twin pair ever shared a seed, H5 would be untestable; if two runs collided, the
replicate error bars would be understated.
"""

import itertools

import pytest
import torch

from advtransfer import config, seeds


def test_seed_for_is_deterministic():
    assert seeds.seed_for("cnn_a", 0) == seeds.seed_for("cnn_a", 0)


def test_all_seeds_are_pairwise_distinct():
    values = list(seeds.all_seeds().values())
    assert len(values) == config.N_MODELS * config.N_REPLICATES
    assert len(set(values)) == len(values)


def test_seed_twin_pair_differs_within_every_replicate():
    first, second = config.SEED_TWIN_PAIR
    for replicate in range(config.N_REPLICATES):
        assert seeds.seed_for(first, replicate) != seeds.seed_for(second, replicate)


def test_same_model_differs_across_replicates():
    for name in config.MODEL_NAMES:
        values = {seeds.seed_for(name, r) for r in range(config.N_REPLICATES)}
        assert len(values) == config.N_REPLICATES


def test_seed_spacing_leaves_room_for_derived_subseeds():
    # Model seeds are spaced by MODEL_SEED_MODEL_STRIDE, so seed + k for small k cannot
    # collide with another model's seed. Training code relies on this for derived seeds.
    ordered = sorted(seeds.all_seeds().values())
    gaps = [b - a for a, b in itertools.pairwise(ordered)]
    assert min(gaps) >= config.MODEL_SEED_MODEL_STRIDE


def test_seed_for_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        seeds.seed_for("resnet50", 0)


@pytest.mark.parametrize("replicate", [-1, config.N_REPLICATES])
def test_seed_for_rejects_out_of_range_replicate(replicate):
    with pytest.raises(ValueError, match="out of range"):
        seeds.seed_for("cnn_a", replicate)


def test_iter_model_replicates_covers_the_study_once():
    pairs = list(seeds.iter_model_replicates())
    assert len(pairs) == config.N_MODELS * config.N_REPLICATES
    assert len(set(pairs)) == len(pairs)


def test_seed_everything_makes_torch_reproducible():
    seeds.seed_everything(123)
    first = torch.randn(8)
    seeds.seed_everything(123)
    assert torch.equal(first, torch.randn(8))

    seeds.seed_everything(124)
    assert not torch.equal(first, torch.randn(8))


def test_torch_generator_is_independent_of_global_state():
    a = torch.randn(4, generator=seeds.torch_generator(7))
    b = torch.randn(4, generator=seeds.torch_generator(7))
    assert torch.equal(a, b)
