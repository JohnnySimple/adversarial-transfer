"""Guards on the experiment constants themselves.

These are cheap, but they catch the class of edit that quietly changes what is being
measured -- a dropped epsilon, a split that no longer sums to the dataset size, a
hypothesis sample size that no longer matches the matrix shape.
"""

from advtransfer import config


def test_split_sizes_sum_to_mnist_train_total():
    assert config.TRAIN_SIZE + config.VAL_SIZE == config.MNIST_TRAIN_TOTAL


def test_five_models_in_canonical_order():
    assert config.N_MODELS == 5
    assert len(set(config.MODEL_NAMES)) == config.N_MODELS
    assert config.MODEL_NAMES == ("cnn_a", "cnn_a_s2", "cnn_b", "lenet", "mlp")


def test_seed_twin_pair_is_in_model_names():
    for name in config.SEED_TWIN_PAIR:
        assert name in config.MODEL_NAMES


def test_ordered_pair_count_matches_correlation_sample_size():
    # The pre-registered pair-level correlations run over off-diagonal ordered pairs.
    # 5 models -> 20 pairs. If this changes, every power statement in the report changes.
    assert config.N_ORDERED_PAIRS == 20


def test_threat_model_budgets():
    assert config.EPSILONS == (0.1, 0.2, 0.3)
    assert config.ATTACKS == ("fgsm", "pgd")
    assert config.PGD_STEPS == 40
    assert config.PGD_RANDOM_START is True


def test_pgd_step_size_matches_threat_model():
    # Threat model: step size = 2.5 * epsilon / 40.
    for epsilon in config.EPSILONS:
        assert config.pgd_step_size(epsilon) == 2.5 * epsilon / 40


def test_pgd_steps_can_traverse_the_epsilon_ball():
    # Sanity on the step-size formula: 40 steps of 2.5eps/40 covers 2.5eps of travel,
    # comfortably more than the eps-ball diameter, so PGD is not step-size starved.
    for epsilon in config.EPSILONS:
        assert config.PGD_STEPS * config.pgd_step_size(epsilon) > 2 * epsilon


def test_primary_metric_choices_are_valid():
    assert config.SIGN_AGREEMENT_PRIMARY in config.SIGN_AGREEMENT_VARIANTS
    assert config.PRIMARY_TRANSFER_METRIC == "transfer_success"


def test_accuracy_gate_and_eval_size():
    assert config.MIN_CLEAN_ACCURACY == 0.97
    assert config.EVAL_SET_SIZE == 2_000


def test_config_hash_is_stable_and_short():
    assert config.config_hash() == config.config_hash()
    assert len(config.config_hash()) == 12
