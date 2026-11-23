"""Single source of truth for every experiment constant.

CLAUDE.md section 8 forbids magic numbers anywhere else in the codebase: each epsilon,
step count, seed base, split size and accuracy gate appears here exactly once. Modules
import these names rather than restating the values.

The values are fixed by the project proposal's threat model (section 3) and experimental
setup (section 4). Changing one changes what is being measured, so every change must be
deliberate and will alter ``config_hash()`` and therefore every downstream run manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Final[Path] = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------------------
# Models and replicates
# --------------------------------------------------------------------------------------

#: The five model specifications, in canonical order. This order fixes row/column order
#: in every 5x5 transfer matrix and is the index basis for ``seeds.seed_for``.
MODEL_NAMES: Final[tuple[str, ...]] = ("cnn_a", "cnn_a_s2", "cnn_b", "lenet", "mlp")

#: ``cnn_a_s2`` is architecturally identical to ``cnn_a`` and differs only in training
#: seed. H5 (same-architecture/different-seed pairs transfer more) rests on this pair, so
#: both names must resolve to the same architecture class.
SEED_TWIN_PAIR: Final[tuple[str, str]] = ("cnn_a", "cnn_a_s2")

#: Base for the deterministic model-seed scheme in ``seeds.seed_for``. Model seeds are
#: spaced 100 apart and replicates 1000 apart, leaving room for derived sub-seeds (e.g.
#: dataloader shuffling) without any chance of collision between runs.
MODEL_SEED_BASE: Final[int] = 20_260_000
MODEL_SEED_REPLICATE_STRIDE: Final[int] = 1_000
MODEL_SEED_MODEL_STRIDE: Final[int] = 100

#: Number of independent full-pipeline replicates. Each replicate retrains all five
#: models under fresh seeds, giving mean +/- std on every reported quantity.
N_REPLICATES: Final[int] = 3

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

MNIST_TRAIN_TOTAL: Final[int] = 60_000
TRAIN_SIZE: Final[int] = 55_000
VAL_SIZE: Final[int] = 5_000

#: Seed for the train/validation split. Deliberately independent of model seeds: every
#: model in every replicate trains on the identical 55k/5k partition, so differences
#: between models are never confounded with differences in training data.
SPLIT_SEED: Final[int] = 20_261_105

#: Evaluation set is the FIRST ``EVAL_SET_SIZE`` test images in canonical order -- no
#: shuffling, no resampling -- identical for every model, attack, budget and replicate.
EVAL_SET_SIZE: Final[int] = 2_000

#: MNIST channel statistics, applied inside each model (see models.normalization), never
#: in the dataset transform. Datasets yield raw [0, 1] pixels so that epsilon always
#: means the same thing in pixel units for every architecture.
MNIST_MEAN: Final[float] = 0.1307
MNIST_STD: Final[float] = 0.3081

# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

LEARNING_RATE: Final[float] = 1e-3
BATCH_SIZE: Final[int] = 128
EPOCHS: Final[int] = 5

#: Hard clean-accuracy gate. A model below this halts the pipeline (CLAUDE.md section 4):
#: it is never silently retrained, re-seeded until it passes, or dropped.
MIN_CLEAN_ACCURACY: Final[float] = 0.97

# --------------------------------------------------------------------------------------
# Threat model / attacks
# --------------------------------------------------------------------------------------

ATTACKS: Final[tuple[str, ...]] = ("fgsm", "pgd")

#: L-infinity budgets in [0, 1] pixel space.
EPSILONS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3)

PIXEL_MIN: Final[float] = 0.0
PIXEL_MAX: Final[float] = 1.0

PGD_STEPS: Final[int] = 40

#: PGD step size is ``PGD_STEP_SIZE_FACTOR * epsilon / PGD_STEPS`` (= 2.5 eps / 40).
PGD_STEP_SIZE_FACTOR: Final[float] = 2.5
PGD_RANDOM_START: Final[bool] = True

#: Slack when asserting the epsilon-ball constraint, to absorb float32 rounding only.
#: These assertions stay enabled in production runs.
EPS_BALL_TOLERANCE: Final[float] = 1e-6

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

#: Gradient norms at or below this are treated as degenerate: cosine similarity is
#: undefined, so the image is excluded and the exclusion is counted and reported rather
#: than silently coerced to zero (CLAUDE.md section 5).
GRAD_NORM_FLOOR: Final[float] = 1e-12

#: Primary sign-agreement variant. "all_pixels" scores strict sign equality over all 784
#: pixels, which is what FGSM transfer actually depends on, since the FGSM perturbation
#: is exactly ``epsilon * sign(grad)``. The "nonzero" variant (pixels where both
#: gradients are non-zero) is also computed, as a robustness check.
SIGN_AGREEMENT_PRIMARY: Final[str] = "all_pixels"
SIGN_AGREEMENT_VARIANTS: Final[tuple[str, ...]] = ("all_pixels", "nonzero")

#: Primary transfer metric for H1-H4: unconditional transfer success, the quantity the
#: threat model's success denominator defines. Conditional transfer success is reported
#: alongside but does not drive the pre-registered verdicts.
PRIMARY_TRANSFER_METRIC: Final[str] = "transfer_success"

# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------

#: Pair-level correlations run over the 20 off-diagonal ordered pairs. n = 20 is small,
#: so every reported correlation ships with a bootstrap confidence interval.
N_BOOTSTRAP: Final[int] = 10_000
CI_LEVEL: Final[float] = 0.95
BOOTSTRAP_SEED: Final[int] = 20_261_123

#: A cell whose transfer successes all exceed this is treated as saturated: between-pair
#: variance has collapsed and correlation claims from it are reported as uninformative
#: rather than dropped quietly (CLAUDE.md section 6, degenerate-regime guard).
SATURATION_THRESHOLD: Final[float] = 0.99

# --------------------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------------------

#: CPU is the default for bit-level reproducibility: MPS does not guarantee deterministic
#: reduction order for all operators. MPS may be enabled by flag for exploration, but
#: every number in the report must come from a CPU run, and the device used is recorded
#: in each run manifest.
DEFAULT_DEVICE: Final[str] = "cpu"

#: Deterministic data loading: single worker, no nondeterministic prefetch interleaving.
DATALOADER_WORKERS: Final[int] = 0

# --------------------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------------------

N_MODELS: Final[int] = len(MODEL_NAMES)

#: Number of off-diagonal ordered (source, target) pairs: the sample size for every
#: pair-level correlation.
N_ORDERED_PAIRS: Final[int] = N_MODELS * (N_MODELS - 1)


def pgd_step_size(epsilon: float) -> float:
    """Return the PGD step size for ``epsilon``: ``2.5 * epsilon / 40``."""
    return PGD_STEP_SIZE_FACTOR * epsilon / PGD_STEPS


def config_hash() -> str:
    """Return a short stable hash of every experiment constant defined here.

    Recorded in each run manifest so a results directory can be matched to the exact
    configuration that produced it. Paths and device are excluded: they affect where a
    run happens, not what is measured.
    """
    payload = {
        "model_names": MODEL_NAMES,
        "n_replicates": N_REPLICATES,
        "model_seed_base": MODEL_SEED_BASE,
        "model_seed_replicate_stride": MODEL_SEED_REPLICATE_STRIDE,
        "model_seed_model_stride": MODEL_SEED_MODEL_STRIDE,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "split_seed": SPLIT_SEED,
        "eval_set_size": EVAL_SET_SIZE,
        "mnist_mean": MNIST_MEAN,
        "mnist_std": MNIST_STD,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "min_clean_accuracy": MIN_CLEAN_ACCURACY,
        "attacks": ATTACKS,
        "epsilons": EPSILONS,
        "pgd_steps": PGD_STEPS,
        "pgd_step_size_factor": PGD_STEP_SIZE_FACTOR,
        "pgd_random_start": PGD_RANDOM_START,
        "sign_agreement_primary": SIGN_AGREEMENT_PRIMARY,
        "primary_transfer_metric": PRIMARY_TRANSFER_METRIC,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]
