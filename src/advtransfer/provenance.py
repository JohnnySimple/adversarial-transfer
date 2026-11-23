"""Provenance capture for checkpoints and run manifests.

CLAUDE.md section 8 requires that every artifact record the git SHA, whether the tree was
dirty, the config hash, resolved dependency versions and the device. The point is that a
results directory can always be traced back to the exact code and configuration that
produced it, and that a number produced from an uncommitted tree is visibly marked as
such rather than quietly reported as reproducible.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from advtransfer import config

logger = logging.getLogger(__name__)

#: Marker used when git metadata cannot be determined (e.g. no repository yet).
UNKNOWN = "unknown"


def _git(*args: str) -> str | None:
    """Run a git command in the project root, returning stripped stdout or None."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_sha() -> str:
    """Return the current commit SHA, or ``UNKNOWN`` if it cannot be determined."""
    return _git("rev-parse", "HEAD") or UNKNOWN


def git_is_dirty() -> bool | None:
    """Return whether the working tree has uncommitted changes.

    ``None`` means the question could not be answered (no repository, or git absent),
    which is itself recorded rather than coerced to a confident False.
    """
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return status != ""


def environment_info(device: str = config.DEFAULT_DEVICE) -> dict[str, Any]:
    """Return the environment fields every artifact records."""
    return {
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "config_hash": config.config_hash(),
        "device": device,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def warn_if_dirty() -> None:
    """Log a warning if artifacts are being produced from an uncommitted tree.

    Section 8 forbids a dirty-tree run from backing a reported number. The run is not
    blocked -- that would make iteration painful -- but it is marked in the artifact and
    announced at the time, so it cannot be mistaken later for a clean one.
    """
    if git_is_dirty():
        logger.warning(
            "working tree is dirty; artifacts from this run are marked dirty and must "
            "not back a reported number"
        )
