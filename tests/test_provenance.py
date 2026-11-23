"""Provenance guards.

The dirty-tree flag matters: section 8 forbids a dirty-tree run from backing a reported
number, which only works if the flag is actually recorded and distinguishable from
"unknown".
"""

from advtransfer import config, provenance


def test_environment_info_has_every_required_field():
    info = provenance.environment_info()
    for field in (
        "git_sha",
        "git_dirty",
        "config_hash",
        "device",
        "python",
        "torch",
        "numpy",
        "platform",
    ):
        assert field in info, field


def test_environment_info_records_current_config_hash():
    assert provenance.environment_info()["config_hash"] == config.config_hash()


def test_environment_info_records_requested_device():
    assert provenance.environment_info(device="mps")["device"] == "mps"


def test_git_sha_is_a_sha_or_explicit_unknown():
    sha = provenance.git_sha()
    assert sha == provenance.UNKNOWN or (
        len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    )


def test_git_dirty_is_tri_state():
    # None means "could not determine" and must stay distinguishable from False, so an
    # unknown provenance is never reported as a clean tree.
    assert provenance.git_is_dirty() in (True, False, None)
