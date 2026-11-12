"""The extra-info contract no longer carries foot-derived keys."""

from __future__ import annotations

import inspect

from unigym.env.backends.base import EXTRA_INFO_KEYS, check_extra_info


def test_foot_derived_extra_keys_are_removed():
    """foot_contact / force-z / pos-z are not part of the backend extra contract."""
    assert {"foot_contact", "foot_contact_force_z", "foot_pos_z"}.isdisjoint(EXTRA_INFO_KEYS)


def test_check_extra_info_has_no_foot_count_argument():
    """The validator needs joints only; feet are semantic metadata."""
    params = inspect.signature(check_extra_info).parameters

    assert "n_feet" not in params
    assert list(params) == ["extra", "num_envs", "n_joints", "device"]
