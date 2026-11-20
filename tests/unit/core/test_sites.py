"""Neutral asset site specs: validation, normalization and ownership rows."""

from __future__ import annotations

import math

import pytest

from unigym.sites import SiteEntry, SiteSpec


def test_site_spec_normalizes_the_orientation_once() -> None:
    """A non-unit authored quaternion is stored as a unit wxyz quaternion."""
    spec = SiteSpec("tcp", "hand", pos=(1.0, 2.0, 3.0), quat_wxyz=(2.0, 0.0, 0.0, 0.0))

    assert spec.pos == (1.0, 2.0, 3.0)
    assert spec.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert math.sqrt(sum(component * component for component in spec.quat_wxyz)) == pytest.approx(
        1.0
    )


def test_site_spec_normalizes_a_very_large_finite_quaternion() -> None:
    """An overflow-scale finite quaternion normalizes without becoming zero."""
    spec = SiteSpec("tcp", "hand", quat_wxyz=(1e308, 1e308, 0.0, 0.0))

    assert spec.quat_wxyz == pytest.approx((2**-0.5, 2**-0.5, 0.0, 0.0))
    assert all(math.isfinite(component) for component in spec.quat_wxyz)


def test_site_spec_rejects_a_quaternion_whose_norm_overflows() -> None:
    """A finite quaternion with an infinite safe norm fails loudly."""
    with pytest.raises(ValueError, match="finite non-zero norm"):
        SiteSpec("tcp", "hand", quat_wxyz=(1e308, 1e308, 1e308, 1e308))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "non-empty"),
        ({"body": ""}, "non-empty"),
        ({"name": "a b"}, "whitespace"),
        ({"pos": (0.0, 0.0)}, "3 finite numbers"),
        ({"pos": (0.0, 0.0, float("inf"))}, "3 finite numbers"),
        ({"quat_wxyz": (0.0, 0.0, 0.0, 0.0)}, "non-zero"),
        ({"quat_wxyz": (1.0, 0.0, 0.0)}, "4 finite numbers"),
    ],
)
def test_site_spec_rejects_malformed_fields(kwargs: dict, match: str) -> None:
    """Invalid names, vectors and zero quaternions fail at construction."""
    values = {"name": "tcp", "body": "hand", **kwargs}
    with pytest.raises(ValueError, match=match):
        SiteSpec(**values)


def test_site_entry_carries_owner_and_kind() -> None:
    entry = SiteEntry("arm", "articulation", SiteSpec("tcp", "hand"))

    assert entry.owner == "arm"
    assert entry.kind == "articulation"
    assert entry.spec.name == "tcp"

    with pytest.raises(ValueError, match="kind must be"):
        SiteEntry("arm", "scene", entry.spec)
    with pytest.raises(ValueError, match="spec must be a SiteSpec"):
        SiteEntry("arm", "prop", "tcp")
