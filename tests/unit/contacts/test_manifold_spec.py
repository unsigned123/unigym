"""ContactGroupSpec.manifold opt-in validation.

A manifold read is only meaningful for a filtered (pairwise) group; the flag is
off by default so existing force/obs groups keep their behavior unchanged.
"""

from __future__ import annotations

import pytest

from unigym.contacts.base import ContactGroupSpec
from unigym.contacts.refs import ContactBodyRef


def _pairwise(**overrides) -> ContactGroupSpec:
    values = {
        "name": "grip",
        "sensor_refs": [ContactBodyRef(owner="robot", body="tip")],
        "filter_refs": [ContactBodyRef(owner="robot", body="ball")],
        "frame": "world",
    }
    values.update(overrides)
    return ContactGroupSpec(**values)


def test_manifold_defaults_off_and_keeps_a_general_group_unchanged() -> None:
    """Omitting manifold leaves a general group a force-only group."""
    spec = ContactGroupSpec(
        name="feet",
        sensor_refs=[ContactBodyRef(owner="robot", body="sole")],
        frame="world",
    )

    assert spec.manifold is False
    assert spec.mode == "general"


def test_manifold_accepts_a_pairwise_filter_group() -> None:
    """A pairwise group is the only shape a filtered manifold can come from."""
    spec = _pairwise(manifold=True)

    assert spec.manifold is True
    assert spec.mode == "pairwise"


def test_manifold_true_rejects_a_general_group() -> None:
    """A general group has no filter axis to report a manifold from."""
    with pytest.raises(ValueError, match="manifold: true requires a pairwise"):
        ContactGroupSpec(
            name="feet",
            sensor_refs=[ContactBodyRef(owner="robot", body="sole")],
            frame="world",
            manifold=True,
        )


def test_manifold_rejects_a_non_bool_flag() -> None:
    """The opt-in is an explicit bool, not a truthy integer."""
    with pytest.raises(ValueError, match="manifold must be a bool"):
        _pairwise(manifold=1)


def test_from_yaml_accepts_the_manifold_key() -> None:
    """The yaml entry declares the opt-in with the same construction rule."""
    spec = ContactGroupSpec.from_yaml(
        {
            "name": "grip",
            "sensor_refs": [{"owner": "robot", "body": "tip"}],
            "filter_refs": [{"owner": "robot", "body": "ball"}],
            "frame": "world",
            "manifold": True,
        }
    )

    assert spec.manifold is True
