"""Run-level contact groups require an explicit articulation owner when multi."""

from __future__ import annotations

import pytest

from unigym.contacts.base import ContactGroupSpec
from unigym.contacts.manager import build_contact_manager

_ARTS = ("cartpole", "g1")


def _groups():
    return {
        "cartpole": {"all": ("<all>",)},
        "g1": {"all": ("<all>",)},
    }


def test_multi_articulation_group_without_owner_fails_with_the_declared_set():
    """Omitting the owner in a multi-articulation run is a teaching error."""
    with pytest.raises(ValueError, match="must name its owner") as err:
        build_contact_manager(
            [{"name": "feet", "sensor_bodies": ["foot"], "frame": "world"}],
            backend="isaaclab2",
            articulation_names=_ARTS,
        )

    message = str(err.value)
    assert "cartpole" in message and "g1" in message
    assert "declares articulations" in message


def test_multi_articulation_group_with_unknown_owner_fails():
    """An explicit owner must be one of the run's articulations."""
    with pytest.raises(ValueError, match="not declared by the run") as err:
        build_contact_manager(
            [
                {
                    "name": "feet",
                    "articulation": "ghost",
                    "sensor_bodies": ["foot"],
                    "frame": "world",
                }
            ],
            backend="isaaclab2",
            articulation_names=_ARTS,
        )

    assert "ghost" in str(err.value)


def test_single_articulation_omitted_owner_stays_implicit():
    """A one-articulation run keeps the historical owner-less group."""
    manager = build_contact_manager(
        [{"name": "feet", "sensor_bodies": ["foot"], "frame": "world"}],
        backend="isaaclab2",
        articulation_names=("cartpole",),
    )

    assert manager is not None
    assert manager.specs[0].articulation is None


def test_reference_owner_is_recorded_and_accepted_for_a_multi_run():
    """The contact_group reference is the explicit owner of the resolved group."""
    manager = build_contact_manager(
        [
            {
                "name": "humanoid",
                "sensor_bodies": {"contact_group": {"articulation": "g1", "group": "all"}},
                "frame": "world",
            }
        ],
        backend="isaaclab2",
        contact_groups=_groups(),
        articulation_names=_ARTS,
    )

    assert manager is not None
    assert manager.specs[0].articulation == "g1"
    assert manager.specs[0].sensor_bodies == ["<all>"]


def test_duplicate_run_articulation_names_fail():
    """The articulation order cannot contain duplicate names."""
    with pytest.raises(ValueError, match="duplicate articulation names"):
        build_contact_manager(
            [{"name": "feet", "sensor_bodies": ["foot"], "frame": "world"}],
            backend="isaaclab2",
            articulation_names=("g1", "g1"),
        )


@pytest.mark.parametrize("value", ["", " g1", 3])
def test_contact_group_spec_rejects_an_invalid_articulation(value):
    """The owner field is a non-empty, trimmed string or None."""
    with pytest.raises(ValueError, match="articulation must be"):
        ContactGroupSpec(
            name="feet",
            sensor_bodies=["foot"],
            frame="world",
            articulation=value,
        )
