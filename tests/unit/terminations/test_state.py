"""TerminationState extraction from live ``extra`` and recorded episode data."""

from __future__ import annotations

import numpy as np

from unigym.terminations import TerminationState


def test_from_extra_groups_articulation_keys_and_aliases_recorded_base_pos():
    """Live extra state is grouped by articulation and base_pos maps to base_pos_w."""
    base = np.array([[0.0, 0.0, 0.5]])
    batch = np.array([3])
    state = TerminationState.from_extra(
        {
            "articulation/g1/base_pos": base,
            "articulation/g1/joint_pos": np.zeros((1, 2)),
            "body_contact_forces": np.zeros((1, 1, 3)),
            "body_names": ["base"],
            "time_in_episode": batch,
        }
    )

    assert state.articulations["g1"]["base_pos_w"] is base
    assert "base_pos" not in state.articulations["g1"]
    assert state.body_names == ["base"]
    assert state.batch is batch


def test_from_episode_data_aliases_flat_recorded_base_pos():
    """Recorded namespaced base_pos is exposed under the canonical base_pos_w leaf."""
    base = np.array([[0.0, 0.0, 0.5]])
    state = TerminationState.from_episode_data(
        {
            "articulation/g1/base_pos": base,
            "articulation/g1/gravity_b": np.array([[0.0, 0.0, -1.0]]),
        }
    )

    assert state.articulations["g1"]["base_pos_w"] is base
    assert "base_pos" not in state.articulations["g1"]


def test_from_episode_data_aliases_supplied_articulation_mapping():
    """A supplied ``articulations`` mapping applies the same base_pos alias."""
    base = np.array([[0.0, 0.0, 0.5]])
    state = TerminationState.from_episode_data({"articulations": {"g1": {"base_pos": base}}})

    assert state.articulations["g1"]["base_pos_w"] is base
    assert "base_pos" not in state.articulations["g1"]
