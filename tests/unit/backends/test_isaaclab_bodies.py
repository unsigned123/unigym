"""Shared IsaacLab env-site body resolution (no Kit required)."""

from __future__ import annotations

import types

import pytest

from unigym.backends.shared.isaaclab_bodies import (
    body_binding_candidates,
    resolve_site_body_index,
    site_body_index,
)

_BACKENDS = ("isaaclab2 backend", "isaaclab3_physx")


def test_body_binding_candidates_prefers_exact_then_unique_leaf() -> None:
    """An exact live name wins; otherwise the path leaf is matched."""
    names = ["pelvis", "left_hand/left_hand_palm_link", "left_hand_palm_link"]

    assert body_binding_candidates(names, "pelvis") == (0,)
    assert body_binding_candidates(names, "Robot/left_hand/left_hand_palm_link") == (1, 2)


def test_site_body_index_returns_the_single_live_body() -> None:
    """A profile binding that matches one live body resolves to its index."""
    assert site_body_index(["base", "tip"], "tip", "tool") == 1


def test_site_body_index_rejects_an_ambiguous_binding() -> None:
    """Zero or several matches fail instead of picking traversal order."""
    with pytest.raises(ValueError, match=r"does not resolve to exactly one live body"):
        site_body_index(["base", "left/tip", "right/tip"], "tip", "tool")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_env_site_missing_live_body_names_fails_fast(backend: str) -> None:
    """An empty live ``data.body_names`` is a teaching error, never a fallback."""
    data = types.SimpleNamespace(body_names=[])

    with pytest.raises(ValueError, match="exposes no live data.body_names") as caught:
        resolve_site_body_index(
            data,
            usd_name="left_hand_palm_link",
            site_name="tool",
            owner="g1",
            backend=backend,
        )

    message = str(caught.value)
    assert message.startswith(backend)
    assert "'tool'" in message
    assert "'g1'" in message
    assert "'left_hand_palm_link'" in message


@pytest.mark.parametrize("backend", _BACKENDS)
def test_env_site_absent_data_block_fails_fast(backend: str) -> None:
    """A missing data block is the same fail-fast, not an empty resolution."""
    with pytest.raises(ValueError, match="exposes no live data.body_names"):
        resolve_site_body_index(
            None,
            usd_name="tip",
            site_name="tool",
            owner="g1",
            backend=backend,
        )


def test_env_site_resolves_from_live_names_only() -> None:
    """Resolution reads the live ``data.body_names`` order, not an asset list."""
    data = types.SimpleNamespace(body_names=["base", "left_hand_palm_link"])

    index = resolve_site_body_index(
        data,
        usd_name="left_hand_palm_link",
        site_name="tool",
        owner="g1",
        backend="isaaclab2 backend",
    )

    assert index == 1
