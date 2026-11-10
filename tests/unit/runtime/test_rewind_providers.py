"""Rewind provider wiring and initializer state capture/restore boundaries."""

from __future__ import annotations

import pytest

from unigym.initializers.base import InitializerBase
from unigym.runtime.rewind_providers import (
    InitializerStateProvider,
    build_rewind_providers,
)


def _initializer(seed: int = 0) -> InitializerBase:
    return InitializerBase({"seed": seed}, None, None)


def test_build_rewind_providers_mounts_only_capable_initializer():
    """An initializer joins the rewind package only when it can restore."""
    capable = _initializer()
    providers = build_rewind_providers(None, initializer=capable)

    assert [provider.name for provider in providers] == ["initializer"]
    assert build_rewind_providers(None, initializer=None) == []
    assert build_rewind_providers(None, initializer=object()) == []


def test_initializer_provider_snapshot_is_isolated_from_live_state():
    """A captured snapshot is a deep copy; mutating it leaves the RNG untouched."""
    initializer = _initializer(seed=11)
    provider = InitializerStateProvider(initializer)
    snapshot = provider.rewind_state()

    snapshot["rng"]["state"]["state"] += 1

    reference = _initializer(seed=11)
    assert initializer.rng.integers(0, 1_000_000) == reference.rng.integers(0, 1_000_000)


def test_initializer_provider_restore_replays_the_same_stream():
    """Applying a captured state replays the initializer's RNG stream exactly."""
    initializer = _initializer(seed=5)
    provider = InitializerStateProvider(initializer)
    state = provider.rewind_state()

    expected = [int(initializer.rng.integers(0, 1000)) for _ in range(3)]
    provider.apply_rewind_state(state)

    assert [int(initializer.rng.integers(0, 1000)) for _ in range(3)] == expected


@pytest.mark.parametrize(
    ("initializer", "state", "match"),
    [
        (_initializer(), None, "rewind state is missing"),
        (object(), {}, "cannot restore rewind state"),
    ],
    ids=["missing-state", "unrestorable-initializer"],
)
def test_initializer_provider_validate_rejects_unrestorable_state(initializer, state, match):
    """Pre-flight refuses state an initializer could not restore."""
    provider = InitializerStateProvider(initializer)

    with pytest.raises(ValueError, match=match):
        provider.validate_rewind_state(state)
