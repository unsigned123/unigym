"""Build-time termination signature validation."""

from __future__ import annotations

import pytest

from unigym.terminations import (
    Termination,
    build_terminations,
    register_termination,
    restore_termination_registry,
    snapshot_termination_registry,
)


class _LegacyTermination(Termination):
    """A term still written against the removed ``check(self, state)`` API."""

    def check(self, state):  # noqa: ANN001 - legacy shape under test
        return state


def test_build_rejects_a_legacy_check_signature_before_the_first_tick():
    """A legacy term fails at build, naming the term and expected signature."""
    snapshot = snapshot_termination_registry()
    register_termination("legacy_check", _LegacyTermination)
    try:
        with pytest.raises(
            ValueError,
            match=r"termination 'legacy_check'.*check\(self, ctx\).*got check\(state\)",
        ):
            build_terminations([{"name": "legacy_check"}])
    finally:
        restore_termination_registry(snapshot)
