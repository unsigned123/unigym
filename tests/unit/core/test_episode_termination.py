"""Episode termination facts are owned by the core module."""

from __future__ import annotations

import unigym.core.episode_termination as core_termination


def test_canonical_module_exports_the_fact_vocabulary():
    """The canonical module is the owner of every public termination symbol."""
    assert set(core_termination.__all__) == {
        "EpisodeOutcome",
        "TerminationSource",
        "TerminationExtension",
        "Termination",
        "EpisodeTermination",
        "OUTCOME_PRIORITY",
        "validate_code",
        "ensure_json_value",
    }
    for name in core_termination.__all__:
        assert hasattr(core_termination, name)
