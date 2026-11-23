"""Compatibility re-export of the canonical episode termination facts.

The canonical owner is :mod:`unigym.core.episode_termination`.  This module keeps
the historical ``unigym.benchmarks.termination`` import path valid for external
callers; it defines no facts of its own.
"""

from __future__ import annotations

from unigym.core.episode_termination import (  # noqa: F401
    OUTCOME_PRIORITY,
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
    ensure_json_value,
    validate_code,
)
from unigym.core.episode_termination import (  # noqa: F401
    code_counts,
    outcome_counts,
    source_counts,
)
from unigym.core.episode_termination import __all__ as __all__  # noqa: F401
