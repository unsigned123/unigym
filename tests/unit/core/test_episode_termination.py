"""Episode termination facts are owned by the benchmark-neutral core module."""

from __future__ import annotations

import unigym.benchmarks as benchmarks
import unigym.benchmarks.termination as benchmark_termination
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


def test_benchmark_module_reexports_the_same_objects():
    """The historical import path resolves to the canonical objects, not copies."""
    assert benchmark_termination.__all__ == core_termination.__all__
    for name in core_termination.__all__:
        assert getattr(benchmark_termination, name) is getattr(core_termination, name)
    for name in ("code_counts", "outcome_counts", "source_counts"):
        assert getattr(benchmark_termination, name) is getattr(core_termination, name)


def test_benchmarks_package_binds_the_termination_shim():
    """``import unigym.benchmarks`` keeps the historical package attribute."""
    assert benchmarks.termination is benchmark_termination
