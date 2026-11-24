"""Public package import smoke."""

from __future__ import annotations

import importlib.metadata
import re

import unigym


def test_version_is_public_release_line():
    """The public core reports its own new version line."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", unigym.__version__)
    assert unigym.__version__ == importlib.metadata.version("unigym")


def test_public_names_resolve():
    """Every advertised public name is importable through the lazy facade."""
    for name in unigym.__all__:
        assert getattr(unigym, name) is not None


def test_benchmark_public_names_resolve():
    """The benchmark facade exports its documented task and tick types."""
    import unigym.benchmarks as benchmarks

    for name in ("EpisodeData", "Task", "TickResult", "TickTrace"):
        assert name in benchmarks.__all__, name
        assert getattr(benchmarks, name) is not None, name


def test_evaluation_public_names_resolve():
    """The evaluation context facade exports exactly its advertised names."""
    import unigym.evaluation as evaluation

    assert set(evaluation.__all__) == {
        "EnvStepContext",
        "EnvView",
        "EvalContext",
        "EvalQuery",
        "EvalQueryUnavailableError",
        "EvalViewError",
        "EvalViewStaleError",
    }
    for name in evaluation.__all__:
        assert getattr(evaluation, name) is not None, name
