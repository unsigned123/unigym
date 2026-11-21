"""Public contract: RUNTIME_QUERY trait, registry factory and run.query gate."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from unigym.backends.registry import Trait, backend_info, validate_trait_rules
from unigym.query.errors import QueryStaleError
from unigym.run.instance import RunInstance

_QUERY_BACKENDS = ("mujoco", "mjwarp", "isaaclab2", "isaaclab3_physx", "isaaclab3_newton")


def _instance(backend: str, env: object = None) -> RunInstance:
    """A RunInstance exposing only the fields the query gate reads."""
    return RunInstance(
        spec=SimpleNamespace(backend=backend),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=env,
        bridge=None,
    )


def test_all_backends_declare_runtime_query_with_a_factory() -> None:
    """The trait and its lazy factory are declared together on every backend."""
    for name in _QUERY_BACKENDS:
        info = backend_info(name)
        assert Trait.RUNTIME_QUERY in info.traits
        assert info.query_backend is not None


def test_trait_without_factory_fails_registration() -> None:
    """A backend cannot advertise the trait without a query factory."""
    broken = replace(backend_info("mujoco"), query_backend=None)

    with pytest.raises(RuntimeError, match="RUNTIME_QUERY.*query_backend"):
        validate_trait_rules({"mujoco": broken})


def test_factory_without_trait_fails_registration() -> None:
    """A backend cannot carry a query factory without the trait."""
    info = backend_info("isaaclab2")
    without_trait = frozenset(trait for trait in info.traits if trait is not Trait.RUNTIME_QUERY)
    broken = replace(info, traits=without_trait, query_backend=info.query_backend)

    with pytest.raises(RuntimeError, match="query_backend.*RUNTIME_QUERY"):
        validate_trait_rules({"isaaclab2": broken})


def test_run_query_is_gated_by_the_registry_trait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend without the trait raises a teaching NotImplementedError."""
    import unigym.run.instance as instance_module

    info = backend_info("isaaclab2")
    without_trait = frozenset(trait for trait in info.traits if trait is not Trait.RUNTIME_QUERY)
    monkeypatch.setattr(
        instance_module,
        "backend_info",
        lambda _name: replace(info, traits=without_trait),
    )

    with pytest.raises(NotImplementedError, match="RUNTIME_QUERY"):
        _instance("isaaclab2").query


def test_run_query_reuses_the_env_query_handle() -> None:
    """run.query returns the exact env.query object when the backend exposes it."""
    handle = object()
    env = SimpleNamespace(query=handle)
    instance = _instance("isaaclab2", env=env)

    assert instance.query is handle
    assert instance.query is env.query


def test_run_query_rejects_a_closed_run_before_constructing() -> None:
    """A closed run cannot mint a query handle."""
    instance = _instance("mujoco")
    instance.close()

    with pytest.raises(QueryStaleError, match="closed"):
        instance.query
