"""Public contract: RUNTIME_QUERY trait, registry factory and run.query gate."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from unigym.backends.compose import BackendRuntime, backend_runtime, validate_backend_runtime
from unigym.capabilities.traits import Trait
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
        runtime = backend_runtime(name)
        assert Trait.RUNTIME_QUERY in runtime.spec.traits
        assert runtime.bindings.query_backend is not None


def test_trait_without_factory_fails_registration() -> None:
    """A backend cannot advertise the trait without a query factory."""
    runtime = backend_runtime("mujoco")
    broken = BackendRuntime(
        spec=runtime.spec, bindings=replace(runtime.bindings, query_backend=None)
    )

    with pytest.raises(RuntimeError, match="RUNTIME_QUERY.*query_backend"):
        validate_backend_runtime(broken)


def test_factory_without_trait_fails_registration() -> None:
    """A backend cannot carry a query factory without the trait."""
    runtime = backend_runtime("isaaclab2")
    without_trait = frozenset(
        trait for trait in runtime.spec.traits if trait is not Trait.RUNTIME_QUERY
    )
    broken = BackendRuntime(
        spec=replace(runtime.spec, traits=without_trait), bindings=runtime.bindings
    )

    with pytest.raises(RuntimeError, match="query_backend.*RUNTIME_QUERY"):
        validate_backend_runtime(broken)


def test_run_query_is_gated_by_the_registry_trait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend without the trait raises a teaching NotImplementedError."""
    import unigym.run.instance as instance_module
    from unigym.capabilities.specs import backend_spec as real_backend_spec

    info = real_backend_spec("isaaclab2")
    without_trait = frozenset(trait for trait in info.traits if trait is not Trait.RUNTIME_QUERY)
    monkeypatch.setattr(
        instance_module,
        "backend_spec",
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
