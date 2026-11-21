"""Public contract: RUNTIME_QUERY trait, registry factory and run.query gate."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from unigym.backends.registry import Trait, backend_info, validate_trait_rules
from unigym.run.instance import RunInstance


def _instance(backend: str) -> RunInstance:
    """A RunInstance exposing only the fields the query gate reads."""
    return RunInstance(
        spec=SimpleNamespace(backend=backend),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=None,
        bridge=None,
    )


def test_only_mujoco_declares_runtime_query_with_a_factory() -> None:
    """The trait and its lazy factory are declared together, on MuJoCo only."""
    mujoco = backend_info("mujoco")
    assert Trait.RUNTIME_QUERY in mujoco.traits
    assert mujoco.query_backend is not None

    for name in ("mjwarp", "isaaclab2", "isaaclab3_physx", "isaaclab3_newton"):
        info = backend_info(name)
        assert Trait.RUNTIME_QUERY not in info.traits
        assert info.query_backend is None


def test_trait_without_factory_fails_registration() -> None:
    """A backend cannot advertise the trait without a query factory."""
    broken = replace(backend_info("mujoco"), query_backend=None)

    with pytest.raises(RuntimeError, match="RUNTIME_QUERY.*query_backend"):
        validate_trait_rules({"mujoco": broken})


def test_factory_without_trait_fails_registration() -> None:
    """A backend cannot carry a query factory without the trait."""
    mujoco = backend_info("mujoco")
    broken = replace(backend_info("mjwarp"), query_backend=mujoco.query_backend)

    with pytest.raises(RuntimeError, match="query_backend.*RUNTIME_QUERY"):
        validate_trait_rules({"mjwarp": broken})


def test_run_query_is_gated_by_the_registry_trait() -> None:
    """A backend without the trait raises a teaching NotImplementedError."""
    with pytest.raises(NotImplementedError, match="RUNTIME_QUERY"):
        _instance("isaaclab2").query


def test_run_query_rejects_a_closed_run_before_constructing() -> None:
    """A closed run cannot mint a query handle."""
    from unigym.query.errors import QueryStaleError

    instance = _instance("mujoco")
    instance.close()

    with pytest.raises(QueryStaleError, match="closed"):
        instance.query
