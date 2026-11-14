"""Unit invariants for the process-global adapter registry."""

from __future__ import annotations

import pytest

from unigym.adapters import (
    Adapter,
    ManualAdapter,
    adapter_params_registry,
    register_adapter,
    restore_adapter_registry,
    snapshot_adapter_registry,
)


class _NamedAdapter(Adapter):
    def __init__(self, num_envs=1, device="cpu", **kwargs):
        pass

    def get_obs_requests(self):
        return []

    @property
    def obs_dim(self):
        return 0

    @property
    def control_dt(self):
        return 0.02

    def compute_obs(self, obs_cache, cmd, env):
        return None

    def act(self, obs):
        return None

    def postprocess_action(self, raw):
        return {}


class _NotAdapter:
    pass


@pytest.fixture(autouse=True)
def _restore_registry():
    """Restore the process-global registry and origins after every test."""
    snapshot = snapshot_adapter_registry()
    yield
    restore_adapter_registry(snapshot)


@pytest.mark.parametrize(
    "target",
    [_NamedAdapter(), lambda: _NamedAdapter()],
    ids=["instance", "factory"],
)
def test_register_adapter_rejects_non_class_target(target: object) -> None:
    with pytest.raises(TypeError, match="must be a class"):
        register_adapter("acme.bad", target)


def test_register_adapter_rejects_non_adapter_class() -> None:
    with pytest.raises(TypeError, match="must subclass Adapter"):
        register_adapter("acme.bad", _NotAdapter)


def test_register_adapter_duplicate_names_the_existing_class() -> None:
    register_adapter("acme.duplicate", _NamedAdapter)

    with pytest.raises(ValueError) as info:
        register_adapter("acme.duplicate", _NamedAdapter)

    message = str(info.value)
    assert "acme.duplicate" in message
    assert _NamedAdapter.__name__ in message
    assert _NamedAdapter.__module__ in message


def test_register_adapter_rejects_one_class_under_two_names() -> None:
    register_adapter("acme.first", _NamedAdapter)

    with pytest.raises(ValueError) as info:
        register_adapter("acme.alias", _NamedAdapter)

    assert "already registered as 'acme.first'" in str(info.value)


def test_adapter_params_registry_sees_dynamic_registration() -> None:
    register_adapter("acme.params", _NamedAdapter)

    assert adapter_params_registry()["acme.params"]["declared"] is False


def test_manual_adapter_has_a_core_origin() -> None:
    from unigym.adapters import adapter_origin
    from unigym.provenance import ProviderOrigin

    assert adapter_origin("manual") == ProviderOrigin(kind="core")


@pytest.mark.parametrize(
    "name",
    ["", "Acme", "acme-policy", ".acme", "acme.", "acme..policy", "_acme"],
)
def test_register_adapter_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="adapter name"):
        register_adapter(name, ManualAdapter)
