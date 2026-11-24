"""Public contract for the core environment step boundary type."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest
import torch

import unigym.evaluation as evaluation
from unigym.evaluation import EnvStepContext, EnvView
from unigym.evaluation.step import EnvStepContext as StepModuleContext
from unigym._typing import JsonValue
from unigym.values import ValueStore, ValueView


class _DuckValues:
    """A non-store object that satisfies only the read-only value protocol."""

    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self._values = values

    def get(self, name: str) -> torch.Tensor | None:
        return self._values.get(name)

    def snapshot(self) -> dict[str, torch.Tensor]:
        return dict(self._values)

    def __contains__(self, name: object) -> bool:
        return name in self._values


def _view(num_envs: int = 2) -> EnvView:
    return EnvView(record={}, obs=None, env_id=None, num_envs=num_envs, backend="fake")


def _context(**overrides: object) -> EnvStepContext:
    num_envs = int(overrides.pop("num_envs", 2))
    view = overrides.pop("view", None)
    values = overrides.pop("values", None)
    kwargs: dict[str, object] = {
        "view": _view(num_envs) if view is None else view,
        "values": ValueStore(num_envs, "cpu").snapshot() if values is None else values,
        "t": 0,
        "step_index": torch.zeros(num_envs, dtype=torch.int64),
        "time_s": None,
        "cmd": None,
        "reset_state": {},
    }
    kwargs.update(overrides)
    return EnvStepContext(**kwargs)  # type: ignore[arg-type]


def test_env_step_context_is_publicly_exported() -> None:
    """The boundary type is re-exported from the evaluation package."""
    assert evaluation.EnvStepContext is StepModuleContext
    assert "EnvStepContext" in evaluation.__all__


def test_field_order_is_the_public_boundary() -> None:
    """The packed subclass relies on this exact base field order."""
    assert [field.name for field in fields(EnvStepContext)] == [
        "view",
        "values",
        "t",
        "step_index",
        "time_s",
        "cmd",
        "reset_state",
    ]


def test_context_is_a_frozen_slots_value() -> None:
    """The boundary has no per-instance dict and rejects mutation."""
    context = _context()

    assert not hasattr(context, "__dict__")
    assert EnvStepContext.__slots__ == tuple(field.name for field in fields(EnvStepContext))
    with pytest.raises(FrozenInstanceError):
        context.t = 1  # type: ignore[misc]


def test_num_envs_and_device_are_derived_from_the_view() -> None:
    """Properties mirror the borrowed view, with a normalized device."""
    context = _context(num_envs=5)

    assert context.num_envs == 5
    assert context.device == torch.device("cpu")
    assert isinstance(context.device, torch.device)


def test_values_boundary_is_the_read_only_protocol() -> None:
    """The field accepts any ``ValueView``, not a concrete store."""
    duck = _DuckValues({"x": torch.zeros(2)})
    context = _context(values=duck)

    assert isinstance(context.values, ValueView)
    assert context.values is duck
    assert not isinstance(context.values, ValueStore)


def test_reset_state_is_an_immutable_mapping() -> None:
    """The public field cannot be mutated after construction."""
    context = _context(reset_state={0: {"done": True}})

    assert isinstance(context.reset_state, MappingProxyType)
    with pytest.raises(TypeError):
        context.reset_state[1] = {"late": True}  # type: ignore[index]


@pytest.mark.parametrize(
    "name",
    [
        "events",
        "terminals",
        "Termination",
        "BenchmarkStepContext",
        "TickFrame",
        "Probe",
        "Signal",
    ],
)
def test_public_surface_has_no_pack_or_watcher_symbols(name: str) -> None:
    """Core's step boundary owns no pack, terminal or watcher vocabulary."""
    assert not hasattr(EnvStepContext, name)
    assert not hasattr(evaluation, name)
    assert name not in evaluation.__all__


def test_reset_payload_values_are_json_boundary_data() -> None:
    """Reset values are JSON trees, never live tensors."""
    context = _context(reset_state={1: [1, "two", {"three": None}]})

    value: JsonValue = context.reset_state[1]
    assert value == [1, "two", {"three": None}]
