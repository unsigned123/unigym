"""Public contract for the custom-sink construction context."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from unigym._typing import JsonValue
from unigym.run import SinkContext
from unigym.run.spec import build_spec
from unigym.values import ValuePublisher, ValueStore


def _cfg() -> dict[str, JsonValue]:
    return {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }


def _context() -> SinkContext:
    spec = build_spec(_cfg(), Namespace(config="run.yaml"))
    return SinkContext(
        spec=spec,
        mode=spec.mode,
        out_dir="data_out/r",
        num_envs=1,
        policy_dt=0.02,
        seed=0,
        action_groups=(),
        run_identity=None,
        options={},
    )


def test_sink_context_field_order_keeps_values_last() -> None:
    """The published boundary is the exact field sequence with values last."""
    assert [field.name for field in fields(SinkContext)] == [
        "spec",
        "mode",
        "out_dir",
        "num_envs",
        "policy_dt",
        "seed",
        "action_groups",
        "run_identity",
        "options",
        "values",
    ]


def test_sink_context_values_defaults_to_none_and_has_no_signals_field() -> None:
    """The construction context is write-only and exposes no signal face."""
    context = _context()

    assert context.values is None
    assert "signals" not in {field.name for field in fields(SinkContext)}
    assert not hasattr(context, "signals")
    for live_field in ("env", "writer", "logger", "bridge"):
        assert not hasattr(context, live_field)


def test_sink_context_values_is_a_write_only_publisher() -> None:
    """The boundary hands over a publisher backed by the run's value store."""
    store = ValueStore(num_envs=1, device="cpu")
    publisher = store.publisher("sink")
    context = replace(_context(), values=publisher)

    assert isinstance(context.values, ValuePublisher)
    assert not hasattr(publisher, "get")
    assert not hasattr(publisher, "snapshot")
    assert not hasattr(publisher, "rewind_state")
    assert not hasattr(publisher, "apply_rewind_state")
    assert not hasattr(publisher, "__contains__")

    publisher.publish("acme/probe", torch.tensor([[1.0]]))
    stored = store.get("acme/probe")
    assert stored is not None
    assert torch.equal(stored, torch.tensor([[1.0]]))

    publisher.clear("acme/probe")
    assert store.get("acme/probe") is None


def test_sink_context_is_frozen() -> None:
    """Construction metadata cannot be reassigned after the factory receives it."""
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.values = None  # type: ignore[misc]


def test_collect_sink_spec_is_inert_immutable_data() -> None:
    """The built-in collect selection carries no options or live resources."""
    from dataclasses import fields

    from unigym.run import CollectSinkSpec

    declaration = CollectSinkSpec()

    assert (declaration.kind, declaration.name) == ("collect", "collect")
    assert [field.name for field in fields(CollectSinkSpec)] == ["kind", "name"]
    with pytest.raises(FrozenInstanceError):
        declaration.name = "other"  # type: ignore[misc]
    for live_field in ("options", "factory", "env", "writer", "logger"):
        assert not hasattr(declaration, live_field)
