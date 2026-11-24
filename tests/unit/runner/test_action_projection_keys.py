"""Watcher/trace action keys: authoring projection first, joint fallback second."""

from __future__ import annotations

import numpy as np
import torch

from unigym.runtime.actions import ActionGroupValue, NamedAction, OrderedActionGroup
from unigym.trace.record import build_dump_expr
from unigym.watchers.base import WatchContext


def _entry(articulation="arm", joints=("shoulder",), values=None, kind="position"):
    if values is None:
        values = torch.zeros(1, len(joints))
    return NamedAction(articulation, values, tuple(joints), kind)


def _bundle(targets, projection=None):
    return {"t": 0, "targets": targets, "action_group_values": projection, "raw": None}


def test_watch_keys_use_the_authoring_projection():
    """A unique group name keeps the historical action/<group> key."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    values = torch.tensor([[1.0]])
    ctx = WatchContext(t=0, bundle=_bundle((), (ActionGroupValue(group, values),)))

    assert "action/body" in ctx.keys()
    assert "action/arm/shoulder" not in ctx.keys()
    assert ctx.key("action/body") is values


def test_watch_keys_disambiguate_a_shared_group_name():
    """A shared name is keyed action/<articulation>/<group>."""
    go1 = OrderedActionGroup("legs", "go1", ("hip",), "position")
    go2 = OrderedActionGroup("legs", "go2", ("hip",), "position")
    ctx = WatchContext(
        t=0,
        bundle=_bundle(
            (),
            (
                ActionGroupValue(go1, torch.zeros(1, 1)),
                ActionGroupValue(go2, torch.zeros(1, 1)),
            ),
        ),
    )

    assert "action/go1/legs" in ctx.keys()
    assert "action/go2/legs" in ctx.keys()
    assert "action/legs" not in ctx.keys()


def test_watch_keys_fall_back_to_joint_addressed_entries():
    """Without a projection, one action/<articulation>/<joints> key per entry."""
    entry = _entry("arm", ("elbow", "shoulder"), torch.tensor([[2.0, 1.0]]))
    ctx = WatchContext(t=0, bundle=_bundle((entry,)))

    assert "action/arm/elbow,shoulder" in ctx.keys()
    assert ctx.key("action/arm/elbow,shoulder") is entry.values


def test_trace_expr_records_authoring_projection_keys(tmp_path):
    """The trace dump uses bundle[\"action_group_values\"] when present."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    bundle = _bundle((), (ActionGroupValue(group, torch.tensor([[1.0]])),))
    path = tmp_path / "tick.npz"

    exec(
        build_dump_expr(str(path)),
        {"np": np, "torch": torch, "bundle": bundle, "obs": {}, "record": {}},
    )

    with np.load(path) as data:
        assert "action/body" in data.files
        assert data["action/body"].tolist() == [[1.0]]


def test_trace_expr_falls_back_to_joint_addressed_keys(tmp_path):
    """Without a projection the trace uses the per-entry joint key."""
    entry = _entry("arm", ("shoulder",), torch.tensor([[3.0]]))
    bundle = _bundle((entry,))
    path = tmp_path / "tick.npz"

    exec(
        build_dump_expr(str(path)),
        {"np": np, "torch": torch, "bundle": bundle, "obs": {}, "record": {}},
    )

    with np.load(path) as data:
        assert "action/arm/shoulder" in data.files
        assert data["action/arm/shoulder"].tolist() == [[3.0]]
