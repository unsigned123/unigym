"""PendingActions: one staged entry per face, insertion-order lowering, clear."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.packs.packset import PackSet, RobotRef
from unigym.sim import ActionBinding, Control, SimulationBuilder
from unigym.sim.errors import PendingActionError
from unigym.sim.pending import PendingActions, StagedAction
from tests._support.sim_fixtures import write_sim_pack


def _bindings(pack_set: PackSet, ref: RobotRef) -> dict[str, ActionBinding]:
    """Two articulation-scoped bindings with distinct global face names."""
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    first = builder.add_articulation(
        ref,
        name="a",
        control=Control.position_pd(joint_set="body", face="a", kp=1.0, kd=0.1),
    )
    second = builder.add_articulation(
        ref,
        name="b",
        control=Control.position_pd(joint_set="body", face="b", kp=1.0, kd=0.1),
    )
    return {
        "a": first.bind_action("a", kind="position"),
        "b": second.bind_action("b", kind="position"),
    }


def _staged(binding: ActionBinding, values: list[list[float]]) -> StagedAction:
    """One staged action with a ``(N, dim)`` float tensor."""
    return StagedAction(binding=binding, values=torch.tensor(values, dtype=torch.float32))


def test_duplicate_face_is_refused_and_keeps_the_first_entry(tmp_path: Path) -> None:
    """A second stage for one face raises and leaves the staged entry intact."""
    pack_set, ref = write_sim_pack(tmp_path)
    bindings = _bindings(pack_set, ref)
    pending = PendingActions()
    first = _staged(bindings["a"], [[0.25, -0.5], [0.75, -0.25]])
    pending.stage("a", first)

    with pytest.raises(PendingActionError, match="already staged"):
        pending.stage("a", _staged(bindings["a"], [[9.0, 9.0], [9.0, 9.0]]))

    assert pending.faces() == frozenset({"a"})
    lowered = pending.lower()
    assert lowered["a"].values is first.values
    assert lowered["a"].values.shape == (2, 2)
    assert torch.equal(lowered["a"].values, first.values)


def test_lower_preserves_staging_order_and_binding_columns(tmp_path: Path) -> None:
    """Lowering keeps insertion order and the binding's names/kind per face."""
    pack_set, ref = write_sim_pack(tmp_path)
    bindings = _bindings(pack_set, ref)
    pending = PendingActions()
    pending.stage("b", _staged(bindings["b"], [[0.1, 0.2]]))
    pending.stage("a", _staged(bindings["a"], [[0.3, 0.4]]))

    lowered = pending.lower()

    assert list(lowered) == ["b", "a"]
    assert lowered["a"].joint_names == list(bindings["a"].joint_names)
    assert lowered["a"].kind == "position"
    assert torch.equal(lowered["a"].values, torch.tensor([[0.3, 0.4]]))


def test_clear_drops_every_face_and_empty_lowers_to_empty(tmp_path: Path) -> None:
    """Clear empties the table; an empty table lowers to an empty dict."""
    pack_set, ref = write_sim_pack(tmp_path)
    bindings = _bindings(pack_set, ref)
    pending = PendingActions()
    pending.stage("a", _staged(bindings["a"], [[0.0, 0.0]]))

    pending.clear()

    assert pending.faces() == frozenset()
    assert pending.lower() == {}
