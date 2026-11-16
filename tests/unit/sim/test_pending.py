"""PendingActions: one staged entry per group identity, order, lowering, clear."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.packs.packset import PackSet, RobotRef
from unigym.runtime.actions import OrderedActionGroup
from unigym.sim import Control, SimulationBuilder
from unigym.sim.errors import PendingActionError
from unigym.sim.pending import PendingActions, StagedAction
from tests._support.sim_fixtures import write_sim_pack


def _groups(pack_set: PackSet, ref: RobotRef) -> dict[str, OrderedActionGroup]:
    """Two articulation-scoped action groups with distinct group names."""
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    first = builder.add_articulation(
        ref,
        name="a",
        control=Control.position_pd(joint_set="body", group="a", kp=1.0, kd=0.1),
    )
    second = builder.add_articulation(
        ref,
        name="b",
        control=Control.position_pd(joint_set="body", group="b", kp=1.0, kd=0.1),
    )
    return {"a": first.action_groups["a"], "b": second.action_groups["b"]}


def _staged(group: OrderedActionGroup, values: list[list[float]]) -> StagedAction:
    """One staged action with a ``(N, dim)`` float tensor."""
    return StagedAction(group=group, values=torch.tensor(values, dtype=torch.float32))


def test_duplicate_group_identity_is_refused_and_keeps_the_first_entry(
    tmp_path: Path,
) -> None:
    """A second stage for one identity raises and leaves the staged entry intact."""
    pack_set, ref = write_sim_pack(tmp_path)
    groups = _groups(pack_set, ref)
    pending = PendingActions()
    first = _staged(groups["a"], [[0.25, -0.5], [0.75, -0.25]])
    pending.stage(first)

    with pytest.raises(PendingActionError, match="already staged"):
        pending.stage(_staged(groups["a"], [[9.0, 9.0], [9.0, 9.0]]))

    assert pending.groups() == frozenset({("a", "a")})
    lowered = pending.lower()
    assert len(lowered) == 1
    assert lowered[0].articulation == "a"
    assert lowered[0].group == "a"
    assert lowered[0].values is first.values
    assert lowered[0].values.shape == (2, 2)
    assert torch.equal(lowered[0].values, first.values)


def test_lower_preserves_staging_order_and_group_columns(tmp_path: Path) -> None:
    """Lowering keeps insertion order and the group's names/kind per identity."""
    pack_set, ref = write_sim_pack(tmp_path)
    groups = _groups(pack_set, ref)
    pending = PendingActions()
    pending.stage(_staged(groups["b"], [[0.1, 0.2]]))
    pending.stage(_staged(groups["a"], [[0.3, 0.4]]))

    lowered = pending.lower()

    assert [entry.group for entry in lowered] == ["b", "a"]
    by_group = {entry.group: entry for entry in lowered}
    assert by_group["a"].articulation == "a"
    assert by_group["a"].joint_names == groups["a"].joints
    assert by_group["a"].kind == "position"
    assert torch.equal(by_group["a"].values, torch.tensor([[0.3, 0.4]]))


def test_clear_drops_every_group_and_empty_lowers_to_empty(tmp_path: Path) -> None:
    """Clear empties the table; an empty table lowers to an empty tuple."""
    pack_set, ref = write_sim_pack(tmp_path)
    groups = _groups(pack_set, ref)
    pending = PendingActions()
    pending.stage(_staged(groups["a"], [[0.0, 0.0]]))

    pending.clear()

    assert pending.groups() == frozenset()
    assert pending.lower() == ()
