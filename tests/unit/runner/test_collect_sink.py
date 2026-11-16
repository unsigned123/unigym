"""CollectSink B1 fallback records joint-addressed action entries deterministically.

The full spec+prop-derived effective-group projection lands in B2; this lane
pins the compile-safe per-entry fallback key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym.runner.sinks.collect import CollectSink, action_group_contract
from unigym.runtime.actions import NamedAction


def _sink():
    records: list[dict] = []
    writer = SimpleNamespace(meta={}, append=records.append)
    sink = CollectSink(
        writer=writer,
        logger=None,
        filter_chain=[],
        language_gen=None,
        lang_interval=1,
        recorder=SimpleNamespace(enabled=False),
        adapter=SimpleNamespace(),
        env=SimpleNamespace(controls={}),
        num_envs=1,
        total_steps=2,
        device="cpu",
        policy_dt=0.02,
        out_dir=".",
        seed=0,
        episode_length_s=1.0,
        dur_lo=1.0,
        dur_hi=1.0,
        command_dim=0,
    )
    return sink, records


def _bundle(targets, t):
    return {
        "t": t,
        "raw": None,
        "targets": targets,
        "obs_collect": torch.zeros(1, 1),
        "extra": {
            "time_in_episode": torch.tensor([1.0]),
            "episode_id": torch.tensor([0], dtype=torch.int64),
        },
        "cmd": None,
    }


def _legs(articulation, values):
    return NamedAction(
        articulation=articulation,
        values=torch.tensor([values], dtype=torch.float32),
        joint_names=("hip", "knee"),
        kind="position",
    )


def _body(joint_names):
    return NamedAction(
        articulation="arm",
        values=torch.zeros(1, len(joint_names)),
        joint_names=joint_names,
        kind="position",
    )


def _hand():
    return NamedAction(
        articulation="arm",
        values=torch.zeros(1, 1),
        joint_names=("finger",),
        kind="position",
    )


def test_action_contract_ignores_sequence_order():
    """The same entries in a different tuple order yield the identical contract."""
    go1 = _legs("go1", [1.0, 2.0])
    go2 = _legs("go2", [3.0, 4.0])

    assert action_group_contract((go1, go2)) == action_group_contract((go2, go1))


def test_entry_keys_are_articulation_plus_ordered_joint_names():
    """B1 fallback keys are deterministic from articulation + joint names."""
    groups = action_group_contract((_hand(), _body(("shoulder", "elbow"))))

    assert [group["key"] for group in groups] == ["arm/finger", "arm/shoulder,elbow"]
    assert [group["column"] for group in groups] == [
        "action_target/arm/finger",
        "action_target/arm/shoulder,elbow",
    ]
    assert groups[0]["articulation"] == "arm"
    assert groups[0]["joint_names"] == ["finger"]
    assert groups[0]["kind"] == "position"


def test_same_named_joints_across_articulations_stay_distinct():
    """go1/go2 identical joint names produce distinct fallback keys."""
    groups = action_group_contract((_legs("go2", [3.0, 4.0]), _legs("go1", [1.0, 2.0])))

    assert [group["key"] for group in groups] == ["go1/hip,knee", "go2/hip,knee"]
    assert [group["column"] for group in groups] == [
        "action_target/go1/hip,knee",
        "action_target/go2/hip,knee",
    ]
    assert [group["articulation"] for group in groups] == ["go1", "go2"]


def test_action_contract_rejects_a_duplicate_entry_key():
    """One articulation + joint set + kind addressed twice is ambiguous."""
    go1 = _legs("go1", [1.0, 2.0])

    with pytest.raises(ValueError, match="more than one NamedAction entry"):
        action_group_contract((go1, _legs("go1", [5.0, 6.0])))


def test_collect_sink_accepts_reordered_targets():
    """A mid-run reordered sequence does not trip the entry-change guard."""
    sink, records = _sink()
    go1 = _legs("go1", [1.0, 2.0])
    go2 = _legs("go2", [3.0, 4.0])

    sink.on_step(0, _bundle((go1, go2), 0))
    sink.on_step(1, _bundle((go2, go1), 1))

    assert [group["column"] for group in sink.writer.meta["action_groups"]] == [
        "action_target/go1/hip,knee",
        "action_target/go2/hip,knee",
    ]
    assert records[1]["action_target/go1/hip,knee"].tolist() == [[1.0, 2.0]]
    assert records[1]["action_target/go2/hip,knee"].tolist() == [[3.0, 4.0]]
