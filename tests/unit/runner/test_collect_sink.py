"""CollectSink action-face contract is addressed by (articulation, group)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym.runner.sinks.collect import CollectSink, action_face_contract
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
        group="legs",
        values=torch.tensor([values], dtype=torch.float32),
        joint_names=("hip", "knee"),
        kind="position",
    )


def _body(joint_names):
    return NamedAction(
        articulation="arm",
        group="body",
        values=torch.zeros(1, len(joint_names)),
        joint_names=joint_names,
        kind="position",
    )


def _hand():
    return NamedAction(
        articulation="arm",
        group="hand",
        values=torch.zeros(1, 1),
        joint_names=("finger",),
        kind="position",
    )


def test_action_face_contract_ignores_sequence_order():
    """The same groups in a different tuple order yield the identical contract."""
    go1 = _legs("go1", [1.0, 2.0])
    go2 = _legs("go2", [3.0, 4.0])

    assert action_face_contract((go1, go2)) == action_face_contract((go2, go1))


def test_unique_group_names_keep_the_historical_column():
    """A group name unique across articulations keeps action_target/<group>."""
    faces = action_face_contract((_hand(), _body(("shoulder", "elbow"))))

    assert [face["key"] for face in faces] == ["body", "hand"]
    assert [face["column"] for face in faces] == ["action_target/body", "action_target/hand"]
    assert faces[0]["articulation"] == "arm"
    assert faces[0]["group"] == "body"


def test_shared_group_names_gain_the_articulation_path():
    """A group name shared by two articulations is disambiguated for uniqueness."""
    faces = action_face_contract((_legs("go2", [3.0, 4.0]), _legs("go1", [1.0, 2.0])))

    assert [face["key"] for face in faces] == ["go1/legs", "go2/legs"]
    assert [face["column"] for face in faces] == [
        "action_target/go1/legs",
        "action_target/go2/legs",
    ]
    assert [face["articulation"] for face in faces] == ["go1", "go2"]


def test_action_face_contract_rejects_a_duplicate_identity():
    """One identity addressed twice is ambiguous."""
    go1 = _legs("go1", [1.0, 2.0])

    with pytest.raises(ValueError, match="more than one NamedAction entry"):
        action_face_contract((go1, _legs("go1", [5.0, 6.0])))


def test_collect_sink_accepts_reordered_targets():
    """A mid-run reordered sequence does not trip the face-change guard."""
    sink, records = _sink()
    go1 = _legs("go1", [1.0, 2.0])
    go2 = _legs("go2", [3.0, 4.0])

    sink.on_step(0, _bundle((go1, go2), 0))
    sink.on_step(1, _bundle((go2, go1), 1))

    assert [face["column"] for face in sink.writer.meta["action_faces"]] == [
        "action_target/go1/legs",
        "action_target/go2/legs",
    ]
    assert records[1]["action_target/go1/legs"].tolist() == [[1.0, 2.0]]
    assert records[1]["action_target/go2/legs"].tolist() == [[3.0, 4.0]]
