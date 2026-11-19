"""CollectSink records spec+prop-derived action columns (no runtime drift)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym.runner.sinks.collect import CollectSink, action_group_contract
from unigym.runtime.actions import ActionGroupValue, NamedAction, OrderedActionGroup
from unigym.runtime.controllers import ControlRuntime


def _sink(action_groups=None, *, logger=None, language_gen=None, adapter=None, env=None):
    records: list[dict] = []
    writer = SimpleNamespace(meta={}, append=records.append, shard_count=0)
    sink = CollectSink(
        writer=writer,
        logger=logger,
        filter_chain=[],
        language_gen=language_gen,
        lang_interval=1,
        recorder=SimpleNamespace(enabled=False),
        adapter=SimpleNamespace() if adapter is None else adapter,
        env=SimpleNamespace(controls={}) if env is None else env,
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
        action_groups=action_groups,
    )
    return sink, records


def _bundle(targets, t, projection=None, *, step_index=None, time_s=None, reset_state=None):
    bundle = {
        "t": t,
        "raw": None,
        "targets": targets,
        "action_group_values": projection,
        "obs_collect": torch.zeros(1, 1),
        "record": {
            "time_in_episode": torch.tensor([1.0]),
            "episode_id": torch.tensor([0], dtype=torch.int64),
        },
        "cmd": None,
    }
    if step_index is not None:
        bundle["step_index"] = step_index
    if time_s is not None:
        bundle["time_s"] = time_s
    if reset_state is not None:
        bundle["reset_state"] = reset_state
    return bundle


def test_non_filter_path_publishes_episode_local_timeline_and_reset_columns():
    """CollectSink writes tick 1/time dt (not the global t) + reset facts."""
    sink, records = _sink()
    bundle = _bundle(
        (),
        500,  # global run tick must not leak into the episode axis
        step_index=torch.tensor([1], dtype=torch.int64),
        time_s=torch.tensor([0.02], dtype=torch.float32),
        reset_state={0: {"reset_index": 0, "reset_valid": False, "reset_reason": "blocked"}},
    )
    sink.on_step(500, bundle)

    rec = records[0]
    assert rec["step_index"].tolist() == [1]
    assert rec["time_s"].tolist() == pytest.approx([0.02])
    assert rec["reset_index"].tolist() == [0]
    assert rec["reset_valid"].tolist() == [0]  # invalid
    assert rec["reset_reason"] == ["blocked"]


def test_non_filter_path_marks_unknown_reset_validity():
    sink, records = _sink()
    bundle = _bundle(
        (),
        0,
        step_index=torch.tensor([2], dtype=torch.int64),
        time_s=torch.tensor([0.04], dtype=torch.float32),
        reset_state={0: {"reset_index": 0, "reset_valid": None, "reset_reason": None}},
    )
    sink.on_step(0, bundle)
    assert records[0]["reset_valid"].tolist() == [-1]
    assert records[0]["reset_reason"] == [""]


def _body(joint_names, values=None):
    if values is None:
        values = torch.zeros(1, len(joint_names))
    return NamedAction(
        articulation="arm",
        values=values,
        joint_names=joint_names,
        kind="position",
    )


def _legs(articulation, values):
    return NamedAction(
        articulation=articulation,
        values=torch.tensor([values], dtype=torch.float32),
        joint_names=("hip", "knee"),
        kind="position",
    )


def _body_group():
    return OrderedActionGroup("body", "arm", ("shoulder", "elbow"), "position")


def _hand_group():
    return OrderedActionGroup("hand", "arm", ("finger",), "position")


def test_action_contract_is_spec_derived_and_includes_prop_groups():
    """The contract is authored groups + backend-declared prop groups, ordered."""
    prop = OrderedActionGroup("door", "arm", ("hinge",), "position", role="aux")

    contract = action_group_contract((_body_group(), _hand_group(), prop))

    assert [row["key"] for row in contract] == ["body", "hand", "door"]
    assert [row["column"] for row in contract] == [
        "action_target/body",
        "action_target/hand",
        "action_target/door",
    ]
    assert contract[2]["articulation"] == "arm"
    assert contract[2]["group"] == "door"
    assert contract[2]["joint_names"] == ["hinge"]
    assert contract[2]["kind"] == "position"


def test_shared_group_name_across_articulations_keeps_distinct_columns():
    """The same group name on two articulations must not collide in metadata."""
    go1 = OrderedActionGroup("legs", "go1", ("hip",), "position")
    go2 = OrderedActionGroup("legs", "go2", ("hip",), "position")

    contract = action_group_contract((go1, go2))

    assert [row["key"] for row in contract] == ["go1/legs", "go2/legs"]
    assert [row["column"] for row in contract] == [
        "action_target/go1/legs",
        "action_target/go2/legs",
    ]
    assert [row["group"] for row in contract] == ["legs", "legs"]


def test_meta_action_groups_is_the_spec_prop_table_with_no_first_tick_capture():
    """meta[\"action_groups\"] is fixed at construction, not from a runtime tick."""
    prop = OrderedActionGroup("door", "arm", ("hinge",), "position")
    sink, _ = _sink(action_groups=(_body_group(), _hand_group(), prop))

    assert sink.writer.meta["action_groups"] == action_group_contract(
        (_body_group(), _hand_group(), prop)
    )


def test_collect_records_split_and_merged_entries_into_authored_columns():
    """Tolerant projection: split/merged entries land in the authored columns."""
    groups = (_body_group(), _hand_group())
    sink, records = _sink(action_groups=groups)
    targets = (
        NamedAction("arm", torch.tensor([[1.0]]), ("shoulder",), "position"),
        NamedAction("arm", torch.tensor([[2.0, 3.0]]), ("elbow", "finger"), "position"),
    )

    sink.on_step(0, _bundle(targets, 0))

    assert records[0]["action_target/body"].tolist() == [[1.0, 2.0]]
    assert records[0]["action_target/hand"].tolist() == [[3.0]]


def test_collect_ignores_sequence_order_and_has_no_mid_run_drift_guard():
    """A reordered/split tick projects identically; there is no mid-run error."""
    groups = (_body_group(), _hand_group())
    sink, records = _sink(action_groups=groups)
    merged = (_body(("finger", "elbow", "shoulder"), torch.tensor([[3.0, 2.0, 1.0]])),)
    split = (
        _body(("shoulder",), torch.tensor([[1.0]])),
        _body(("elbow",), torch.tensor([[2.0]])),
        _body(("finger",), torch.tensor([[3.0]])),
    )

    sink.on_step(0, _bundle(merged, 0))
    sink.on_step(1, _bundle(split, 1))

    for record in records:
        assert record["action_target/body"].tolist() == [[1.0, 2.0]]
        assert record["action_target/hand"].tolist() == [[3.0]]


def test_collect_uses_the_runner_projection_when_present():
    """The runner-supplied projection wins over projecting targets again."""
    groups = (_body_group(), _hand_group())
    sink, records = _sink(action_groups=groups)
    targets = (
        _body(("shoulder", "elbow"), torch.tensor([[9.0, 9.0]])),
        _body(("finger",), torch.tensor([[9.0]])),
    )
    projection = (
        ActionGroupValue(_body_group(), torch.tensor([[1.0, 2.0]])),
        ActionGroupValue(_hand_group(), torch.tensor([[3.0]])),
    )

    sink.on_step(0, _bundle(targets, 0, projection=projection))

    assert records[0]["action_target/body"].tolist() == [[1.0, 2.0]]
    assert records[0]["action_target/hand"].tolist() == [[3.0]]


def test_collect_records_a_prop_group_column():
    """A declared prop group keeps its historical action_target/<prop> column."""
    prop = OrderedActionGroup("door", "arm", ("hinge",), "position")
    sink, records = _sink(action_groups=(_body_group(), _hand_group(), prop))
    targets = (
        _body(("shoulder", "elbow"), torch.tensor([[1.0, 2.0]])),
        _body(("finger",), torch.tensor([[3.0]])),
        _body(("hinge",), torch.tensor([[4.0]])),
    )

    sink.on_step(0, _bundle(targets, 0))

    assert records[0]["action_target/door"].tolist() == [[4.0]]


@pytest.mark.parametrize(
    ("targets", "match"),
    [
        pytest.param(
            (_body(("shoulder",), torch.tensor([[1.0]])), _body(("finger",), torch.zeros(1, 1))),
            "joint 'elbow' is not addressed",
            id="missing-joint",
        ),
        pytest.param(
            (
                _body(("shoulder", "elbow"), torch.zeros(1, 2)),
                NamedAction("arm", torch.zeros(1, 1), ("finger",), "velocity"),
            ),
            "joint 'finger' has action kind 'velocity'",
            id="wrong-kind",
        ),
    ],
)
def test_collect_projection_errors_name_joints(targets, match):
    """Exact coverage/kind failures from the authoring projection name joints."""
    sink, _ = _sink(action_groups=(_body_group(), _hand_group()))

    with pytest.raises(ValueError, match=match):
        sink.on_step(0, _bundle(targets, 0))


def _policy_runtime(name: str, articulation: str) -> ControlRuntime:
    return ControlRuntime(
        name=name,
        adapter="manual",
        articulation=articulation,
        joint_names=("hip", "knee"),
        q0=(0.0, 0.0),
        drive="position_pd",
        target="position",
        role="body",
    )


def _named_state_extra() -> dict:
    return {
        "articulation/go2/base_lin_vel_b": torch.tensor([[1.0, 2.0, 0.0]]),
        "articulation/go2/base_ang_vel_b": torch.tensor([[0.0, 0.0, 3.0]]),
        "articulation/go2/base_pos_w": torch.tensor([[0.0, 0.0, 0.5]]),
    }


def test_language_state_keys_follow_the_articulation_not_the_group_name():
    """A group name different from the articulation still resolves extra keys."""
    seen: dict = {}

    class _Language:
        def describe(self, states, cmd):
            seen.update(states)
            return ["cmd"] * len(cmd)

    runtime = _policy_runtime("legs", "go2")
    adapter = SimpleNamespace(build_language_states=lambda extra, cmd: dict(extra))
    sink, records = _sink(
        language_gen=_Language(),
        adapter=adapter,
        env=SimpleNamespace(controls={"go2/legs": runtime}),
    )
    bundle = _bundle((), 0)
    bundle["record"].update(_named_state_extra())

    sink.on_step(0, bundle)

    assert records[0]["language"] == ["cmd"]
    assert seen["vx"].tolist() == [1.0]
    assert seen["vy"].tolist() == [2.0]
    assert seen["wz"].tolist() == [3.0]
    assert seen["base_z"].tolist() == [0.5]


def test_step_metrics_state_keys_follow_the_articulation_not_the_group_name():
    """Per-runtime metrics read ``articulation/<articulation>/...`` extra keys."""
    steps: list[dict] = []
    runtime = _policy_runtime("legs", "go2")
    sink, _ = _sink(
        logger=SimpleNamespace(log_step=lambda t, metrics: steps.append(metrics)),
        env=SimpleNamespace(controls={"go2/legs": runtime}),
    )

    sink.on_step_end(
        0,
        {
            "record": _named_state_extra(),
            "alive": torch.tensor([True]),
            "term_metrics": {},
        },
    )

    assert steps[0]["vx"] == 1.0
    assert steps[0]["base_z"] == 0.5


def test_direct_construction_falls_back_to_per_entry_columns():
    """A sink built without an authoring table keeps deterministic entry keys."""
    sink, records = _sink()
    go1 = _legs("go1", [1.0, 2.0])
    go2 = _legs("go2", [3.0, 4.0])

    sink.on_step(0, _bundle((go2, go1), 0))
    sink.on_step(1, _bundle((go1, go2), 1))

    assert [row["column"] for row in sink.writer.meta["action_groups"]] == [
        "action_target/go1/hip,knee",
        "action_target/go2/hip,knee",
    ]
    assert records[0]["action_target/go2/hip,knee"].tolist() == [[3.0, 4.0]]
