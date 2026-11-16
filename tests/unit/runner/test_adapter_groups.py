"""Joint-authoritative adapter partitions and runner action validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests._support.fakes.backend import FakeBackend
from unigym.adapters.manual import ManualAdapter
from unigym.run.build import (
    _body_controls,
    _build_adapter_groups,
    _build_adapter_joint_targets,
    _build_effective_action_groups,
)
from unigym.runner.external import ExternalRunner
from unigym.runner.sync import SyncRunner, _adapter_joint_targets
from unigym.runtime.actions import (
    ActionGroupValue,
    NamedAction,
    OrderedActionGroup,
    OrderedActionSpec,
    PolicyBinding,
)
from unigym.runtime.controllers import ControlRuntime
from unigym.terminations import TerminationGroup


def _binding(groups, adapter="a"):
    return PolicyBinding(adapter=adapter, groups=tuple(groups))


def _named(articulation="arm", joints=("shoulder",), kind="position", rows=1):
    return NamedAction(
        articulation=articulation,
        values=torch.zeros(rows, len(joints)),
        joint_names=tuple(joints),
        kind=kind,
    )


def _runner(adapter_joint_targets):
    return SyncRunner(
        cfg={"collect": {"steps": 1}},
        profile=None,
        adapter=SimpleNamespace(_unigym_name="manual"),
        sampler=None,
        term_group=TerminationGroup(),
        env=FakeBackend(),
        sink=SimpleNamespace(),
        adapter_joint_targets=adapter_joint_targets,
    )


def _runtime(name, role, joints=("shoulder",), target="position", adapter="manual"):
    return ControlRuntime(
        name=name,
        adapter=adapter,
        articulation="arm",
        joint_names=joints,
        q0=tuple(0.0 for _ in joints),
        target=target,
        role=role,
    )


def test_body_role_selects_the_body_row_not_the_name():
    """Body selection uses the explicit role, not articulation-name equality."""
    body = _runtime("legs", "body")
    aux = _runtime("arm", "aux")
    env = SimpleNamespace(controls={("arm", "legs"): body, ("arm", "arm"): aux})

    assert _body_controls(env) == [body]


def test_build_adapter_groups_partitions_policies_by_identity():
    """The config policies partition yields canonical (articulation, group) tuples."""
    spec = SimpleNamespace(
        policies=(
            _binding((("arm", "arm"), ("arm", "hand")), adapter="a"),
            _binding((("dog", "legs"),), adapter="b"),
        )
    )

    assert _build_adapter_groups(spec) == {
        "a": (("arm", "arm"), ("arm", "hand")),
        "b": (("dog", "legs"),),
    }


def test_build_adapter_groups_rejects_a_group_claimed_by_two_policies():
    """An action group cannot belong to two policies."""
    spec = SimpleNamespace(
        policies=(
            _binding((("arm", "arm"),), adapter="a"),
            _binding((("arm", "arm"),), adapter="b"),
        )
    )

    with pytest.raises(ValueError, match="claimed by two policies"):
        _build_adapter_groups(spec)


def _spec(*groups):
    return SimpleNamespace(
        action_spec=OrderedActionSpec(tuple(groups)),
        policies=tuple(
            _binding(((group.articulation, group.name),), adapter="a") for group in groups
        ),
    )


def test_build_adapter_joint_targets_derives_joint_kind_partition():
    """Every group joint becomes one adapter-owned (articulation, joint) row."""
    group = OrderedActionGroup("body", "arm", ("shoulder", "elbow"), "position")
    spec = _spec(group)

    assert _build_adapter_joint_targets(spec) == {
        "a": {("arm", "shoulder"): "position", ("arm", "elbow"): "position"}
    }


def test_build_adapter_joint_targets_rejects_an_undeclared_group():
    """A policy binding naming a missing action group fails loudly."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    spec = SimpleNamespace(
        action_spec=OrderedActionSpec((group,)),
        policies=(_binding((("arm", "missing"),), adapter="a"),),
    )

    with pytest.raises(ValueError, match="undeclared action group"):
        _build_adapter_joint_targets(spec)


def _prop_control(name="door", joints=("hinge",), articulation="arm"):
    return ControlRuntime(
        name=name,
        adapter="prop",
        articulation=articulation,
        joint_names=joints,
        q0=tuple(0.0 for _ in joints),
        drive="position_servo",
        target="position",
        required=False,
        prop=True,
    )


def test_build_adapter_joint_targets_includes_prop_controls():
    """Prop control joints join the emitting adapter's partition."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    spec = SimpleNamespace(
        action_spec=OrderedActionSpec((group,)),
        policies=(_binding((("arm", "body"),), adapter="a"),),
        adapter_name="a",
    )

    targets = _build_adapter_joint_targets(spec, controls={("arm", "door"): _prop_control()})

    assert targets == {"a": {("arm", "shoulder"): "position", ("arm", "hinge"): "position"}}


def test_build_adapter_joint_targets_rejects_a_prop_robot_joint_collision():
    """A prop joint may not shadow a robot joint of the same articulation."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    spec = SimpleNamespace(
        action_spec=OrderedActionSpec((group,)),
        policies=(_binding((("arm", "body"),), adapter="a"),),
        adapter_name="a",
    )

    with pytest.raises(ValueError, match="already owns"):
        _build_adapter_joint_targets(
            spec, controls={("arm", "door"): _prop_control(joints=("shoulder",))}
        )


def test_build_effective_action_groups_appends_prop_groups_as_aux():
    """The consumer table is authored groups first, then prop groups as aux."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position", role="body")
    spec = SimpleNamespace(action_spec=OrderedActionSpec((group,)))

    groups = _build_effective_action_groups(spec, controls=[_prop_control()])

    assert [g.name for g in groups] == ["body", "door"]
    assert groups[1].articulation == "arm"
    assert groups[1].joints == ("hinge",)
    assert groups[1].target == "position"
    assert groups[1].role == "aux"


def test_build_effective_action_groups_rejects_a_name_collision():
    """A prop control reusing an authored identity is ambiguous."""
    group = OrderedActionGroup("door", "arm", ("shoulder",), "position")
    spec = SimpleNamespace(action_spec=OrderedActionSpec((group,)))

    with pytest.raises(ValueError, match="declared twice"):
        _build_effective_action_groups(spec, controls=[_prop_control(name="door")])


def test_adapter_joint_targets_reads_the_explicit_partition():
    """An explicit env partition is normalized and returned."""
    env = FakeBackend()
    env.adapter_joint_targets = {"manual": {("arm", "shoulder"): "position"}}
    instance = SimpleNamespace(env=env, adapter=SimpleNamespace(_unigym_name="manual"))

    assert _adapter_joint_targets(instance) == {("arm", "shoulder"): "position"}


@pytest.mark.parametrize(
    ("targets", "match"),
    [
        pytest.param({"arm": "position"}, "must be an .* tuple", id="string-key"),
        pytest.param({("arm",): "position"}, "must be an .* tuple", id="short-key"),
        pytest.param({("arm", "shoulder"): "illegal"}, "illegal target kind", id="bad-kind"),
    ],
)
def test_adapter_joint_targets_rejects_malformed_partitions(targets, match):
    """Malformed adapter partitions fail with a teaching error."""
    env = FakeBackend()
    env.adapter_joint_targets = {"manual": targets}
    instance = SimpleNamespace(env=env, adapter=SimpleNamespace(_unigym_name="manual"))

    with pytest.raises(ValueError, match=match):
        _adapter_joint_targets(instance)


def test_sync_runner_from_instance_rejects_multiple_policy_adapters():
    """SyncRunner v0 is single-adapter; a composite adapter is required."""
    instance = SimpleNamespace(
        spec=SimpleNamespace(runner={}),
        env=FakeBackend(),
        adapter=SimpleNamespace(_unigym_name="manual"),
        profile=None,
        sampler=None,
        term_group=TerminationGroup(),
        sink=SimpleNamespace(),
        reset_on_termination=False,
        replay=None,
    )
    instance.env.adapter_groups = {
        "a": (("arm", "arm"),),
        "b": (("arm", "hand"),),
    }

    with pytest.raises(ValueError, match="exactly one policy adapter"):
        SyncRunner.from_instance(instance)


def test_external_runner_unions_every_adapter_joint_partition():
    """A caller-driven run validates the union of every policy-owned joint."""
    env = FakeBackend()
    env.adapter_joint_targets = {
        "a": {("arm", "shoulder"): "position"},
        "b": {("arm", "finger"): "position"},
    }
    instance = SimpleNamespace(
        spec=SimpleNamespace(runner={}, cfg={"collect": {"steps": 1}}),
        env=env,
        adapter=SimpleNamespace(_unigym_name="manual"),
        profile=None,
        sampler=None,
        term_group=TerminationGroup(),
        sink=SimpleNamespace(),
        reset_on_termination=False,
        replay=None,
    )

    runner = ExternalRunner.from_instance(instance)

    assert runner.adapter_joint_targets == {
        ("arm", "shoulder"): "position",
        ("arm", "finger"): "position",
    }


def test_validate_targets_accepts_the_declared_joint():
    """The matching joint-addressed entry passes validation."""
    runner = _runner({("arm", "shoulder"): "position"})

    runner._validate_targets((_named(),))  # no raise


def test_validate_targets_accepts_split_and_merged_entries():
    """A group may be split across entries or merged into one same-kind entry."""
    runner = _runner(
        {
            ("arm", "shoulder"): "position",
            ("arm", "elbow"): "position",
            ("arm", "finger"): "position",
        }
    )

    split = (
        _named(joints=("shoulder",)),
        _named(joints=("elbow", "finger")),
    )
    merged = (_named(joints=("finger", "shoulder", "elbow")),)

    runner._validate_targets(split)  # no raise
    runner._validate_targets(merged)  # no raise


def test_validate_targets_accepts_separate_entries_for_two_articulations():
    """go1/legs + go2/legs are distinct joint partitions (same joint names)."""
    runner = _runner({("go1", "hip"): "position", ("go2", "hip"): "position"})

    runner._validate_targets((_named("go1", ("hip",)), _named("go2", ("hip",))))  # no raise


def test_validate_targets_accepts_a_permuted_entry():
    """A group's columns may arrive in a different (named) order."""
    runner = _runner({("arm", "shoulder"): "position", ("arm", "elbow"): "position"})

    runner._validate_targets((_named(joints=("elbow", "shoulder")),))  # no raise


def test_validate_targets_rejects_a_duplicate_joint():
    """Two entries addressing the same joint are ambiguous."""
    runner = _runner({("arm", "shoulder"): "position"})

    with pytest.raises(ValueError, match="addressed more than once"):
        runner._validate_targets((_named(), _named()))


def test_validate_targets_rejects_an_unknown_joint():
    """A joint outside the adapter-owned partition fails naming the joint."""
    runner = _runner({("arm", "shoulder"): "position"})

    with pytest.raises(ValueError, match="unknown action joint 'elbow'"):
        runner._validate_targets((_named(joints=("elbow",)),))


def test_validate_targets_rejects_a_wrong_kind():
    """A kind mismatch fails naming articulation, joint and expected kind."""
    runner = _runner({("arm", "shoulder"): "position"})

    with pytest.raises(ValueError, match="joint 'shoulder'.*kind 'velocity'"):
        runner._validate_targets((_named(kind="velocity"),))


def test_validate_targets_rejects_a_missing_joint():
    """Every adapter-owned joint must be covered exactly once."""
    runner = _runner({("arm", "shoulder"): "position", ("arm", "elbow"): "position"})

    with pytest.raises(ValueError, match="unaddressed.*'elbow'"):
        runner._validate_targets((_named(),))


def test_validate_targets_rejects_a_non_named_action_entry():
    """Every sequence entry must be a NamedAction, not a duck-typed object."""
    runner = _runner({("arm", "shoulder"): "position"})

    with pytest.raises(TypeError, match="must be a NamedAction"):
        runner._validate_targets((object(),))


def test_validate_targets_rejects_a_mapping():
    """A mapping action payload is not a sequence of NamedAction."""
    runner = _runner({("arm", "shoulder"): "position"})

    with pytest.raises(TypeError, match="expected a sequence of NamedAction"):
        runner._validate_targets({"arm": _named()})


class _CaptureSink:
    """Minimal sink recording every bundle handed to ``on_step``."""

    def __init__(self):
        self.bundles: list[dict] = []

    def on_step(self, t, bundle):
        self.bundles.append(bundle)

    def on_episode_end(self, env_ids, bundle):
        pass

    def on_step_end(self, t, bundle):
        pass

    def on_run_end(self):
        return {}


def _tick_runner(action_groups):
    sink = _CaptureSink()
    runner = SyncRunner(
        cfg={"collect": {"steps": 1}},
        profile=None,
        adapter=ManualAdapter(),
        sampler=None,
        term_group=TerminationGroup(),
        env=FakeBackend(),
        sink=sink,
        adapter_joint_targets={("arm", "shoulder"): "position"},
        action_groups=action_groups,
    )
    return runner, sink


def test_sync_runner_exposes_the_authoring_projection():
    """bundle[\"action_group_values\"] is the effective-group projection."""
    group = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    runner, sink = _tick_runner((group,))

    runner.tick((_named(rows=1),))

    projection = sink.bundles[0]["action_group_values"]
    assert isinstance(projection, tuple)
    assert len(projection) == 1
    assert isinstance(projection[0], ActionGroupValue)
    assert projection[0].group == group
    assert projection[0].values.tolist() == [[0.0]]


def test_sync_runner_reports_no_projection_without_an_authoring_table():
    """Test doubles without effective groups expose an explicit None."""
    runner, sink = _tick_runner(None)

    runner.tick((_named(rows=1),))

    assert "action_group_values" in sink.bundles[0]
    assert sink.bundles[0]["action_group_values"] is None
