"""Strict (articulation, group) adapter group keys and action validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests._support.fakes.backend import FakeBackend
from unigym.run.build import _body_controls, _build_adapter_groups
from unigym.runner.external import ExternalRunner
from unigym.runner.sync import SyncRunner, _adapter_groups
from unigym.runtime.actions import NamedAction, PolicyBinding
from unigym.runtime.controllers import ControlRuntime
from unigym.terminations import TerminationGroup


def _binding(groups, adapter="a"):
    return PolicyBinding(adapter=adapter, groups=tuple(groups))


def _named(articulation="arm", group="arm", rows=1):
    return NamedAction(
        articulation=articulation,
        group=group,
        values=torch.zeros(rows, 1),
        joint_names=("shoulder",),
        kind="position",
    )


def _runner(adapter_groups):
    return SyncRunner(
        cfg={"collect": {"steps": 1}},
        profile=None,
        adapter=SimpleNamespace(_unigym_name="manual"),
        sampler=None,
        term_group=TerminationGroup(),
        env=FakeBackend(),
        sink=SimpleNamespace(),
        adapter_groups=adapter_groups,
    )


def _runtime(name, role):
    return ControlRuntime(
        name=name,
        action_face=name,
        adapter="manual",
        articulation="arm",
        joint_names=("shoulder",),
        q0=(0.0,),
        role=role,
    )


def test_body_role_selects_the_body_group_not_the_name():
    """Body selection uses the explicit role, not articulation-name equality."""
    body = _runtime("legs", "body")
    aux = _runtime("arm", "aux")
    env = SimpleNamespace(controls={"legs": body, "arm": aux})

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


def test_adapter_groups_reads_explicit_identity_tuples():
    """An explicit partition normalizes to the same canonical identities."""
    env = FakeBackend()
    env.adapter_groups = {"manual": (("arm", "arm"),)}
    instance = SimpleNamespace(env=env, adapter=SimpleNamespace(_unigym_name="manual"))

    assert _adapter_groups(instance) == (("arm", "arm"),)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("arm", id="bare-string"),
        pytest.param(["arm", "arm"], id="list-not-tuple"),
        pytest.param(("arm",), id="short-tuple"),
        pytest.param(("arm", "arm", "extra"), id="long-tuple"),
        pytest.param(("", "arm"), id="empty-articulation"),
        pytest.param(("arm", ""), id="empty-group"),
        pytest.param(("arm", 1), id="non-string-group"),
    ],
)
def test_adapter_groups_rejects_non_canonical_keys(entry):
    """Strings, malformed tuples and mixed forms fail with a teaching error."""
    env = FakeBackend()
    env.adapter_groups = {"manual": (entry,)}
    instance = SimpleNamespace(env=env, adapter=SimpleNamespace(_unigym_name="manual"))

    with pytest.raises(ValueError, match=r"articulation, group|bare string|non-string"):
        _adapter_groups(instance)


def test_adapter_groups_rejects_a_mixed_form():
    """A valid tuple next to a legacy string is rejected, never aliased."""
    env = FakeBackend()
    env.adapter_groups = {"manual": (("arm", "arm"), "arm")}
    instance = SimpleNamespace(env=env, adapter=SimpleNamespace(_unigym_name="manual"))

    with pytest.raises(ValueError, match="bare string"):
        _adapter_groups(instance)


def test_adapter_groups_rejects_duplicate_identities():
    """A duplicate (articulation, group) identity is ambiguous."""
    env = FakeBackend()
    env.adapter_groups = {"manual": (("arm", "arm"), ("arm", "arm"))}
    instance = SimpleNamespace(env=env, adapter=SimpleNamespace(_unigym_name="manual"))

    with pytest.raises(ValueError, match="more than once"):
        _adapter_groups(instance)


def test_sync_runner_constructor_rejects_a_legacy_string_group():
    """Direct construction follows the same strict key contract."""
    with pytest.raises(ValueError, match="bare string"):
        _runner(("arm",))


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


def test_external_runner_accepts_every_policy_group():
    """A caller-driven run is not restricted to one policy adapter."""
    env = FakeBackend()
    env.adapter_groups = {
        "a": (("arm", "body"),),
        "b": (("arm", "hand"),),
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

    assert runner.adapter_groups == (("arm", "body"), ("arm", "hand"))


def test_validate_targets_accepts_the_declared_identity():
    """The matching NamedAction identity passes validation."""
    runner = _runner((("arm", "arm"),))

    runner._validate_targets((_named(),))  # no raise


def test_validate_targets_rejects_a_duplicate_identity():
    """Two entries addressing the same group are ambiguous."""
    runner = _runner((("arm", "arm"),))

    with pytest.raises(ValueError, match="more than one action entry"):
        runner._validate_targets((_named(), _named()))


def test_validate_targets_rejects_a_non_named_action_entry():
    """Every sequence entry must be a NamedAction, not a duck-typed object."""
    runner = _runner((("arm", "arm"),))

    with pytest.raises(TypeError, match="must be a NamedAction"):
        runner._validate_targets((object(),))


def test_validate_targets_rejects_a_mapping():
    """A mapping action payload is not a sequence of NamedAction."""
    runner = _runner((("arm", "arm"),))

    with pytest.raises(TypeError, match="expected a sequence of NamedAction"):
        runner._validate_targets({"arm": _named()})
