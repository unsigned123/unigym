"""Strict (articulation, group) adapter group keys and action validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests._support.fakes.backend import FakeBackend
from unigym.run.build import _build_adapter_groups
from unigym.runner.sync import SyncRunner, _adapter_groups
from unigym.runtime.actions import NamedAction
from unigym.runtime.controls import ControlGroup
from unigym.terminations import TerminationGroup


def _control(name="body", adapter="a", articulation="arm"):
    return ControlGroup(
        name=name,
        adapter=adapter,
        articulation=articulation,
        drive="position_pd",
        target="position",
        joints=("shoulder",),
    )


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


def test_build_adapter_groups_partitions_controls_by_identity():
    """The config controls[] partition yields canonical (articulation, group) tuples."""
    spec = SimpleNamespace(
        controls=(
            _control("body", "a", "arm"),
            _control("hand", "a", "arm"),
            _control("legs", "b", "dog"),
        )
    )

    assert _build_adapter_groups(spec) == {
        "a": (("arm", "body"), ("arm", "hand")),
        "b": (("dog", "legs"),),
    }


def test_build_adapter_groups_rejects_a_group_claimed_by_two_adapters():
    """A control group cannot belong to two adapters."""
    spec = SimpleNamespace(controls=(_control("body", "a", "arm"), _control("body", "b", "arm")))

    with pytest.raises(ValueError, match="claimed by two adapters"):
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
