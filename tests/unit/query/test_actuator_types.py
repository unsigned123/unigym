"""Validation and copy semantics of the actuator query response types."""

from __future__ import annotations

import pytest
import torch

from unigym.query.base import ActuatorFacts, ActuatorRuntime
from unigym.query.errors import QueryShapeError


def _ids(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def _facts(**overrides: object) -> ActuatorFacts:
    values: dict[str, object] = {
        "name": "shoulder_motor",
        "target": "shoulder",
        "kind": "robot",
        "slot": 0,
        "slot_kind": "mujoco_actuator",
        "ctrlrange": (-20.0, 20.0),
        "gear": (1.0,),
        "gainprm": (1.0, 0.0),
        "biasprm": (0.0, 0.0),
    }
    values.update(overrides)
    return ActuatorFacts(**values)


def test_actuator_facts_accepts_a_neutral_row() -> None:
    """A compiled actuator row keeps its neutral name and declared values."""
    facts = _facts()

    assert facts.name == "shoulder_motor"
    assert facts.target == "shoulder"
    assert facts.kind == "robot"
    assert facts.slot == 0
    assert facts.slot_kind == "mujoco_actuator"
    assert facts.ctrlrange == (-20.0, 20.0)
    assert facts.gear == (1.0,)


def test_actuator_facts_allows_an_undeclared_row() -> None:
    """An uncompiled or undeclared actuator has no slot and no parameters."""
    facts = _facts(
        slot=None,
        target=None,
        ctrlrange=None,
        gear=None,
        gainprm=None,
        biasprm=None,
    )

    assert facts.slot is None
    assert facts.target is None
    assert facts.ctrlrange is None


def test_actuator_facts_joint_names_default_to_empty() -> None:
    """A backend that publishes no driven-joint list reports an empty tuple."""
    assert _facts().joint_names == ()


def test_actuator_facts_keeps_the_declared_joint_names() -> None:
    """An actuator group keeps its declared joint names in order."""
    facts = _facts(joint_names=("j1", "j2"))

    assert facts.joint_names == ("j1", "j2")


def test_actuator_facts_rejects_a_bare_joint_name_string() -> None:
    """One name is not a sequence: a string would be split into characters."""
    with pytest.raises(QueryShapeError, match=r"joint_names.*sequence"):
        _facts(joint_names="j1")


def test_actuator_facts_rejects_an_empty_joint_name() -> None:
    """A joint name is a non-empty identity, never a placeholder."""
    with pytest.raises(ValueError, match="joint_names"):
        _facts(joint_names=("j1", ""))


def test_actuator_facts_rejects_an_unknown_slot_kind() -> None:
    """The slot namespace is a closed literal, not a free-form tag."""
    with pytest.raises(ValueError, match="slot_kind"):
        _facts(slot_kind="bogus")


def test_actuator_facts_rejects_a_wrong_ctrlrange_width() -> None:
    """A control range is exactly one low/high pair."""
    with pytest.raises(QueryShapeError, match=r"ctrlrange.*pair"):
        _facts(ctrlrange=(1.0, 2.0, 3.0))


def test_actuator_facts_rejects_an_empty_parameter_tuple() -> None:
    """An empty tuple is undeclared, not an empty parameter vector."""
    with pytest.raises(QueryShapeError, match=r"gear.*empty"):
        _facts(gear=())


def test_actuator_facts_rejects_a_negative_slot() -> None:
    """A compiled slot is a non-negative identity."""
    with pytest.raises(QueryShapeError, match="slot"):
        _facts(slot=-1)


def test_actuator_runtime_copies_every_channel() -> None:
    """A runtime result cannot be changed through the caller's tensors."""
    ids = _ids(0, 1)
    ctrl = torch.tensor([0.1, 0.2], dtype=torch.float64)
    runtime = ActuatorRuntime(ids, ctrl=ctrl, force=None, length=None, velocity=None)

    ids[0] = 9
    ctrl[0] = 9.0

    assert runtime.env_ids.tolist() == [0, 1]
    assert runtime.ctrl is not None
    assert runtime.ctrl[0].item() == pytest.approx(0.1)
    assert runtime.force is None


def test_actuator_runtime_rejects_a_row_count_mismatch() -> None:
    """Every channel carries one value per selected env."""
    with pytest.raises(QueryShapeError, match=r"force.*shape \(2,\).*\(1,\)"):
        ActuatorRuntime(_ids(0, 1), force=torch.zeros(1, dtype=torch.float64))


def test_actuator_runtime_rejects_a_non_float_channel() -> None:
    """A runtime channel is a floating tensor, not an integer index vector."""
    with pytest.raises(QueryShapeError, match="ctrl"):
        ActuatorRuntime(_ids(0), ctrl=torch.zeros(1, dtype=torch.int64))


def test_actuator_runtime_allows_an_empty_selection() -> None:
    """An empty env selection yields zero-row channels, not an error."""
    runtime = ActuatorRuntime(_ids(), ctrl=torch.zeros(0, dtype=torch.float64))

    assert runtime.env_ids.numel() == 0
    assert runtime.ctrl is not None
    assert runtime.ctrl.shape == (0,)
