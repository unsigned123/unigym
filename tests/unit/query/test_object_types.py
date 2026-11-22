"""Validation and copy semantics of the object-state query result types.

Resolving a name to a compiled free prop is covered in the P2-A backend lane;
it needs a real compiled model and is not faked as a private unit test here.
"""

from __future__ import annotations

import pytest
import torch

from unigym.query.base import ComState, ObjectRef, RigidState
from unigym.query.errors import QueryShapeError


def _ids(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def _rigid_state(rows: int = 1) -> RigidState:
    return RigidState(
        env_ids=_ids(*range(rows)),
        pos_w=torch.zeros(rows, 3, dtype=torch.float64),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * rows, dtype=torch.float64),
        lin_vel_w=torch.zeros(rows, 3, dtype=torch.float64),
        ang_vel_w=torch.zeros(rows, 3, dtype=torch.float64),
    )


def test_object_ref_accepts_a_non_empty_prop_name() -> None:
    """A prop object is addressed by its declared name."""
    assert ObjectRef("crate").name == "crate"


def test_object_ref_rejects_an_empty_name() -> None:
    """There is no unnamed prop object."""
    with pytest.raises(ValueError, match="ObjectRef.name"):
        ObjectRef("")


def test_rigid_state_defaults_to_world_body_origin() -> None:
    """A rigid state reports a world-frame body-origin twist."""
    state = _rigid_state()

    assert state.frame == "world"
    assert state.reference == "body_origin"


def test_rigid_state_copies_its_tensors() -> None:
    """A rigid state is detached from the engine's work buffers."""
    linear = torch.ones(1, 3, dtype=torch.float64)
    state = RigidState(
        env_ids=_ids(0),
        pos_w=torch.zeros(1, 3, dtype=torch.float64),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        lin_vel_w=linear,
        ang_vel_w=torch.zeros(1, 3, dtype=torch.float64),
    )

    linear[0, 0] = 9.0
    assert state.lin_vel_w[0, 0].item() == 1.0


def test_rigid_state_rejects_a_row_count_mismatch() -> None:
    """Every rigid-state tensor has one row per selected env."""
    with pytest.raises(QueryShapeError, match=r"ang_vel_w.*shape \(2, 3\)"):
        RigidState(
            env_ids=_ids(0, 1),
            pos_w=torch.zeros(2, 3, dtype=torch.float64),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64),
            lin_vel_w=torch.zeros(2, 3, dtype=torch.float64),
            ang_vel_w=torch.zeros(1, 3, dtype=torch.float64),
        )


def test_rigid_state_rejects_a_non_finite_channel() -> None:
    """A result never carries a fabricated or non-finite channel."""
    with pytest.raises(QueryShapeError, match="finite"):
        RigidState(
            env_ids=_ids(0),
            pos_w=torch.zeros(1, 3, dtype=torch.float64),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
            lin_vel_w=torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.float64),
            ang_vel_w=torch.zeros(1, 3, dtype=torch.float64),
        )


def test_rigid_state_rejects_a_non_world_frame() -> None:
    """Only the world frame is representable in P2."""
    state = _rigid_state()
    with pytest.raises(QueryShapeError, match="frame"):
        RigidState(
            env_ids=state.env_ids,
            pos_w=state.pos_w,
            quat_wxyz=state.quat_wxyz,
            lin_vel_w=state.lin_vel_w,
            ang_vel_w=state.ang_vel_w,
            frame="body",
        )


def test_rigid_state_rejects_a_non_origin_reference() -> None:
    """A rigid state is only defined at the body-frame origin in P2."""
    state = _rigid_state()
    with pytest.raises(QueryShapeError, match="reference"):
        RigidState(
            env_ids=state.env_ids,
            pos_w=state.pos_w,
            quat_wxyz=state.quat_wxyz,
            lin_vel_w=state.lin_vel_w,
            ang_vel_w=state.ang_vel_w,
            reference="com",
        )


def test_com_state_has_no_orientation_or_angular_velocity() -> None:
    """A center of mass is a point, not an oriented frame."""
    state = ComState(
        env_ids=_ids(0),
        pos_w=torch.zeros(1, 3, dtype=torch.float64),
        lin_vel_w=torch.ones(1, 3, dtype=torch.float64),
    )

    assert not hasattr(state, "quat_wxyz")
    assert not hasattr(state, "ang_vel_w")
    assert state.frame == "world"


def test_com_state_copies_its_tensors() -> None:
    """A COM state cannot mutate the caller's input tensors."""
    position = torch.zeros(1, 3, dtype=torch.float64)
    state = ComState(
        env_ids=_ids(0),
        pos_w=position,
        lin_vel_w=torch.zeros(1, 3, dtype=torch.float64),
    )

    position[0, 2] = 4.0
    assert state.pos_w[0, 2].item() == 0.0


def test_com_state_rejects_a_non_world_frame() -> None:
    """Only the world frame is representable in P2."""
    with pytest.raises(QueryShapeError, match="frame"):
        ComState(
            env_ids=_ids(0),
            pos_w=torch.zeros(1, 3, dtype=torch.float64),
            lin_vel_w=torch.zeros(1, 3, dtype=torch.float64),
            frame="body",
        )
