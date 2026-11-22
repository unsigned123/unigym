"""Validation and copy semantics of the neutral query request/response types."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from unigym.query.base import (
    AABB,
    ComRef,
    DistanceResult,
    DofLayout,
    EntityRef,
    Jacobian,
    PointRef,
    Pose,
)
from unigym.query.errors import QueryShapeError


def _ids(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def _quats(*values: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _layout() -> DofLayout:
    base = ("base.px", "base.py", "base.pz", "base.rx", "base.ry", "base.rz")
    return DofLayout(
        articulation="tiny",
        base_dofs=base,
        base_names=("base",),
        joint_names=("shoulder", "elbow"),
        nv=8,
        order=(*base, "shoulder", "elbow"),
    )


def test_pose_copies_its_tensors() -> None:
    """A pose cannot be changed by mutating the caller's input tensors."""
    position = torch.zeros(2, 3, dtype=torch.float64)
    quaternion = _quats((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    pose = Pose(_ids(0, 1), position, quaternion)

    position[0, 0] = 9.0
    assert pose.pos_w[0, 0].item() == 0.0


def test_pose_rejects_a_row_count_mismatch() -> None:
    """Every pose tensor must have one row per selected env."""
    with pytest.raises(QueryShapeError, match=r"pos_w.*shape \(2, 3\).*\(1, 3\)"):
        Pose(_ids(0, 1), torch.zeros(1, 3), _quats((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))


def test_env_ids_reject_a_negative_value() -> None:
    """Output env ids are non-negative row selectors."""
    with pytest.raises(QueryShapeError, match="non-negative"):
        Pose(_ids(-1), torch.zeros(1, 3), _quats((1.0, 0.0, 0.0, 0.0)))


def test_distance_result_requires_a_bool_cap_flag() -> None:
    """The capped horizon flag is a per-env bool tensor."""
    with pytest.raises(QueryShapeError, match="capped"):
        DistanceResult(
            _ids(0),
            torch.tensor([0.5], dtype=torch.float64),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.tensor([1], dtype=torch.int64),
        )


def test_aabb_rejects_reversed_bounds() -> None:
    """An AABB requires min_w <= max_w on every row."""
    with pytest.raises(QueryShapeError, match="min_w <= max_w"):
        AABB(_ids(0), torch.ones(1, 3), torch.zeros(1, 3))


def test_dof_layout_requires_order_length_to_match_nv() -> None:
    """The order vector is the per-column truth and must match nv."""
    with pytest.raises(QueryShapeError, match="order"):
        DofLayout(
            articulation="tiny",
            base_dofs=(),
            base_names=(),
            joint_names=("shoulder",),
            nv=2,
            order=("shoulder",),
        )


def test_dof_layout_requires_order_to_start_with_base_dofs() -> None:
    """A free-base layout documents its six leading columns."""
    with pytest.raises(ValueError, match="base_dofs"):
        DofLayout(
            articulation="tiny",
            base_dofs=("base.px", "base.py", "base.pz", "base.rx", "base.ry", "base.rz"),
            base_names=("base",),
            joint_names=("shoulder", "elbow"),
            nv=8,
            order=("a", "b", "c", "d", "e", "f", "shoulder", "elbow"),
        )


def _object_layout() -> DofLayout:
    base = ("crate.px", "crate.py", "crate.pz", "crate.rx", "crate.ry", "crate.rz")
    return DofLayout(
        articulation=None,
        base_dofs=base,
        base_names=("crate",),
        joint_names=(),
        nv=6,
        order=base,
        owner_kind="object",
        owner="crate",
    )


def test_object_dof_layout_records_an_object_owner() -> None:
    """An object layout is never labelled as an articulation."""
    layout = _object_layout()

    assert layout.owner_kind == "object"
    assert layout.owner == "crate"
    assert layout.articulation is None
    assert layout.order == layout.base_dofs


def test_object_dof_layout_rejects_a_non_null_articulation() -> None:
    """An object layout must not name an articulation."""
    with pytest.raises(ValueError, match="articulation must be None"):
        replace(_object_layout(), articulation="tiny")


def test_object_dof_layout_rejects_a_missing_owner() -> None:
    """An object layout must name its owning prop."""
    with pytest.raises(ValueError, match="owner must be non-empty"):
        replace(_object_layout(), owner=None)


def test_object_dof_layout_rejects_joints() -> None:
    """A rigid free prop has no internal joints."""
    layout = _object_layout()
    with pytest.raises(ValueError, match="no joint_names"):
        replace(layout, joint_names=("hinge",), nv=7, order=(*layout.base_dofs, "hinge"))


def test_object_dof_layout_requires_six_columns() -> None:
    """A rigid free prop is exactly a six-DOF object."""
    with pytest.raises(ValueError, match="nv=6"):
        replace(_object_layout(), nv=7, order=(*_object_layout().base_dofs, "extra"))


def test_object_dof_layout_requires_six_base_dofs() -> None:
    """The object layout's six columns are the free-joint DOFs."""
    with pytest.raises(ValueError, match="six base_dofs"):
        replace(_object_layout(), base_dofs=(), order=("a", "b", "c", "d", "e", "f"))


def test_object_dof_layout_requires_one_base_name() -> None:
    """A rigid free prop has one owning body name."""
    with pytest.raises(ValueError, match="one base_names"):
        replace(_object_layout(), base_names=())


def test_articulation_dof_layout_rejects_an_object_owner() -> None:
    """An articulation layout leaves the object owner unset."""
    with pytest.raises(ValueError, match="owner must be None"):
        replace(_layout(), owner="crate", owner_kind="articulation")


def test_entity_ref_rejects_an_unknown_kind() -> None:
    """Entity kinds are the closed body/site/geom vocabulary."""
    with pytest.raises(ValueError, match="EntityRef.kind"):
        EntityRef("joint", "shoulder")  # type: ignore[arg-type]


def test_point_ref_requires_a_body_entity() -> None:
    """A world point Jacobian is anchored to a body."""
    with pytest.raises(ValueError, match="body reference"):
        PointRef(EntityRef("site", "ee"), torch.zeros(3))


def test_com_ref_rejects_an_empty_articulation() -> None:
    """A COM reference always names its articulation."""
    with pytest.raises(ValueError, match="articulation"):
        ComRef("")


def test_jacobian_validates_shape_against_layout() -> None:
    """Translational and rotational blocks both follow ``(E, 3, nv)``."""
    layout = _layout()
    with pytest.raises(QueryShapeError, match="transl"):
        Jacobian(
            _ids(0),
            torch.zeros(1, 3, 7),
            torch.zeros(1, 3, 8),
            layout,
            "world",
            "body_origin",
        )


def test_jacobian_copies_its_tensors() -> None:
    """A Jacobian result is detached from the engine's work buffer."""
    layout = _layout()
    transl = torch.zeros(1, 3, 8, dtype=torch.float64)
    jacobian = Jacobian(
        _ids(0),
        transl,
        torch.zeros(1, 3, 8, dtype=torch.float64),
        layout,
        "world",
        "body_origin",
    )

    transl[0, 0, 0] = 5.0
    assert jacobian.transl[0, 0, 0].item() == 0.0
