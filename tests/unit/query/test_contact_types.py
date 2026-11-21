"""Validation and copy semantics of the neutral contact query types."""

from __future__ import annotations

import pytest
import torch

from unigym.query.base import ContactJacobian, ContactManifold, ContactPoint, DofLayout
from unigym.query.errors import QueryShapeError


def _point(**overrides) -> ContactPoint:
    values = {
        "env_id": 0,
        "group": "crate_floor",
        "body1": "crate",
        "body2": "world",
        "geom1": "crate_geom",
        "geom2": "floor",
        "pos_w": torch.zeros(3, dtype=torch.float64),
        "normal_w": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        "frame_w": torch.eye(3, dtype=torch.float64),
        "separation_m": -0.001,
        "normal_force_n": 4.9,
        "friction_force_w": torch.zeros(3, dtype=torch.float64),
        "contact_id": 0,
    }
    values.update(overrides)
    return ContactPoint(**values)


def test_contact_point_copies_its_tensors() -> None:
    """A caller cannot mutate a result through the input tensors."""
    position = torch.zeros(3, dtype=torch.float64)
    point = _point(pos_w=position)

    position[0] = 9.0
    assert float(point.pos_w[0]) == 0.0


def test_contact_point_accepts_unnamed_geoms() -> None:
    """A compiled geom without a name is None, not an empty string."""
    point = _point(geom1=None, geom2=None)

    assert point.geom1 is None
    assert point.geom2 is None


def test_contact_point_rejects_a_non_finite_force() -> None:
    """A fabricated or divergent force is never silently accepted."""
    with pytest.raises(QueryShapeError, match="friction_force_w"):
        _point(friction_force_w=torch.tensor([float("nan"), 0.0, 0.0]))


def test_contact_point_requires_a_contact_frame_shape() -> None:
    """The contact frame is the 3x3 n/t1/t2 world axis matrix."""
    with pytest.raises(QueryShapeError, match=r"frame_w.*shape \(3, 3\)"):
        _point(frame_w=torch.eye(2, dtype=torch.float64))


def test_contact_point_rejects_an_empty_group_or_body() -> None:
    """Group and body identities are non-empty names."""
    with pytest.raises(ValueError, match="group"):
        _point(group="")
    with pytest.raises(ValueError, match="body2"):
        _point(body2="")


def test_contact_manifold_accepts_an_empty_point_tuple() -> None:
    """No active contact is an empty manifold, not an error."""
    manifold = ContactManifold(torch.tensor([0], dtype=torch.int64), "crate_floor", ())

    assert manifold.points == ()
    assert manifold.env_ids.tolist() == [0]


def test_contact_manifold_copies_its_env_ids() -> None:
    """The manifold owns its env selection tensor."""
    ids = torch.tensor([0, 1], dtype=torch.int64)
    manifold = ContactManifold(ids, "crate_floor", ())

    ids[0] = 5
    assert manifold.env_ids.tolist() == [0, 1]


def test_contact_manifold_rejects_a_point_from_another_group() -> None:
    """Every point belongs to the manifold's group."""
    with pytest.raises(ValueError, match="group"):
        ContactManifold(
            torch.tensor([0], dtype=torch.int64),
            "crate_floor",
            (_point(group="other"),),
        )


def test_contact_manifold_rejects_a_point_outside_the_env_selection() -> None:
    """A point cannot name an env the manifold did not select."""
    with pytest.raises(ValueError, match="env_id"):
        ContactManifold(
            torch.tensor([1], dtype=torch.int64),
            "crate_floor",
            (_point(env_id=0),),
        )


def _layout() -> DofLayout:
    return DofLayout(
        articulation="tiny",
        base_dofs=(),
        base_names=(),
        joint_names=("elbow", "shoulder"),
        nv=2,
        order=("elbow", "shoulder"),
    )


def test_contact_jacobian_copies_its_rows() -> None:
    """A caller cannot mutate a result through the input row block."""
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    jacobian = ContactJacobian(env_id=0, frame="contact", rows=rows, layout=_layout())

    rows[0, 0] = 99.0

    assert float(jacobian.rows[0, 0]) == 1.0
    assert jacobian.env_id == 0
    assert jacobian.frame == "contact"
    assert tuple(jacobian.rows.shape) == (3, 2)


def test_contact_jacobian_requires_the_layout_column_count() -> None:
    """The row block has one column per articulation DOF and any row count."""
    with pytest.raises(QueryShapeError, match=r"rows.*dim, 2"):
        ContactJacobian(
            env_id=0,
            frame="contact",
            rows=torch.zeros((3, 3), dtype=torch.float64),
            layout=_layout(),
        )


def test_contact_jacobian_rejects_an_unknown_frame_or_non_finite_rows() -> None:
    """Only the contact frame and finite constraint rows are accepted."""
    with pytest.raises(QueryShapeError, match="frame"):
        ContactJacobian(
            env_id=0,
            frame="world",
            rows=torch.zeros((1, 2), dtype=torch.float64),
            layout=_layout(),
        )
    with pytest.raises(QueryShapeError, match="rows"):
        ContactJacobian(
            env_id=0,
            frame="contact",
            rows=torch.tensor([[float("nan"), 0.0]], dtype=torch.float64),
            layout=_layout(),
        )


def test_contact_jacobian_requires_float64_rows() -> None:
    """The constraint-row contract is a float64 tensor, not any float dtype."""
    with pytest.raises(QueryShapeError, match="float64"):
        ContactJacobian(
            env_id=0,
            frame="contact",
            rows=torch.zeros((1, 2), dtype=torch.float32),
            layout=_layout(),
        )
