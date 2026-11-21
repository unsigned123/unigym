"""Real IsaacLab 3 PhysX Jacobian behavior on the Cartpole lane fixture."""

from __future__ import annotations

import torch

import pytest

from unigym.query import ComRef, EntityRef, PointRef, QueryCapabilityError

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"


def _target(cartpole_env):
    """Return the live asset, a movable body index, its Jacobian row and names."""
    asset = cartpole_env.robots[ARTICULATION]
    names = tuple(asset.body_names)
    joints = tuple(asset.joint_names)
    num_base = int(asset.num_base_dofs)
    if num_base == 6:
        body_index = len(names) - 1
        return asset, body_index, body_index, joints, names
    root_index = names.index(tuple(asset.backend_body_names)[0])
    body_index = next(index for index in range(len(names)) if index != root_index)
    jacobi_index = body_index if body_index < root_index else body_index - 1
    return asset, body_index, jacobi_index, joints, names


def test_body_jacobian_matches_the_native_link_row(cartpole_env) -> None:
    """A body Jacobian row is the native link row in the base-first layout."""
    query = cartpole_env.query
    asset, body_index, jacobi_index, joints, names = _target(cartpole_env)
    num_base = int(asset.num_base_dofs)

    jac = query.kinematics.jacobian(
        EntityRef("body", names[body_index], articulation=ARTICULATION), env=0
    )
    native = asset.data.body_link_jacobian_w.torch[0, jacobi_index].to(
        device="cpu", dtype=torch.float64
    )

    assert tuple(jac.transl.shape) == (1, 3, len(joints) + num_base)
    assert jac.layout.joint_names == joints
    assert jac.layout.order == jac.layout.base_dofs + jac.layout.joint_names
    assert len(jac.layout.base_dofs) == num_base
    if num_base == 6:
        base_body_id = cartpole_env.articulations[ARTICULATION].base_body_id
        assert jac.layout.base_names == (str(base_body_id),)
    else:
        assert jac.layout.base_dofs == ()
    torch.testing.assert_close(jac.transl[0], native[0:3])
    torch.testing.assert_close(jac.rot[0], native[3:6])


def test_com_jacobian_matches_the_native_com_row(cartpole_env) -> None:
    """A body COM row is native; a subtree COM row has no engine source."""
    query = cartpole_env.query
    asset, body_index, jacobi_index, _, names = _target(cartpole_env)

    jac = query.kinematics.jacobian(ComRef(ARTICULATION, body=names[body_index]), env=0)
    native = asset.data.body_com_jacobian_w.torch[0, jacobi_index].to(
        device="cpu", dtype=torch.float64
    )

    torch.testing.assert_close(jac.transl[0], native[0:3])
    torch.testing.assert_close(jac.rot[0], native[3:6])

    with pytest.raises(QueryCapabilityError, match="subtree"):
        query.kinematics.jacobian(ComRef(ARTICULATION, body=names[body_index], subtree=True))


def test_world_point_jacobian_composes_the_body_link_row(cartpole_env) -> None:
    """A world-point row is the body link row shifted by ``omega x r``."""
    query = cartpole_env.query
    asset, body_index, _, _, names = _target(cartpole_env)
    body_name = names[body_index]

    pose = query.state.body_pose(body_name, articulation=ARTICULATION, env=0)
    point = pose.pos_w[0] + torch.tensor([0.1, 0.0, 0.2], dtype=torch.float64)
    reference = query.kinematics.jacobian(
        EntityRef("body", body_name, articulation=ARTICULATION), env=0
    )
    jac = query.kinematics.jacobian(
        PointRef(EntityRef("body", body_name, articulation=ARTICULATION), point), env=0
    )

    offset = point - pose.pos_w[0]
    columns = int(reference.transl.shape[-1])
    expected = reference.transl[0] + torch.cross(
        reference.rot[0], offset[:, None].expand(-1, columns), dim=0
    )
    torch.testing.assert_close(jac.transl[0], expected)
    torch.testing.assert_close(jac.rot[0], reference.rot[0])


def test_cartpole_ir_has_typed_joints_and_a_complete_parent_chain(cartpole_env) -> None:
    """The live root prim resolves the Cartpole joint types and body parents."""
    ir = cartpole_env.query.model.ir(ARTICULATION)

    types = {name: ir.joints[name].type for name in ir.joints.names()}
    assert types == {"slider_to_cart": "slide", "cart_to_pole": "hinge"}
    assert "unknown" not in types.values()

    bodies = {name: ir.bodies[name] for name in ir.bodies.names()}
    assert bodies["slider"].parent is None
    assert bodies["cart"].parent == "slider"
    assert bodies["pole"].parent == "cart"
