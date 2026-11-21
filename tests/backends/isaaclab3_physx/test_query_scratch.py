"""Real IsaacLab 3 PhysX transaction-scratch behavior on the Cartpole fixture."""

from __future__ import annotations

import torch

import pytest

from unigym.core.quat import rotate
from unigym.query import EntityRef, PointRef, QueryCapabilityError
from unigym.query.errors import QueryStaleError

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
JOINT = "cart_to_pole"


def _body_names(cartpole_env) -> tuple[str, ...]:
    return tuple(cartpole_env.robots[ARTICULATION].body_names)


def _joint_names(cartpole_env) -> tuple[str, ...]:
    return tuple(cartpole_env.robots[ARTICULATION].joint_names)


def _write_delta(cartpole_env, delta: float = 0.2) -> torch.Tensor:
    """Return one joint target above its current position."""
    column = _joint_names(cartpole_env).index(JOINT)
    current = cartpole_env.query.state.joint_position(ARTICULATION, env=0)
    return torch.tensor([float(current[0, column]) + delta], dtype=torch.float64)


def _jacobi_index(cartpole_env, body: str) -> int:
    """Resolve one public body row onto the engine Jacobian body axis."""
    asset = cartpole_env.robots[ARTICULATION]
    names = _body_names(cartpole_env)
    index = names.index(body)
    if int(asset.num_base_dofs) == 6:
        return index
    root_index = names.index(tuple(asset.backend_body_names)[0])
    return index if index < root_index else index - 1


def _tip(pose) -> torch.Tensor:
    """Return a material point 0.2 m along the body's local x axis, in world."""
    local = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
    return pose.pos_w[0] + rotate(pose.quat_wxyz[0], local)


def test_scratch_joint_write_changes_state_and_restores_on_close(cartpole_env) -> None:
    """A transaction changes live pose/Jacobian and close restores the capture."""
    query = cartpole_env.query
    asset = cartpole_env.robots[ARTICULATION]
    body = _body_names(cartpole_env)[-1]
    column = _joint_names(cartpole_env).index(JOINT)
    before_pose = query.state.body_pose(body, articulation=ARTICULATION, env=0)
    before_joints = query.state.joint_position(ARTICULATION, env=0).clone()
    before_tip_jac = query.kinematics.jacobian(
        PointRef(EntityRef("body", body, articulation=ARTICULATION), _tip(before_pose)),
        env=0,
    )
    target = _write_delta(cartpole_env)

    with query.scratch(0) as scratch:
        scratch.set_joint_state(ARTICULATION, (JOINT,), target)
        scratch.forward()
        inside_joints = query.state.joint_position(ARTICULATION, env=0)
        inside_pose = scratch.body_pose(body, articulation=ARTICULATION)
        assert abs(float(inside_joints[0, column]) - float(target[0])) < 1e-5
        # The pole body origin sits on the hinge, so its position can stay put;
        # the observable transaction effect is the body orientation.
        assert not torch.allclose(inside_pose.quat_wxyz, before_pose.quat_wxyz, atol=1e-6, rtol=0)
        jac = scratch.jacobian(EntityRef("body", body, articulation=ARTICULATION))
        native = asset.data.body_link_jacobian_w.torch[0, _jacobi_index(cartpole_env, body)].to(
            device="cpu", dtype=torch.float64
        )
        torch.testing.assert_close(jac.transl[0], native[0:3])
        torch.testing.assert_close(jac.rot[0], native[3:6])
        inside_tip_jac = scratch.jacobian(
            PointRef(EntityRef("body", body, articulation=ARTICULATION), _tip(inside_pose))
        )
        assert not torch.allclose(inside_tip_jac.transl, before_tip_jac.transl, atol=1e-7, rtol=0)
        base_dofs = int(asset.num_base_dofs)
        assert tuple(jac.transl.shape) == (1, 3, before_joints.shape[1] + base_dofs)

    after_pose = query.state.body_pose(body, articulation=ARTICULATION, env=0)
    after_joints = query.state.joint_position(ARTICULATION, env=0)
    torch.testing.assert_close(after_joints, before_joints, atol=1e-6, rtol=0)
    torch.testing.assert_close(after_pose.pos_w, before_pose.pos_w, atol=1e-6, rtol=0)
    torch.testing.assert_close(after_pose.quat_wxyz, before_pose.quat_wxyz, atol=1e-6, rtol=0)


def test_scratch_rejects_a_fixed_root_and_scratch_geometry(cartpole_env) -> None:
    """The fixed Cartpole root is read-only and scratch geometry stays unsupported."""
    query = cartpole_env.query
    body = _body_names(cartpole_env)[-1]

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError, match="fixed root"):
            scratch.set_root_state(
                ARTICULATION,
                pos_w=torch.zeros(3, dtype=torch.float64),
                quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            )
        with pytest.raises(QueryCapabilityError, match="scratch.geometry"):
            scratch.geometry().distance(
                EntityRef("body", body, articulation=ARTICULATION),
                EntityRef("body", body, articulation=ARTICULATION),
            )


def test_scratch_is_exclusive_per_env_and_stale_after_close(cartpole_env) -> None:
    """One env holds one transaction and a closed scratch rejects further reads."""
    query = cartpole_env.query
    body = _body_names(cartpole_env)[-1]

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError, match="already has an active scratch"):
            with query.scratch(0):
                pass

    with pytest.raises(QueryStaleError, match="scratch world is closed"):
        scratch.body_pose(body)


def test_scratch_restores_after_a_body_exception(cartpole_env) -> None:
    """A body exception still restores the captured joint state."""
    query = cartpole_env.query
    before_joints = query.state.joint_position(ARTICULATION, env=0).clone()
    target = _write_delta(cartpole_env)

    with pytest.raises(RuntimeError, match="boom"):
        with query.scratch(0) as scratch:
            scratch.set_joint_state(ARTICULATION, (JOINT,), target)
            scratch.forward()
            raise RuntimeError("boom")

    after_joints = query.state.joint_position(ARTICULATION, env=0)
    torch.testing.assert_close(after_joints, before_joints, atol=1e-6, rtol=0)
