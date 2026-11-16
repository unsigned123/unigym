"""IsaacLab 2 state-write contract on the Cartpole lane."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"


def _joint_index(env, name: str) -> int:
    return list(env.articulations[ARTICULATION].asset.joint_names).index(name)


def test_set_joint_state_writes_a_named_subset(cartpole_env):
    """Joint columns resolve by name; a later settle keeps the written state."""
    index = _joint_index(cartpole_env, "cart_to_pole")

    returned = cartpole_env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("cart_to_pole",),
        position=torch.tensor([[0.3]], device=cartpole_env.device),
        velocity=torch.tensor([[0.2]], device=cartpole_env.device),
    )
    cartpole_env.settle(None)

    assert returned.dtype == torch.int64
    assert returned.tolist() == [0]
    data = cartpole_env.articulations[ARTICULATION].asset.data
    assert float(data.joint_pos[0, index]) == pytest.approx(0.3, abs=1e-5)
    assert float(data.joint_vel[0, index]) == pytest.approx(0.2, abs=1e-5)


def test_set_joint_state_without_velocity_leaves_velocity_unchanged(cartpole_env):
    """velocity=None writes positions only."""
    index = _joint_index(cartpole_env, "cart_to_pole")
    cartpole_env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("cart_to_pole",),
        position=torch.tensor([[0.1]], device=cartpole_env.device),
        velocity=torch.tensor([[0.7]], device=cartpole_env.device),
    )
    before = float(cartpole_env.articulations[ARTICULATION].asset.data.joint_vel[0, index])

    cartpole_env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("cart_to_pole",),
        position=torch.tensor([[0.4]], device=cartpole_env.device),
    )

    data = cartpole_env.articulations[ARTICULATION].asset.data
    assert float(data.joint_pos[0, index]) == pytest.approx(0.4, abs=1e-5)
    assert float(data.joint_vel[0, index]) == pytest.approx(before, abs=1e-6)


def test_set_root_state_rejects_the_fixed_cartpole_root(cartpole_env):
    """Cartpole is a fixed-base articulation: root writes are an honest error."""
    assert bool(cartpole_env.articulations[ARTICULATION].asset.is_fixed_base)

    with pytest.raises(ValueError, match="fixed root"):
        cartpole_env.set_root_state(
            None,
            name=ARTICULATION,
            pos_w=torch.zeros(1, 3, device=cartpole_env.device),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=cartpole_env.device),
            lin_vel_w=torch.zeros(1, 3, device=cartpole_env.device),
            ang_vel_w=torch.zeros(1, 3, device=cartpole_env.device),
        )


def test_settle_returns_the_selected_ids_and_keeps_the_clock(cartpole_env):
    """settle flushes/forwards the view without advancing the episode clock."""
    before = cartpole_env._time_in_episode.clone()

    returned = cartpole_env.settle([0])

    assert returned.dtype == torch.int64
    assert returned.tolist() == [0]
    assert torch.equal(cartpole_env._time_in_episode, before)


def test_set_joint_state_rejects_unknown_names_and_bad_shapes(cartpole_env):
    """Unknown joint names and wrong shapes fail at the boundary."""
    with pytest.raises(ValueError, match=r"not asset joints.*\['wrist'\]"):
        cartpole_env.set_joint_state(
            None,
            articulation=ARTICULATION,
            joint_names=("wrist",),
            position=torch.zeros(1, 1, device=cartpole_env.device),
        )
    with pytest.raises(ValueError, match=r"position shape"):
        cartpole_env.set_joint_state(
            None,
            articulation=ARTICULATION,
            joint_names=("cart_to_pole",),
            position=torch.zeros(1, 2, device=cartpole_env.device),
        )


def test_set_root_state_rejects_a_zero_norm_quat(cartpole_env):
    """Normalization is enforced before the root is resolved."""
    with pytest.raises(ValueError, match="zero-norm"):
        cartpole_env.set_root_state(
            None,
            name=ARTICULATION,
            pos_w=torch.zeros(1, 3, device=cartpole_env.device),
            quat_wxyz=torch.zeros(1, 4, device=cartpole_env.device),
            lin_vel_w=torch.zeros(1, 3, device=cartpole_env.device),
            ang_vel_w=torch.zeros(1, 3, device=cartpole_env.device),
        )
