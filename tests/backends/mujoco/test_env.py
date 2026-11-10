"""MuJoCo backend-local construction, IR, limits, permutation and teardown."""

from __future__ import annotations

import pytest
import torch

from unigym.env.backends.base import assert_backend_contract
from unigym.runtime.actions import NamedAction


def _zero_action(env):
    control = env.controls["tiny"]
    return NamedAction.zeros(env.num_envs, list(control.joint_names), control.target)


def test_env_constructs_from_tiny_mjcf_and_satisfies_backend_contract(tiny_env):
    """The real MuJoCo env loads the tiny MJCF and passes the live contract."""
    assert_backend_contract(tiny_env, exercise_step=True)
    assert tiny_env.num_envs == 1
    assert str(tiny_env.device) == "cpu"


def test_model_ir_publishes_expected_body_joint_and_actuator_names(tiny_env):
    """The compiled IR exposes the tiny model's declared topology and names."""
    ir = tiny_env.model_set.articulations["tiny"]

    assert [body.name for body in ir.bodies] == ["world", "base", "link", "tip"]
    assert [joint.name for joint in ir.joints] == ["root", "shoulder", "elbow"]
    assert [actuator.name for actuator in ir.actuators] == [
        "shoulder_motor",
        "elbow_motor",
    ]
    assert [actuator.target for actuator in ir.actuators] == ["shoulder", "elbow"]

    bodies = {body.name: body for body in ir.bodies}
    assert bodies["link"].parent == "base"
    assert bodies["tip"].parent == "link"
    assert bodies["base"].parent == "world"


def test_env_loads_only_the_tmp_mjcf_without_external_assets(tiny_mjcf, tiny_env_factory):
    """Construction consumes the tmp asset only: no mesh, texture or hfield."""
    assert [path.name for path in tiny_mjcf.parent.iterdir()] == ["tiny_arm.xml"]

    env = tiny_env_factory()
    model = env.mj_model
    assert model.nmesh == 0
    assert model.ntex == 0
    assert model.nhfield == 0

    ir = env.model_set.articulations["tiny"]
    assert list(ir.materials) == []
    if ir.mj is not None:
        assert list(ir.mj.textures) == []


def test_policy_joint_pos_limits_follow_the_control_order(tiny_env):
    """Limits are (N, J) float32 bounds in the control's joint order."""
    lower, upper = tiny_env.policy_joint_pos_limits
    assert tiny_env.controls["tiny"].joint_names == ("elbow", "shoulder")

    for bound in (lower, upper):
        assert tuple(bound.shape) == (tiny_env.num_envs, 2)
        assert bound.dtype == torch.float32
        assert str(bound.device) == str(tiny_env.device)

    torch.testing.assert_close(lower, torch.tensor([[-0.5, -1.57]]))
    torch.testing.assert_close(upper, torch.tensor([[1.0, 1.57]]))
    assert bool((lower <= upper).all())


def test_joint_permutation_maps_il_and_policy_order_both_ways(tiny_env):
    """The IL<->policy permutation is a real reorder and round-trips exactly."""
    assert tiny_env.perm_il2pol.tolist() == [1, 0]
    assert tiny_env.perm_pol2il.tolist() == [1, 0]

    policy_values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    il_values = tiny_env.to_il_order(policy_values)
    assert il_values.tolist() == [[20.0, 10.0], [40.0, 30.0]]
    assert torch.equal(tiny_env.to_policy_order(il_values), policy_values)

    assert tiny_env.to_policy_order(torch.tensor([[1.0, 2.0]])).tolist() == [[2.0, 1.0]]


def test_close_is_idempotent_and_post_close_step_fails(tiny_env):
    """Closing twice is a no-op; stepping a closed env never succeeds silently."""
    tiny_env.close()
    tiny_env.close()
    with pytest.raises((RuntimeError, TypeError)):
        tiny_env.step({"tiny": _zero_action(tiny_env)})
