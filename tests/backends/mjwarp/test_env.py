"""MuJoCo Warp backend-local construction, IR, limits, permutation and teardown."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.closing import FailingClose, RecordingClose
from unigym.env.backends.base import assert_backend_contract
from unigym.runtime.actions import NamedAction

pytestmark = pytest.mark.sim


def _zero_action(env):
    control = env.controls["tiny"]
    return NamedAction.zeros(env.num_envs, list(control.joint_names), control.target)


def test_env_constructs_from_tiny_mjcf_and_satisfies_backend_contract(tiny_env):
    """The real warp env loads the tiny MJCF and passes the live contract."""
    assert_backend_contract(tiny_env, exercise_step=True)
    assert tiny_env.num_envs == 1
    assert tiny_env.device == "cpu"
    assert tiny_env.warp_device == "cpu"


def test_env_constructs_a_native_batch_of_two_worlds(tiny_env_factory):
    """N=2 is one native warp batch that still satisfies the backend contract."""
    env = tiny_env_factory(num_envs=2)

    assert_backend_contract(env, exercise_step=True)

    assert env.num_envs == 2
    lower, upper = env.policy_joint_pos_limits
    assert tuple(lower.shape) == (2, len(env.controls["tiny"].joint_names))
    assert tuple(upper.shape) == (2, len(env.controls["tiny"].joint_names))


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


def test_close_is_idempotent_and_releases_the_sim_handles(tiny_env):
    """A second close is a safe no-op after the engine handles are released."""
    tiny_env.close()
    tiny_env.close()

    assert tiny_env.mj_model is None
    assert tiny_env.mj_data is None
    assert tiny_env.mj_datas is None


def test_step_after_close_raises_the_closed_error(tiny_env):
    """Stepping a closed env reports the closed backend, not a handle crash."""
    tiny_env.close()

    with pytest.raises(RuntimeError) as excinfo:
        tiny_env.step({"tiny": _zero_action(tiny_env)})

    assert str(excinfo.value) == "mjwarp backend: environment is closed"


def test_reset_envs_after_close_raises_the_closed_error(tiny_env):
    """A subset reset on a closed env reports the closed backend, not a handle crash."""
    tiny_env.close()

    with pytest.raises(RuntimeError) as excinfo:
        tiny_env.reset_envs(torch.tensor([0], dtype=torch.int64))

    assert str(excinfo.value) == "mjwarp backend: environment is closed"


def test_reset_all_after_close_raises_the_closed_error(tiny_env):
    """A full reset on a closed env reports the closed backend, not a handle crash."""
    tiny_env.close()

    with pytest.raises(RuntimeError) as excinfo:
        tiny_env.reset_all()

    assert str(excinfo.value) == "mjwarp backend: environment is closed"


def test_close_failure_leaves_backend_guarded_and_retryable(tiny_env, monkeypatch):
    """A manager teardown failure blocks use but keeps the backend retryable."""
    camera = RecordingClose()
    failing_prop = FailingClose()
    monkeypatch.setattr(tiny_env, "camera_manager", camera)
    monkeypatch.setattr(tiny_env, "prop_backend", failing_prop)

    with pytest.raises(RuntimeError, match="prop backend teardown failed"):
        tiny_env.close()

    assert camera.close_calls == 1
    assert tiny_env.mj_datas is not None
    assert tiny_env.mj_data is not None
    assert tiny_env.mj_model is not None

    action = {"tiny": _zero_action(tiny_env)}

    def _step():
        return tiny_env.step(action)

    def _reset_subset():
        return tiny_env.reset_envs(torch.tensor([0], dtype=torch.int64))

    for call in (_step, _reset_subset, tiny_env.reset_all):
        with pytest.raises(RuntimeError) as excinfo:
            call()
        assert str(excinfo.value) == "mjwarp backend: environment is closed"

    monkeypatch.setattr(failing_prop, "close", lambda: None)
    tiny_env.close()

    assert camera.close_calls == 2
    assert tiny_env.mj_datas is None
    assert tiny_env.mj_data is None
    assert tiny_env.mj_model is None
