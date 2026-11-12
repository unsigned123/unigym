"""MuJoCo Warp backend-local named-action stepping, failure paths and resets."""

from __future__ import annotations

import pytest
import torch

from unigym.env.backends.base import check_extra_info, check_obs_cache
from unigym.runtime.actions import NamedAction

pytestmark = pytest.mark.sim


def _action(env, values):
    control = env.controls["tiny"]
    names = list(control.joint_names)
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(env.num_envs, len(names))
    return {"tiny": NamedAction(tensor, names, control.target)}


def test_named_action_step_advances_joints_and_returns_contract_shaped_output(tiny_env):
    """One named-position tick moves the joints and honors the obs/extra contract."""
    control = tiny_env.controls["tiny"]

    obs, extra = tiny_env.step(_action(tiny_env, [[0.3, -0.2]]))

    check_obs_cache(obs, tiny_env.num_envs, device=tiny_env.device)
    check_obs_cache(tiny_env.obs_cache, tiny_env.num_envs, device=tiny_env.device)
    check_extra_info(
        extra,
        tiny_env.num_envs,
        len(control.joint_names),
        device=tiny_env.device,
    )

    joint_pos = extra["articulation/tiny/joint_pos"]
    assert 0.0 < joint_pos[0, 0] <= 0.3
    assert -0.2 <= joint_pos[0, 1] < 0.0
    assert extra["should_reset"] is False
    assert tuple(extra["reset_env_ids"].shape) == (0,)
    assert extra["episode_id"].tolist() == [1]


def test_named_action_step_keeps_batch_rows_independent(tiny_env_factory):
    """Different per-row targets drive different per-row joints in one batch."""
    env = tiny_env_factory(num_envs=2)

    _, extra = env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    joint_pos = extra["articulation/tiny/joint_pos"]
    assert joint_pos[0, 0] > joint_pos[1, 0] > 0.0
    assert joint_pos[0, 1] < 0.0 < joint_pos[1, 1]
    assert extra["episode_id"].tolist() == [1, 1]


def test_reset_envs_resets_only_the_selected_row_and_returns_its_id(tiny_env_factory):
    """A subset reset restores that row only and returns the requested ids."""
    env = tiny_env_factory(num_envs=2)
    env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    returned = env.reset_envs(torch.tensor([1], dtype=torch.int64))

    assert returned.tolist() == [1]
    assert returned.dtype == torch.int64

    _, extra = env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    joint_pos = extra["articulation/tiny/joint_pos"]
    assert float(joint_pos[1].abs().max()) < 0.01
    assert float(joint_pos[0].abs().max()) > 0.01
    assert extra["episode_id"].tolist() == [1, 2]
    assert extra["time_in_episode"].tolist() == pytest.approx([0.04, 0.02])


def test_reset_all_resets_every_row_and_returns_the_full_id_range(tiny_env_factory):
    """reset_all restores every row and returns arange(num_envs)."""
    env = tiny_env_factory(num_envs=2)
    env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    returned = env.reset_all()

    assert returned.tolist() == [0, 1]
    assert returned.dtype == torch.int64

    _, extra = env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    assert float(extra["articulation/tiny/joint_pos"].abs().max()) < 0.01
    assert extra["episode_id"].tolist() == [2, 2]
    assert extra["time_in_episode"].tolist() == pytest.approx([0.02, 0.02])


def test_reset_envs_empty_is_a_noop_and_preserves_requested_id_order(tiny_env_factory):
    """An empty reset changes nothing; a subset reset returns ids in input order."""
    env = tiny_env_factory(num_envs=2)
    env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    empty = env.reset_envs(torch.empty(0, dtype=torch.int64))
    assert empty.numel() == 0
    assert empty.dtype == torch.int64

    _, extra = env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    assert extra["episode_id"].tolist() == [1, 1]
    assert extra["time_in_episode"].tolist() == pytest.approx([0.04, 0.04])

    returned = env.reset_envs(torch.tensor([1, 0], dtype=torch.int64))
    assert returned.tolist() == [1, 0]
    assert returned.dtype == torch.int64


@pytest.mark.parametrize(
    "joint_names",
    [
        pytest.param(["elbow", "wrist"], id="nonmatching-joints"),
        pytest.param(["elbow", "shoulder"], id="matching-joints"),
    ],
)
def test_step_rejects_an_unknown_action_face(tiny_env, joint_names):
    """An unknown face is rejected even when its joints match a control."""
    entry = NamedAction(torch.zeros(tiny_env.num_envs, 2), joint_names, "position")

    with pytest.raises(ValueError) as excinfo:
        tiny_env.step({"wrist": entry})

    message = str(excinfo.value)
    assert "action entry 'wrist' is not a declared control face" in message
    assert "accepted faces: ['tiny']" in message


@pytest.mark.parametrize("kind", ["torque", "velocity", "effort", "current"])
def test_step_rejects_an_action_kind_that_differs_from_the_control_target(tiny_env, kind):
    """A mismatched NamedAction.kind is rejected against the control's target."""
    control = tiny_env.controls["tiny"]
    entry = NamedAction.zeros(tiny_env.num_envs, list(control.joint_names), kind)

    with pytest.raises(ValueError) as excinfo:
        tiny_env.step({"tiny": entry})

    message = str(excinfo.value)
    assert "action entry 'tiny'" in message
    assert f"kind '{kind}'" in message
    assert "control 'tiny' expects target 'position'" in message


@pytest.mark.parametrize(
    ("joint_names", "fragment"),
    [
        pytest.param(["elbow"], "leaves joints unaddressed: ['shoulder']", id="missing-joint"),
        pytest.param(["elbow", "wrist"], "names joints ['wrist']", id="unowned-joint"),
    ],
)
def test_step_rejects_an_action_that_does_not_match_the_control_joint_set(
    tiny_env, joint_names, fragment
):
    """An action's joint columns must address exactly the control's joints."""
    entry = NamedAction(torch.zeros(tiny_env.num_envs, len(joint_names)), joint_names, "position")

    with pytest.raises(ValueError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
        tiny_env.step({"tiny": entry})


def test_step_rejects_a_non_mapping_action_payload(tiny_env):
    """The step entry point requires dict[str, NamedAction], not a sequence."""
    entry = _action(tiny_env, [[0.0, 0.0]])["tiny"]

    with pytest.raises(TypeError, match=r"dict\[str, NamedAction\]"):
        tiny_env.step([("tiny", entry)])
