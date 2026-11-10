"""MuJoCo backend-local named-action stepping, fail-fast routing and resets."""

from __future__ import annotations

import pytest
import torch

from unigym.env.backends.base import check_extra_info, check_obs_cache
from unigym.runtime.actions import NamedAction


def _action(env, values, joint_names=None):
    control = env.controls["tiny"]
    names = list(control.joint_names if joint_names is None else joint_names)
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(env.num_envs, len(names))
    return {"tiny": NamedAction(tensor, names, control.target)}


def test_named_action_step_advances_joints_and_returns_contract_shaped_output(tiny_env):
    """One named-position tick moves the joints and honors the obs/extra contract."""
    control = tiny_env.controls["tiny"]
    action = _action(tiny_env, [[0.3, -0.2]])

    obs, extra = tiny_env.step(action)

    check_obs_cache(obs, tiny_env.num_envs, device=tiny_env.device)
    check_obs_cache(tiny_env.obs_cache, tiny_env.num_envs, device=tiny_env.device)
    check_extra_info(
        extra,
        tiny_env.num_envs,
        len(control.joint_names),
        len(control.feet),
        device=tiny_env.device,
    )

    joint_pos = extra["articulation/tiny/joint_pos"]
    assert 0.0 < joint_pos[0, 0] <= 0.3
    assert -0.2 <= joint_pos[0, 1] < 0.0
    assert extra["should_reset"] is False
    assert tuple(extra["reset_env_ids"].shape) == (0,)


def test_step_rejects_unknown_face_with_nonmatching_joints(tiny_env):
    """An unknown face whose joints match no control is rejected (no fallback)."""
    entry = NamedAction(torch.zeros(tiny_env.num_envs, 2), ["elbow", "wrist"], "position")
    with pytest.raises(ValueError, match="not a declared control face"):
        tiny_env.step({"wrist": entry})


def test_step_rejects_a_missing_control_joint(tiny_env):
    """A required control whose joint is left unaddressed fails fast."""
    action = {"tiny": NamedAction(torch.zeros(tiny_env.num_envs, 1), ["elbow"], "position")}
    with pytest.raises(ValueError, match=r"leaves joints unaddressed: \['shoulder'\]"):
        tiny_env.step(action)


def test_step_rejects_a_joint_name_the_control_does_not_own(tiny_env):
    """An action column naming a non-control joint fails fast."""
    action = {
        "tiny": NamedAction(torch.zeros(tiny_env.num_envs, 2), ["elbow", "wrist"], "position")
    }
    with pytest.raises(ValueError, match=r"names joints \['wrist'\]"):
        tiny_env.step(action)


def test_step_rejects_a_non_mapping_action_payload(tiny_env):
    """The step entry point requires dict[str, NamedAction], not a sequence."""
    entry = _action(tiny_env, [[0.0, 0.0]])["tiny"]
    with pytest.raises(TypeError, match=r"dict\[str, NamedAction\]"):
        tiny_env.step([("tiny", entry)])


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
