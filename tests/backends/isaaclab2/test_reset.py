"""IsaacLab 2 Cartpole per-env, empty and whole-batch episode reset."""

from __future__ import annotations

import pytest

from unigym.runtime.actions import NamedAction

pytestmark = pytest.mark.sim

ARTICULATION = "cartpole"
CART_JOINTS = ("slider_to_cart", "cart_to_pole")


def _zero_action(env):
    return {ARTICULATION: NamedAction.zeros(env.num_envs, list(CART_JOINTS), "position")}


def test_reset_envs_resets_only_the_requested_row(two_env):
    """A single-env reset returns that id and restarts only its episode clock."""
    _, first = two_env.step(_zero_action(two_env))
    first_time = first["time_in_episode"].tolist()
    first_episode = first["episode_id"].tolist()

    reset_ids = two_env.reset_envs([0])

    assert reset_ids.tolist() == [0]

    _, extra = two_env.step(_zero_action(two_env))
    assert extra["time_in_episode"].tolist() == pytest.approx(
        [two_env.step_dt, first_time[1] + two_env.step_dt]
    )
    assert extra["episode_id"].tolist() == [first_episode[0] + 1, first_episode[1]]


def test_reset_envs_with_no_ids_is_a_noop(two_env):
    """An empty reset list returns no ids and leaves both episode clocks advancing."""
    _, first = two_env.step(_zero_action(two_env))
    first_time = first["time_in_episode"].tolist()
    first_episode = first["episode_id"].tolist()

    reset_ids = two_env.reset_envs([])

    assert reset_ids.tolist() == []

    _, extra = two_env.step(_zero_action(two_env))
    assert extra["time_in_episode"].tolist() == pytest.approx(
        [first_time[0] + two_env.step_dt, first_time[1] + two_env.step_dt]
    )
    assert extra["episode_id"].tolist() == first_episode


def test_reset_all_restarts_every_row(two_env):
    """A whole-batch reset returns every id and increments every episode id."""
    _, first = two_env.step(_zero_action(two_env))
    first_episode = first["episode_id"].tolist()

    reset_ids = two_env.reset_all()

    assert reset_ids.tolist() == [0, 1]

    _, extra = two_env.step(_zero_action(two_env))
    assert extra["time_in_episode"].tolist() == pytest.approx([two_env.step_dt, two_env.step_dt])
    assert extra["episode_id"].tolist() == [episode + 1 for episode in first_episode]
