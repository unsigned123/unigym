"""Canonical timeline mapping and integer deadline authority."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.benchmarks.base import EpisodeData
from unigym.benchmarks.episode import (
    EpisodeLimits,
    step_index_for_frame,
    time_for_frame,
    timeline_step_index,
    timeline_time_s,
)


def test_frame_zero_is_tick_one():
    assert step_index_for_frame(0) == 1
    assert time_for_frame(0, 0.05) == pytest.approx(0.05)


def test_frame_380_is_tick_381_time_19_05():
    assert step_index_for_frame(380) == 381
    assert time_for_frame(380, 0.05) == pytest.approx(19.05)
    assert timeline_time_s(381, 0.05)[-1] == pytest.approx(19.05)
    assert timeline_step_index(381)[-1] == 381


def test_episode_data_derives_canonical_timeline():
    ep = EpisodeData(
        data={},
        env_id=0,
        episode_id=0,
        length=3,
        duration_s=0.0,
        ended_by_termination=False,
        policy_dt=0.02,
        sim_dt=0.005,
    )
    assert ep.step_index.tolist() == [1, 2, 3]
    assert ep.time_s.tolist() == pytest.approx([0.02, 0.04, 0.06])
    assert ep.reset_index == 0


def test_episode_data_rejects_noncanonical_mapping():
    with pytest.raises(ValueError, match="canonical"):
        EpisodeData(
            data={},
            env_id=0,
            episode_id=0,
            length=2,
            duration_s=0.0,
            ended_by_termination=False,
            policy_dt=0.02,
            step_index=np.array([0, 1], dtype=np.int64),
            time_s=np.array([0.0, 0.02]),
        )


def test_deadline_uses_floor_semantics():
    assert EpisodeLimits(timeout_s=10.0).validate_against(0.02) == 500
    assert EpisodeLimits(timeout_s=9.99).validate_against(0.02) == 499
    assert EpisodeLimits(timeout_steps=42).validate_against(0.02) == 42
    assert EpisodeLimits().validate_against(0.02) is None


def test_conflicting_deadlines_fail():
    with pytest.raises(ValueError, match="conflicting"):
        EpisodeLimits(timeout_s=10.0, timeout_steps=400).validate_against(0.02)
    assert EpisodeLimits(timeout_s=10.0, timeout_steps=500).validate_against(0.02) == 500


def test_deadline_must_be_positive():
    with pytest.raises(ValueError):
        EpisodeLimits(timeout_s=0.0)
    with pytest.raises(ValueError):
        EpisodeLimits(timeout_steps=0)
