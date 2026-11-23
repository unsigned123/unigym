"""Live-first scoring: built-in/formula tasks accept and ignore the context."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.benchmarks.base import EpisodeData
from unigym.benchmarks.formula import FormulaTask
from unigym.benchmarks.tasks import StandTask, VelocityTrackingTask
from unigym.evaluation import EnvView, EvalContext

_ARTICULATIONS = {"g1": {}}


def _episode(env_id: int = 0, frames: int = 3) -> EpisodeData:
    return EpisodeData(
        data={
            "articulation/g1/gravity_b": np.tile(np.array([[0.0, 0.0, -1.0]]), (frames, 1)),
            "articulation/g1/base_pos": np.tile(np.array([[0.0, 0.0, 0.8]]), (frames, 1)),
            "articulation/g1/base_lin_vel": np.tile(np.array([[0.4, 0.0, 0.0]]), (frames, 1)),
            "articulation/g1/command": np.tile(np.array([[0.4, 0.0, 0.0]]), (frames, 1)),
            "time_in_episode": np.arange(frames, dtype=float) * 0.02,
        },
        env_id=env_id,
        episode_id=1,
        length=frames,
        duration_s=0.02 * frames,
        ended_by_termination=False,
        policy_dt=0.02,
        sim_dt=0.005,
    )


def _context(ep: EpisodeData) -> EvalContext:
    return EvalContext(
        env=EnvView(
            record={"articulation/g1/joint_pos": torch.zeros(2, 2)},
            obs=None,
            env_id=ep.env_id,
            num_envs=2,
            backend="fake",
        )
    )


def test_stand_task_metrics_are_unchanged_with_a_context():
    """The live context does not change ``stand``'s episode-derived metrics."""
    ep = _episode()
    task = StandTask(params={"articulation": "g1"}, meta={"articulations": _ARTICULATIONS})

    result = task.score(ep, _context(ep))

    assert result.metrics == pytest.approx(
        {
            "alive_fraction": 1.0,
            "base_z_mean": 0.8,
            "base_z_var": 0.0,
            "tilt_exceed_fraction": 0.0,
        }
    )
    assert result.success is True


def test_velocity_tracking_metrics_are_unchanged_with_a_context():
    """The live context does not change ``velocity_tracking`` metrics."""
    ep = _episode()
    task = VelocityTrackingTask(
        params={"articulation": "g1"},
        meta={"articulations": _ARTICULATIONS, "has_command": True},
    )

    result = task.score(ep, _context(ep))

    assert result.metrics == {
        "tracking_error_mean": 0.0,
        "tracking_error_max": 0.0,
        "success_fraction": 1.0,
    }
    assert result.success is True


def test_formula_score_and_score_detail_accept_the_context():
    """FormulaTask keeps its metric/predicate/frame faces with ``ctx``."""
    ep = _episode()
    task = FormulaTask(
        params={
            "description": "mean episode time",
            "metrics": [{"id": "tmean", "term": "frame_mean", "key": "time_in_episode"}],
            "success": {"all": [{"metric": "tmean", "gte": 0.0}]},
        },
        meta={"articulations": _ARTICULATIONS},
    )

    result = task.score(ep, _context(ep))
    detail, leaves, frames = task.score_detail(ep, _context(ep), retain_frames=True)

    assert result.metrics == {"tmean": float(np.mean([0.0, 0.02, 0.04]))}
    assert result.success is True
    assert detail.metrics == result.metrics
    assert leaves == {"tmean": True}
    assert set(frames) == {"tmean"}
