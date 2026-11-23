"""Joint limits share the live record ``joint_pos`` policy order."""

from __future__ import annotations

import torch

from unigym.evaluation import EnvView, EvalContext
from unigym.runtime.controllers import ArticulationRuntime
from unigym.terminations import JointLimitTermination


def _ctx(joint_pos: torch.Tensor, limits: tuple[torch.Tensor, torch.Tensor]) -> EvalContext:
    # ``joint_names`` is asset order; the limits are policy/record order.
    art = ArticulationRuntime(
        name="robot",
        root="fixed",
        joint_names=("shoulder", "elbow"),
        q0=(0.0, 0.0),
        joint_pos_limits=limits,
    )
    return EvalContext(
        env=EnvView(
            record={"articulation/robot/joint_pos": joint_pos},
            obs=None,
            env_id=None,
            num_envs=1,
            device="cpu",
            articulations={"robot": art},
        )
    )


def test_joint_limits_follow_record_joint_pos_order_not_asset_order():
    """Policy-order limits and positions line up even when asset order differs."""
    limits = (
        torch.tensor([[-0.5, -1.57]]),  # elbow, shoulder
        torch.tensor([[1.0, 1.57]]),
    )
    term = JointLimitTermination({"articulation": "robot"})

    elbow_below = _ctx(torch.tensor([[-0.6, 0.0]]), limits)
    shoulder_above = _ctx(torch.tensor([[0.0, 1.6]]), limits)
    inside = _ctx(torch.tensor([[0.0, 0.0]]), limits)

    assert term.check(elbow_below).tolist() == [True]
    assert term.check(shoulder_above).tolist() == [True]
    assert term.check(inside).tolist() == [False]
    lower, upper = inside.env.joint_pos_limits("robot")
    assert torch.equal(lower, limits[0])
    assert torch.equal(upper, limits[1])
