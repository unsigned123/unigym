"""Composite termination groups: OR/AND, inversion, empty groups and validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.evaluation import EnvView, EvalContext
from unigym.runtime.controllers import ArticulationRuntime
from unigym.terminations import (
    FallTermination,
    InvertedTermination,
    Termination,
    TerminationGroup,
)
from unigym.terminations.composite import _same_device


def _ctx(*, num_envs: int = 2, env_id: int | None = None) -> EvalContext:
    record = {
        "articulation/robot/base_pos_w": torch.tensor(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]][:num_envs]
        ),
        "articulation/robot/gravity_b": torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]][:num_envs]
        ),
    }
    arts = {
        "robot": ArticulationRuntime(
            name="robot",
            root="fixed",
            joint_names=("j0", "j1"),
            q0=(0.0, 0.0),
        )
    }
    return EvalContext(
        env=EnvView(
            record=record,
            obs=None,
            env_id=env_id,
            num_envs=num_envs,
            device="cpu",
            articulations=arts,
        )
    )


class _FixedTermination(Termination):
    """Return one already-built mask and record the received context."""

    def __init__(self, mask, name: str | None = None) -> None:
        super().__init__(None, name)
        self.mask = mask
        self.calls = 0
        self.seen: EvalContext | None = None

    def check(self, ctx: EvalContext) -> torch.Tensor:
        self.calls += 1
        self.seen = ctx
        return self.mask


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("or", [True, True]),
        ("and", [True, False]),
    ],
)
def test_composite_group_combines_masks_with_its_declared_mode(mode, expected):
    """OR fires when any term fires; AND fires only when every term fires."""
    low_only = FallTermination(
        {"articulation": "robot", "height_threshold": 0.15, "tilt_threshold": None},
        name="low_only",
    )
    wide_height = FallTermination(
        {"articulation": "robot", "height_threshold": 0.6, "tilt_threshold": None},
        name="wide_height",
    )
    group = TerminationGroup([low_only, wide_height], mode=mode)

    assert group.check(_ctx()).tolist() == expected


def test_inverted_termination_complements_the_wrapped_mask():
    """Inversion turns a fault mask into its health complement."""
    fall = FallTermination({"articulation": "robot", "tilt_threshold": None})
    inverted = InvertedTermination(fall, name="healthy")

    assert fall.check(_ctx()).tolist() == [True, False]
    assert inverted.check(_ctx()).tolist() == [False, True]


def test_composite_empty_group_never_fires():
    """An empty group is inactive: it reports an all-False bool mask."""
    mask = TerminationGroup([]).check(_ctx(num_envs=3))

    assert isinstance(mask, torch.Tensor)
    assert tuple(mask.shape) == (3,)
    assert mask.dtype == torch.bool
    assert not bool(mask.any())


def test_composite_group_rejects_unknown_mode():
    """Only OR and AND are legal; any other mode fails naming the value."""
    with pytest.raises(ValueError, match="mode must be 'or' or 'and', got 'xor'"):
        TerminationGroup([], mode="xor")


def test_composite_group_rejects_a_per_env_context():
    """Termination groups judge the whole live vector, not one episode."""
    fall = FallTermination({"articulation": "robot"})

    with pytest.raises(ValueError, match="requires a vector evaluation context"):
        TerminationGroup([fall]).check(_ctx(env_id=0))


def test_composite_group_passes_the_same_context_to_every_term():
    """Each term receives the one boundary context object."""
    first = _FixedTermination(torch.zeros(2, dtype=torch.bool), name="first")
    second = _FixedTermination(torch.zeros(2, dtype=torch.bool), name="second")
    ctx = _ctx()

    TerminationGroup([first, second]).check(ctx)

    assert first.seen is ctx
    assert second.seen is ctx


@pytest.mark.parametrize(
    ("mask", "error", "match"),
    [
        (np.zeros(2, dtype=np.bool_), TypeError, "expected a torch.Tensor"),
        (torch.zeros(2, dtype=torch.float32), TypeError, "expected torch.bool"),
        (torch.zeros(3, dtype=torch.bool), ValueError, r"shape .*expected \(2,\)"),
        (
            torch.zeros(2, dtype=torch.bool, device="meta"),
            ValueError,
            "expected cpu",
        ),
    ],
)
def test_composite_group_validates_mask_shape_dtype_and_device(mask, error, match):
    """A malformed custom mask fails fast instead of corrupting the group mask."""
    term = _FixedTermination(mask, name="bad")

    with pytest.raises(error, match=match):
        TerminationGroup([term]).check(_ctx(num_envs=2))


def test_composite_combine_returns_none_for_an_empty_detail():
    """The raw reducer has no reference to synthesize an empty mask from."""
    assert TerminationGroup([]).combine({}) is None


def test_device_comparison_normalizes_unindexed_cuda():
    """An unindexed ``cuda`` and its concrete index address the same device."""
    assert _same_device(torch.device("cuda"), torch.device("cuda:0"))
    assert _same_device("cpu", torch.device("cpu"))
    assert not _same_device(torch.device("cuda:0"), torch.device("cuda:1"))
