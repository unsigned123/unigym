"""The shared reset-time extra reader keeps its branch order and defaults."""

from __future__ import annotations

import torch

from unigym.runner.state import read_extra
from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from tests._support.fakes.runner import BuildOnlyEnv, BuildStateEnv, DeviceOnlyEnv, ExtraOnlyEnv


def test_build_extra_consumes_the_read_state_payload() -> None:
    """A backend with both seams gets ``_build_extra(_read_state())``."""
    env = BuildStateEnv()

    extra = read_extra(env)

    assert extra["built_from"] == {"reader": "state"}
    assert env.read_calls == 1
    assert env.build_calls == 1
    assert extra["should_reset"] is False
    assert isinstance(extra["reset_env_ids"], torch.Tensor)
    assert extra["reset_env_ids"].numel() == 0
    assert extra["reset_env_ids"].dtype == torch.int64


def test_build_extra_without_read_state_is_called_directly() -> None:
    """A build-only seam is called with no argument."""
    env = BuildOnlyEnv()

    extra = read_extra(env)

    assert extra["built"] is True
    assert env.build_calls == 1
    assert extra["should_reset"] is False


def test_extra_fallback_is_used_when_no_builder_exists() -> None:
    """A minimal double's ``_extra()`` mapping is adopted unchanged."""
    env = ExtraOnlyEnv()

    extra = read_extra(env)

    assert extra["fallback"] is True
    assert env.extra_calls == 1
    assert extra["should_reset"] is False


def test_env_without_any_state_face_returns_only_the_defaults() -> None:
    """No state builder means the two bookkeeping defaults only."""
    extra = read_extra(DeviceOnlyEnv())

    assert set(extra) == {"reset_env_ids", "should_reset"}
    assert extra["should_reset"] is False
    assert isinstance(extra["reset_env_ids"], torch.Tensor)
    assert extra["reset_env_ids"].numel() == 0


def test_existing_bookkeeping_values_are_not_overwritten() -> None:
    """The shared backend extra already carries defaults; the reader keeps them."""
    env = FakeBackend()

    extra = read_extra(env)

    assert f"articulation/{ARTICULATION_NAME}/joint_pos" in extra
    assert extra["should_reset"] is False
    assert isinstance(extra["reset_env_ids"], torch.Tensor)
    assert extra["reset_env_ids"].numel() == 0
