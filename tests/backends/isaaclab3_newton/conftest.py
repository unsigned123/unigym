"""Environment gates for the IsaacLab 3 Newton backend lane."""

from __future__ import annotations

import os

import pytest

from tests._support.envs import require_optional_module

# The rc1 launcher requires the EULA to be accepted non-interactively.  This is
# lane-local because the runner does not inject it.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


@pytest.fixture(scope="session", autouse=True)
def _stack() -> None:
    for module in (
        "isaaclab",
        "isaaclab_physx",
        "isaaclab_newton",
        "newton",
        "mujoco",
        "mujoco_warp",
        "warp",
    ):
        require_optional_module(module, reason="the IsaacLab 3 Newton lane")
