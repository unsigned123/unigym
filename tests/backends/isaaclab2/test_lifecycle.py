"""IsaacLab 2 Cartpole close lifecycle."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sim


def test_close_is_idempotent(cartpole_env):
    """Closing the real Kit/PhysX env twice does not raise."""
    cartpole_env.close()
    cartpole_env.close()
