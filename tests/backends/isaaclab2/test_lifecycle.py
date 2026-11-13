"""IsaacLab 2 Cartpole close lifecycle."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]


def test_close_is_idempotent(cartpole_env):
    """Closing the real Kit/PhysX env twice does not raise."""
    cartpole_env.close()
    cartpole_env.close()


@pytest.mark.parametrize(
    ("method", "args"),
    [("step", ({},)), ("reset_envs", ([0],)), ("reset_all", ())],
)
def test_closed_env_rejects_use(cartpole_env, method, args):
    """Stepping or resetting after close raises the backend closed error."""
    cartpole_env.close()

    with pytest.raises(RuntimeError) as excinfo:
        getattr(cartpole_env, method)(*args)

    assert str(excinfo.value) == "isaaclab2 backend: environment is closed"
