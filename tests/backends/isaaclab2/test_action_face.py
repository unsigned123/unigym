"""IsaacLab 2 body action term is named after the articulation."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sim


def test_action_entry_is_the_articulation_name(cartpole_env):
    """A non-G1 articulation gets its own action entry and no G1 alias."""
    from unigym.env.backends.base import assert_backend_contract

    assert cartpole_env.action_entry_names == ["cartpole"]
    assert "g1" not in cartpole_env.action_entry_names
    assert_backend_contract(cartpole_env)
