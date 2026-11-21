"""IsaacLab 3 Newton runtime-query capability identity.

The Newton lane does not build a Cartpole env; this file checks the Newton
identity against the registry and its own capability object.
"""

from __future__ import annotations

import pytest

from unigym.backends.registry import Trait, backend_info
from unigym.backends.shared.isaaclab3_common.query import (
    ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
)

pytestmark = pytest.mark.sim


def test_newton_query_capabilities_and_registry_declaration() -> None:
    """Newton declares RUNTIME_QUERY with its own honest capability identity."""
    info = backend_info("isaaclab3_newton")
    caps = ISAACLAB3_NEWTON_QUERY_CAPABILITIES

    assert Trait.RUNTIME_QUERY in info.traits
    assert info.query_backend is not None
    assert caps.backend == "isaaclab3_newton"
    assert caps.supports("model.ir") == "native"
    assert caps.supports("state.body_pose") == "native"
    assert caps.supports("scratch.copy") == "synthetic"
    assert caps.supports("scratch.live_untouched") == "partial"
    assert caps.scratch.mode == "transaction"
    assert caps.scratch.isolation == "restored"
    assert caps.supports("scratch.geometry") == "unsupported"
    assert caps.supports("geometry.distance") == "unsupported"
