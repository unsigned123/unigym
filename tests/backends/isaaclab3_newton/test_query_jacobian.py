"""IsaacLab 3 Newton runtime-query Jacobian capability identity.

The Newton lane does not build a Cartpole env here; this file checks that the
Newton identity declares the same documented Jacobian support levels.  Its
native body/COM Jacobian implementation is exercised by the shared query code
in the PhysX lane and in the unit fake.
"""

from __future__ import annotations

import pytest

from unigym.capabilities.specs import backend_spec
from unigym.capabilities.traits import Trait
from unigym.backends.shared.isaaclab3_common.query import (
    ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
)

pytestmark = pytest.mark.sim


def test_newton_declares_native_and_synthetic_jacobian_levels() -> None:
    """Newton shares the IL3 Jacobian contract: native body/COM, synthetic site/point."""
    info = backend_spec("isaaclab3_newton")
    caps = ISAACLAB3_NEWTON_QUERY_CAPABILITIES

    assert Trait.RUNTIME_QUERY in info.traits
    assert caps.supports("jacobian.body") == "native"
    assert caps.supports("jacobian.com") == "partial"
    assert caps.supports("jacobian.site") == "synthetic"
    assert caps.supports("jacobian.point") == "synthetic"
