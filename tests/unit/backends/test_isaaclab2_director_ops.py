"""IsaacLab 2 director ops identity + engine-free weld capability query.

The ops object is pure bookkeeping until an op runs: constructing it and
asking :meth:`has_weld` on the d0 base environment (no engine installed) must
answer from the static trait table, not raise ``BackendSelectionError``.
"""

from __future__ import annotations

from unigym.backends.isaaclab2.director_ops import IsaacLabDirectorOps
from unigym.capabilities.specs import backend_spec
from unigym.capabilities.traits import Trait


class _EnvCarrier:
    """Minimal env carrier: ops construction touches nothing else."""

    device = "cpu"
    prop_manager = None
    robot = None


def test_isaaclab2_director_ops_identity() -> None:
    ops = IsaacLabDirectorOps(_EnvCarrier())

    assert ops.backend == "isaaclab2"


def test_has_weld_is_a_static_boolean_without_an_engine() -> None:
    ops = IsaacLabDirectorOps(_EnvCarrier())

    result = ops.has_weld()

    assert isinstance(result, bool)
    assert result == backend_spec("isaaclab2").has(Trait.DIRECTOR_ATTACH_WELD)


def test_backend_spec_resolves_the_isaaclab2_identity() -> None:
    info = backend_spec("isaaclab2")

    assert info.name == "isaaclab2"
    assert info.has(Trait.ENV_STEP_RESET)
