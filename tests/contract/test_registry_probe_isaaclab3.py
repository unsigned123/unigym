"""The canonical-only selection contract for the IsaacLab 3 identities.

These assertions only read the registry and the pure :class:`StackProbe`
decision function; they never import an engine.  The live missing-stack cases
describe the d0 base environment, so they are skipped when *any* IsaacLab
stack is installed (2.x and 3.x both change the observed error); the 2.x/3.x
messages are covered deterministically by the pure probes below.
"""

from __future__ import annotations

import pytest

from unigym.capabilities.selection import (
    BackendSelectionError,
    StackProbe,
    select_backend,
    selection_error,
)
from unigym.capabilities.versions import isaaclab_version


def _message(exc: BackendSelectionError) -> str:
    return f"{exc.error} | {exc.hint}"


@pytest.mark.parametrize("name", ["isaaclab3", "isaaclab", None])
def test_select_backend_rejects_removed_aliases_and_none(name: str | None) -> None:
    with pytest.raises(BackendSelectionError) as excinfo:
        select_backend(name)

    message = _message(excinfo.value)
    for canonical in (
        "mujoco",
        "mjwarp",
        "isaaclab2",
        "isaaclab3_physx",
        "isaaclab3_newton",
    ):
        assert canonical in message
    assert "aliases were removed" in message


def _require_d0() -> None:
    """Skip the live missing-stack cases unless NO IsaacLab stack is installed.

    An installed 2.x or 3.x stack makes ``select_backend`` report a version
    mismatch instead of the missing-stack error, so every installed version is
    skipped here; both messages are asserted with pure probes below.
    """
    version = isaaclab_version()
    if version is not None:
        pytest.skip(
            "requires the d0 base environment without an IsaacLab stack "
            f"(found isaaclab {version[0]}.{version[1]})"
        )


def test_select_backend_physx_fails_without_the_isaaclab_stack() -> None:
    _require_d0()
    with pytest.raises(BackendSelectionError) as excinfo:
        select_backend("isaaclab3_physx")

    assert "isaaclab stack" in _message(excinfo.value)


def test_select_backend_newton_fails_without_the_isaaclab_stack() -> None:
    _require_d0()
    with pytest.raises(BackendSelectionError) as excinfo:
        select_backend("isaaclab3_newton")

    assert "isaaclab stack" in _message(excinfo.value)


@pytest.mark.parametrize("canonical", ["isaaclab3_physx", "isaaclab3_newton"])
def test_isaaclab3_identities_require_the_isaaclab_stack(canonical: str) -> None:
    """Pure probe: absent isaaclab is the missing-stack error, not a TypeError."""
    error = selection_error(canonical, StackProbe())

    assert error is not None
    assert "isaaclab stack" in _message(error)


@pytest.mark.parametrize("canonical", ["isaaclab3_physx", "isaaclab3_newton"])
@pytest.mark.parametrize("version", [(2, 3), (1, 0)])
def test_isaaclab3_identities_reject_an_installed_isaaclab2_stack(
    canonical: str, version: tuple[int, int]
) -> None:
    """Any installed 2.x stack is rejected with the version, never selected."""
    error = selection_error(canonical, StackProbe(isaaclab=version))

    assert error is not None
    message = _message(error)
    assert canonical in message
    assert f"{version[0]}.{version[1]}" in message
    assert "isaaclab2" in message


def test_isaaclab2_rejects_an_installed_isaaclab3_stack() -> None:
    """The 2.x identity does not silently share a 3.x installation."""
    error = selection_error("isaaclab2", StackProbe(isaaclab=(3, 0)))

    assert error is not None
    message = _message(error)
    assert "3.0" in message
    assert "isaaclab3_physx" in message


def test_isaaclab3_newton_requires_the_whole_newton_stack() -> None:
    """IsaacLab 3.x alone does not satisfy the Newton identity."""
    error = selection_error("isaaclab3_newton", StackProbe(isaaclab=(3, 0)))

    assert error is not None
    message = _message(error)
    for dist in ("newton", "mujoco-warp", "warp-lang"):
        assert dist in message
