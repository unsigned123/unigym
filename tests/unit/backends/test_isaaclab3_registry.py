"""Canonical-only backend registry contract for the IsaacLab 3 identities.

Selection is canonical-only: the five canonical names are the whole public
surface, the removed ``isaaclab`` / ``isaaclab3`` aliases are rejected, and
the stack decision is a pure function of a :class:`StackProbe` value (no
monkeypatched version probes).
"""

from __future__ import annotations

import pytest

from unigym.backends.registry import (
    CANONICAL_BACKENDS,
    BackendSelectionError,
    StackProbe,
    backend_info,
    build_name,
    known_backends,
    selection_error,
    validate_backend_options,
)

CANONICAL = (
    "mujoco",
    "mjwarp",
    "isaaclab2",
    "isaaclab3_physx",
    "isaaclab3_newton",
)
REMOVED_ALIASES = ("isaaclab", "isaaclab3")


def test_canonical_backends_are_exactly_the_five_names() -> None:
    assert CANONICAL_BACKENDS == CANONICAL
    assert known_backends() == list(CANONICAL)
    assert "isaaclab" not in CANONICAL_BACKENDS
    assert "isaaclab3" not in CANONICAL_BACKENDS


@pytest.mark.parametrize("name", CANONICAL)
def test_build_name_accepts_every_canonical_name(name: str) -> None:
    assert build_name(name) == name


@pytest.mark.parametrize("name", [None, "", *REMOVED_ALIASES, "bogus"])
def test_build_name_rejects_none_empty_aliases_and_unknown(name: str | None) -> None:
    with pytest.raises(ValueError, match=r"aliases were removed"):
        build_name(name)


@pytest.mark.parametrize("name", CANONICAL)
def test_backend_info_accepts_every_canonical_name(name: str) -> None:
    info = backend_info(name)
    assert info.name == name


@pytest.mark.parametrize("name", [None, *REMOVED_ALIASES, "bogus"])
def test_backend_info_rejects_aliases_none_and_unknown(name: str | None) -> None:
    with pytest.raises(BackendSelectionError, match=r"unknown env backend"):
        backend_info(name)


_SATISFIED = StackProbe(
    isaaclab=(3, 0),
    newton=(0, 1),
    mujoco=(3, 0),
    mujoco_warp=(0, 1),
    warp=(1, 0),
)


@pytest.mark.parametrize(
    ("canonical", "probe"),
    [
        # mujoco: needs the mujoco stack only.
        ("mujoco", StackProbe()),
        ("mujoco", StackProbe(mujoco=(3, 0))),
        ("mujoco", _SATISFIED),
        # mjwarp: needs mujoco-warp + warp-lang.
        ("mjwarp", StackProbe(mujoco_warp=(0, 1))),
        ("mjwarp", StackProbe(warp=(1, 0))),
        ("mjwarp", _SATISFIED),
        # isaaclab2: needs isaaclab with major < 3.
        ("isaaclab2", StackProbe()),
        ("isaaclab2", StackProbe(isaaclab=(2, 3))),
        ("isaaclab2", _SATISFIED),
        # isaaclab3_physx: needs isaaclab with major >= 3.
        ("isaaclab3_physx", StackProbe()),
        ("isaaclab3_physx", StackProbe(isaaclab=(2, 3))),
        ("isaaclab3_physx", StackProbe(isaaclab=(3, 0))),
        ("isaaclab3_physx", _SATISFIED),
        # isaaclab3_newton: needs isaaclab 3.x + newton + mujoco-warp + warp.
        ("isaaclab3_newton", StackProbe()),
        ("isaaclab3_newton", StackProbe(isaaclab=(3, 0))),
        ("isaaclab3_newton", StackProbe(isaaclab=(3, 0), mujoco_warp=(0, 1), warp=(1, 0))),
        ("isaaclab3_newton", StackProbe(isaaclab=(3, 0), newton=(0, 1), warp=(1, 0))),
        ("isaaclab3_newton", StackProbe(isaaclab=(3, 0), newton=(0, 1), mujoco_warp=(0, 1))),
        ("isaaclab3_newton", _SATISFIED),
    ],
)
def test_selection_error_is_the_pure_stack_decision(canonical: str, probe: StackProbe) -> None:
    """Feed fake probes; never monkeypatch the live version probes."""
    expected = {
        "mujoco": probe.mujoco,
        "mjwarp": probe.mujoco_warp if probe.warp is not None else None,
        "isaaclab2": probe.isaaclab if (probe.isaaclab and probe.isaaclab[0] < 3) else None,
        "isaaclab3_physx": (
            probe.isaaclab if (probe.isaaclab and probe.isaaclab[0] >= 3) else None
        ),
        "isaaclab3_newton": (
            probe.isaaclab
            if (
                probe.isaaclab
                and probe.isaaclab[0] >= 3
                and probe.newton is not None
                and probe.mujoco_warp is not None
                and probe.warp is not None
            )
            else None
        ),
    }[canonical]
    if expected is None:
        assert isinstance(selection_error(canonical, probe), BackendSelectionError)
    else:
        assert selection_error(canonical, probe) is None


def test_selection_error_rejects_unknown_spellings() -> None:
    for name in (*REMOVED_ALIASES, "bogus"):
        assert isinstance(selection_error(name, _SATISFIED), BackendSelectionError)


@pytest.mark.parametrize("backend", CANONICAL)
def test_validate_backend_options_accepts_empty_options(backend: str) -> None:
    assert isinstance(validate_backend_options(backend, None), dict)
    assert isinstance(validate_backend_options(backend, {}), dict)


@pytest.mark.parametrize("alias", REMOVED_ALIASES)
def test_validate_backend_options_rejects_removed_alias_namespaces(alias: str) -> None:
    """The removed spellings are not selectable at all."""
    with pytest.raises(BackendSelectionError):
        validate_backend_options(alias, {})


@pytest.mark.parametrize("alias", REMOVED_ALIASES)
def test_validate_backend_options_rejects_foreign_alias_option_namespaces(
    alias: str,
) -> None:
    """An options block keyed by a removed alias is a foreign namespace."""
    with pytest.raises(ValueError, match=r"only accepts its own namespace"):
        validate_backend_options("mujoco", {alias: {}})


@pytest.mark.parametrize(
    "backend",
    ["isaaclab3_physx", "isaaclab3_newton"],
)
def test_isaaclab3_env_kwargs_factory_is_wired(backend: str) -> None:
    from unigym.backends.shared.isaaclab3_common.build import env_kwargs

    info = backend_info(backend)
    assert info.env_kwargs is not None
    assert info.env_kwargs() is env_kwargs
