"""Canonical backend selection and live-stack validation.

Selection is canonical-only: no ``isaaclab`` / ``isaaclab3`` aliases and no
version-based family auto-selection.  The explicitly chosen stack is verified
against a live probe and never falls back silently.  This module depends only
on :mod:`unigym.capabilities.traits` and
:mod:`unigym.capabilities.versions`.
"""

from __future__ import annotations

from dataclasses import dataclass

from unigym.capabilities.traits import CANONICAL_BACKENDS
from unigym.capabilities.versions import (
    isaaclab_version,
    mujoco_version,
    mujoco_warp_version,
    newton_version,
    warp_version,
)

__all__ = [
    "BackendSelectionError",
    "StackProbe",
    "canonical_or_error",
    "select_backend",
    "selection_error",
]


class BackendSelectionError(RuntimeError):
    """A backend was requested that the probe cannot satisfy.

    Carries the ``{error, hint}`` pair the CLI/daemon surfaces verbatim
    (fail-fast: no silent fallback to another backend).
    """

    def __init__(self, error: str, hint: str = ""):
        super().__init__(error)
        self.error = error
        self.hint = hint or "(none)"


def _canonical_hint() -> str:
    """The teaching hint shared by every rejection path."""
    return (
        f"available backends: {', '.join(CANONICAL_BACKENDS)} "
        f"(canonical names only; 'isaaclab'/'isaaclab3' aliases were removed)"
    )


def canonical_or_error(name: str | None) -> str:
    """Return *name* if it is canonical, else raise ``BackendSelectionError``.

    No aliases and no version probing: ``None``/``""``/``isaaclab``/
    ``isaaclab3`` are rejected just like any unknown spelling.
    """
    raw = "" if name is None else str(name).strip().lower()
    if raw in CANONICAL_BACKENDS:
        return raw
    raise BackendSelectionError(f"unknown env backend {name!r}", _canonical_hint())


@dataclass(frozen=True)
class StackProbe:
    """A snapshot of the engine stacks a backend selection may require.

    The pure decision function :func:`selection_error` consumes this value
    object, so unit tests feed fake probes instead of monkeypatching the live
    version probes.  Each field is ``(major, minor)`` or ``None`` (absent).
    """

    isaaclab: tuple[int, int] | None = None
    newton: tuple[int, int] | None = None
    mujoco: tuple[int, int] | None = None
    mujoco_warp: tuple[int, int] | None = None
    warp: tuple[int, int] | None = None


def selection_error(canonical: str, probe: StackProbe) -> BackendSelectionError | None:
    """Decide whether *canonical* is usable against *probe* (pure).

    Never calls a version probe: the caller owns the :class:`StackProbe`.
    Returns the :class:`BackendSelectionError` for an unusable selection, or
    ``None`` when the explicitly chosen stack is satisfied.
    """
    if canonical not in CANONICAL_BACKENDS:
        return BackendSelectionError(
            f"unknown env backend {canonical!r}",
            _canonical_hint(),
        )

    if canonical == "mujoco":
        if probe.mujoco is None:
            return BackendSelectionError(
                "backend 'mujoco' requested but the mujoco stack is not "
                "importable in this environment",
                "run on the host .venv_sim (pip install mujoco) or pick a " "different env.backend",
            )
        return None

    if canonical == "mjwarp":
        missing = [
            dist
            for dist, ver in (("mujoco-warp", probe.mujoco_warp), ("warp-lang", probe.warp))
            if ver is None
        ]
        if missing:
            return BackendSelectionError(
                "backend 'mjwarp' requested but the "
                + " + ".join(missing)
                + " stack is not importable in this environment",
                "install the warp extra (pip install 'unigym[warp]'; "
                "mujoco>=3.13 + mujoco-warp + warp-lang) or pick a "
                "different env.backend",
            )
        return None

    ver = probe.isaaclab
    if canonical == "isaaclab2":
        if ver is None:
            return BackendSelectionError(
                "backend 'isaaclab2' requested but the isaaclab stack is not "
                "importable in this environment",
                "isaaclab runs need the isaac-lab container (AppLauncher must "
                "start first); check env.backend in the config",
            )
        if ver[0] >= 3:
            return BackendSelectionError(
                f"backend 'isaaclab2' requested but installed isaaclab is {ver[0]}.{ver[1]}",
                "set env.backend: isaaclab3_physx or isaaclab3_newton "
                "(or install the isaaclab 2.3 stack)",
            )
        return None

    # The two IsaacLab 3 identities both require IsaacLab 3.x.
    if ver is None:
        return BackendSelectionError(
            f"backend {canonical!r} requested but the isaaclab stack is not "
            f"importable in this environment",
            "isaaclab runs need the isaac-lab container (AppLauncher must "
            "start first); check env.backend in the config",
        )
    if ver[0] < 3:
        return BackendSelectionError(
            f"backend {canonical!r} requested but installed isaaclab is {ver[0]}.{ver[1]}",
            "set env.backend: isaaclab2 (or install isaaclab 3.x)",
        )
    if canonical == "isaaclab3_physx":
        return None

    # isaaclab3_newton: IsaacLab 3.x plus newton / mujoco-warp / warp-lang.
    missing = [
        dist
        for dist, dist_ver in (
            ("newton", probe.newton),
            ("mujoco-warp", probe.mujoco_warp),
            ("warp-lang", probe.warp),
        )
        if dist_ver is None
    ]
    if missing:
        return BackendSelectionError(
            "backend 'isaaclab3_newton' requested but the "
            + " + ".join(missing)
            + " stack is not importable in this environment",
            "install newton + mujoco-warp + warp-lang (the IsaacLab 3 Newton "
            "extra) or set env.backend: isaaclab3_physx",
        )
    return None


def _live_probe() -> StackProbe:
    """Snapshot the live stacks for :func:`select_backend` (probes once)."""
    return StackProbe(
        isaaclab=isaaclab_version(),
        newton=newton_version(),
        mujoco=mujoco_version(),
        mujoco_warp=mujoco_warp_version(),
        warp=warp_version(),
    )


def select_backend(name: str) -> str:
    """Verify the explicitly chosen canonical backend against the live stack.

    Canonical-only: no aliases and no automatic family selection.  ``None``,
    ``""``, ``"isaaclab"``, ``"isaaclab3"`` and unknown names are rejected.
    The chosen backend's stack is verified with :func:`selection_error`; a
    missing or mismatched stack raises :class:`BackendSelectionError` with an
    actionable hint — never a silent fallback.
    """
    canonical = canonical_or_error(name)
    error = selection_error(canonical, _live_probe())
    if error is not None:
        raise error
    return canonical
