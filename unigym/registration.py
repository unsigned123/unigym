"""Process-global registration phases for every pack-registerable face.

The pack lifecycle is a monotonic sequence::

    MANIFEST -> ACTIVATE -> SPEC -> ASSEMBLE -> BACKEND_BUILD -> RUNTIME

Each registry face (adapter, robot, termination, scene provider, actuator law,
...) has a :class:`PhaseWindow` declaring when a registration is legal.
Bootstrap (``phase is None``) permits the core built-in imports: the first
explicit phase entered by a run takes over, after which registrations are
checked against the face window.

``RUNTIME`` is terminal and sticky: once a live run has been built no pack
registry may mutate.  :func:`seal_runtime` records that state.

This module is standard-library only: it owns the pure registration policy
data and imports no backend module (no composition cycle).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

__all__ = [
    "ACTUATOR_IMPLEMENTATION_WINDOWS",
    "BACKEND_REGISTRATION_POLICIES",
    "FACE_WINDOWS",
    "GATE",
    "BackendRegistrationPolicy",
    "PackFace",
    "PhaseWindow",
    "RegistrationError",
    "RegistrationGate",
    "RegistrationPhase",
    "RegistrationSealedError",
    "at_least_phase",
    "current_phase",
    "enter_phase",
    "is_sealed",
    "registration_phase",
    "require_registration",
    "reset_registration",
    "seal_runtime",
]


class RegistrationPhase(Enum):
    """The monotonic pack registration phases."""

    MANIFEST = "MANIFEST"
    ACTIVATE = "ACTIVATE"
    SPEC = "SPEC"
    ASSEMBLE = "ASSEMBLE"
    BACKEND_BUILD = "BACKEND_BUILD"
    RUNTIME = "RUNTIME"


PHASE_ORDER: tuple[RegistrationPhase, ...] = (
    RegistrationPhase.MANIFEST,
    RegistrationPhase.ACTIVATE,
    RegistrationPhase.SPEC,
    RegistrationPhase.ASSEMBLE,
    RegistrationPhase.BACKEND_BUILD,
    RegistrationPhase.RUNTIME,
)
_PHASE_INDEX = {phase: index for index, phase in enumerate(PHASE_ORDER)}


class PackFace(Enum):
    """Every face a pack may register through."""

    RUN_CONFIG = "run_config"
    SCENE_PRESET = "scene_preset"
    ASSET = "asset"
    SELECTION = "selection"
    SCENE_PROVIDER = "scene_provider"
    SCENE_PROVIDER_INLINE = "scene_provider_inline"
    SCENE_CONTENT_FILLER = "scene_content_filler"
    PROP_TEMPLATE = "prop_template"
    ROBOT = "robot"
    ADAPTER = "adapter"
    TERMINATION = "termination"
    INITIALIZER = "initializer"
    EXECUTOR = "executor"
    ACTUATOR_LAW = "actuator_law"
    ACTUATOR_IMPLEMENTATION = "actuator_implementation"


class RegistrationError(ValueError):
    """A registration happened outside its legal phase window."""


class RegistrationSealedError(RegistrationError):
    """A registration was attempted after the runtime was sealed."""


@dataclass(frozen=True)
class PhaseWindow:
    """An inclusive ``[earliest, latest]`` registration window."""

    earliest: RegistrationPhase
    latest: RegistrationPhase

    def __post_init__(self) -> None:
        if not isinstance(self.earliest, RegistrationPhase):
            raise TypeError(f"earliest must be a RegistrationPhase, got {self.earliest!r}")
        if not isinstance(self.latest, RegistrationPhase):
            raise TypeError(f"latest must be a RegistrationPhase, got {self.latest!r}")
        if _PHASE_INDEX[self.earliest] > _PHASE_INDEX[self.latest]:
            raise ValueError(
                f"phase window {self.earliest.value}..{self.latest.value} is empty "
                "(earliest must not follow latest)"
            )

    def contains(self, phase: RegistrationPhase) -> bool:
        """True when *phase* falls inside this inclusive window."""
        index = _PHASE_INDEX[phase]
        return _PHASE_INDEX[self.earliest] <= index <= _PHASE_INDEX[self.latest]

    def narrow(self, other: PhaseWindow) -> PhaseWindow:
        """Return the intersection with *other* (never widens)."""
        earliest = PHASE_ORDER[max(_PHASE_INDEX[self.earliest], _PHASE_INDEX[other.earliest])]
        latest = PHASE_ORDER[min(_PHASE_INDEX[self.latest], _PHASE_INDEX[other.latest])]
        if _PHASE_INDEX[earliest] > _PHASE_INDEX[latest]:
            raise RegistrationError(f"phase windows {self} and {other} do not intersect")
        return PhaseWindow(earliest, latest)

    def __str__(self) -> str:
        return f"{self.earliest.value}..{self.latest.value}"


@dataclass(frozen=True)
class BackendRegistrationPolicy:
    """A backend's narrowing of the global registration windows.

    ``windows`` maps a face to the backend's narrower window.  A face absent
    from ``windows`` keeps the global window.  ``denied_faces`` is a hard
    prohibition independent of phase.
    """

    backend: str
    windows: Mapping[PackFace, PhaseWindow] = field(default_factory=lambda: MappingProxyType({}))
    denied_faces: frozenset[PackFace] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", MappingProxyType(dict(self.windows)))
        object.__setattr__(self, "denied_faces", frozenset(self.denied_faces))

    def window_for(self, face: PackFace) -> PhaseWindow | None:
        """The backend window for *face*, or ``None`` when not narrowed."""
        if face in self.denied_faces:
            return None
        return self.windows.get(face)


#: Global face windows (the plan's face/window table).
FACE_WINDOWS: Mapping[PackFace, PhaseWindow] = MappingProxyType(
    {
        PackFace.RUN_CONFIG: PhaseWindow(RegistrationPhase.MANIFEST, RegistrationPhase.MANIFEST),
        PackFace.SCENE_PRESET: PhaseWindow(RegistrationPhase.MANIFEST, RegistrationPhase.SPEC),
        PackFace.ASSET: PhaseWindow(RegistrationPhase.MANIFEST, RegistrationPhase.SPEC),
        PackFace.SELECTION: PhaseWindow(RegistrationPhase.MANIFEST, RegistrationPhase.MANIFEST),
        PackFace.SCENE_PROVIDER: PhaseWindow(
            RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE
        ),
        PackFace.SCENE_PROVIDER_INLINE: PhaseWindow(RegistrationPhase.SPEC, RegistrationPhase.SPEC),
        PackFace.SCENE_CONTENT_FILLER: PhaseWindow(
            RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE
        ),
        PackFace.PROP_TEMPLATE: PhaseWindow(RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE),
        PackFace.ROBOT: PhaseWindow(RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE),
        PackFace.ADAPTER: PhaseWindow(RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE),
        PackFace.TERMINATION: PhaseWindow(RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE),
        PackFace.INITIALIZER: PhaseWindow(
            RegistrationPhase.ACTIVATE, RegistrationPhase.BACKEND_BUILD
        ),
        PackFace.EXECUTOR: PhaseWindow(RegistrationPhase.ACTIVATE, RegistrationPhase.SPEC),
        PackFace.ACTUATOR_LAW: PhaseWindow(RegistrationPhase.MANIFEST, RegistrationPhase.SPEC),
        PackFace.ACTUATOR_IMPLEMENTATION: PhaseWindow(
            RegistrationPhase.ASSEMBLE, RegistrationPhase.BACKEND_BUILD
        ),
    }
)

#: The actuator-implementation window each backend narrows to.
ACTUATOR_IMPLEMENTATION_WINDOWS: Mapping[str, PhaseWindow] = MappingProxyType(
    {
        "mujoco": PhaseWindow(RegistrationPhase.BACKEND_BUILD, RegistrationPhase.BACKEND_BUILD),
        "mjwarp": PhaseWindow(RegistrationPhase.BACKEND_BUILD, RegistrationPhase.BACKEND_BUILD),
        "isaaclab2": PhaseWindow(RegistrationPhase.ASSEMBLE, RegistrationPhase.ASSEMBLE),
        "isaaclab3_physx": PhaseWindow(RegistrationPhase.ASSEMBLE, RegistrationPhase.ASSEMBLE),
        "isaaclab3_newton": PhaseWindow(RegistrationPhase.ASSEMBLE, RegistrationPhase.ASSEMBLE),
    }
)

#: Canonical backend -> registration policy (only implementations narrow).
BACKEND_REGISTRATION_POLICIES: Mapping[str, BackendRegistrationPolicy] = MappingProxyType(
    {
        backend: BackendRegistrationPolicy(
            backend=backend,
            windows={PackFace.ACTUATOR_IMPLEMENTATION: window},
        )
        for backend, window in ACTUATOR_IMPLEMENTATION_WINDOWS.items()
    }
)


def _backend_policy(backend: str) -> BackendRegistrationPolicy | None:
    """Return the canonical policy for *backend* (no backend import)."""
    return BACKEND_REGISTRATION_POLICIES.get(backend)


_MIGRATION_HINT = (
    "register pack faces during ACTIVATE (pack runtime activation); actuator "
    "implementations are materialized by the run layer at ASSEMBLE/BACKEND_BUILD"
)


class RegistrationGate:
    """The process-global monotonic phase state and the registration check."""

    def __init__(self, phase: RegistrationPhase | None = None) -> None:
        self._phase = phase
        self._sealed = False

    @property
    def phase(self) -> RegistrationPhase | None:
        """The current phase, or ``None`` during bootstrap."""
        return self._phase

    @property
    def sealed(self) -> bool:
        """True once :meth:`seal_runtime` has run."""
        return self._sealed

    def enter(self, phase: RegistrationPhase) -> None:
        """Enter *phase*; only forward transitions (equal is a no-op) are legal."""
        if not isinstance(phase, RegistrationPhase):
            raise TypeError(f"phase must be a RegistrationPhase, got {phase!r}")
        if self._sealed:
            raise RegistrationSealedError(
                "registration gate is sealed at RUNTIME; no further phase may be entered"
            )
        current = self._phase
        if current is None:
            self._phase = phase
            return
        if _PHASE_INDEX[phase] < _PHASE_INDEX[current]:
            raise RegistrationError(
                f"cannot enter phase {phase.value} after {current.value}; phases are monotonic"
            )
        self._phase = phase

    def at_least(self, phase: RegistrationPhase) -> None:
        """Enter *phase* unless the gate is already at or past it."""
        if self._phase is not None and _PHASE_INDEX[self._phase] >= _PHASE_INDEX[phase]:
            return
        self.enter(phase)

    def seal_runtime(self) -> None:
        """Enter the terminal ``RUNTIME`` phase and seal the gate."""
        self._phase = RegistrationPhase.RUNTIME
        self._sealed = True

    def reset(self) -> None:
        """Reset to bootstrap (tests/repl only; never called by a live run)."""
        self._phase = None
        self._sealed = False

    def require(
        self,
        face: PackFace,
        *,
        name: str | None = None,
        backend: str | None = None,
    ) -> None:
        """Fail unless *face* registration is legal right now."""
        if self._sealed or self._phase is RegistrationPhase.RUNTIME:
            raise RegistrationSealedError(
                self._message(face, name, backend, phase=RegistrationPhase.RUNTIME)
            )
        phase = self._phase
        if phase is None:
            return
        if not isinstance(face, PackFace):
            raise TypeError(f"face must be a PackFace, got {face!r}")
        window = FACE_WINDOWS.get(face)
        if window is None:
            raise RegistrationError(f"no registration window is declared for face {face.value}")
        if not window.contains(phase):
            raise RegistrationError(self._message(face, name, backend, phase=phase, window=window))
        if backend is not None:
            policy = _backend_policy(backend)
            if policy is not None:
                if face in policy.denied_faces:
                    raise RegistrationError(
                        self._message(face, name, backend, phase=phase, window=window, denied=True)
                    )
                narrowed = policy.window_for(face)
                if narrowed is not None and not narrowed.contains(phase):
                    raise RegistrationError(
                        self._message(
                            face,
                            name,
                            backend,
                            phase=phase,
                            window=window,
                            policy_window=narrowed,
                        )
                    )

    def _message(
        self,
        face: PackFace,
        name: str | None,
        backend: str | None,
        *,
        phase: RegistrationPhase,
        window: PhaseWindow | None = None,
        policy_window: PhaseWindow | None = None,
        denied: bool = False,
    ) -> str:
        face_window = window or FACE_WINDOWS[face]
        label = f" {name!r}" if name is not None else ""
        parts = [
            f"registration of {face.value}{label} is not allowed in phase "
            f"{phase.value}; the global window is {face_window}"
        ]
        if backend is not None:
            if denied:
                parts.append(f"backend {backend!r} denies face {face.value}")
            elif policy_window is not None:
                parts.append(f"backend {backend!r} narrows it to {policy_window}")
            else:
                parts.append(f"backend {backend!r} applies no narrowing")
        parts.append(f"hint: {_MIGRATION_HINT}")
        return " | ".join(parts)


GATE = RegistrationGate()
"""The process-global registration gate."""


def current_phase() -> RegistrationPhase | None:
    """The current registration phase (``None`` during bootstrap)."""
    return GATE.phase


def is_sealed() -> bool:
    """True once the process-global gate has been sealed."""
    return GATE.sealed


def enter_phase(phase: RegistrationPhase) -> None:
    """Enter *phase* on the process-global gate."""
    GATE.enter(phase)


def at_least_phase(phase: RegistrationPhase) -> None:
    """Advance the process-global gate to at least *phase*."""
    GATE.at_least(phase)


def seal_runtime() -> None:
    """Enter ``RUNTIME`` and seal the process-global gate."""
    GATE.seal_runtime()


def reset_registration() -> None:
    """Reset the process-global gate to bootstrap (tests only)."""
    GATE.reset()


def require_registration(
    face: PackFace,
    *,
    name: str | None = None,
    backend: str | None = None,
) -> None:
    """Fail unless registering *face* now is legal; otherwise return ``None``.

    This is the single check every ``register_*`` helper calls at entry.
    """
    GATE.require(face, name=name, backend=backend)


def registration_phase() -> RegistrationPhase | None:
    """Backward-compatible alias for :func:`current_phase`."""
    return current_phase()
