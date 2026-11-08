"""Backend-private name map for composed MuJoCo-family assets (v105.3).

The neutral contract speaks only ``unigym_name`` (the name declared by the
run config).  A composed backend asset may carry a backend-owned prefix; the
mapping between the two lives here and never leaks into the neutral
``ControlRuntime`` / ``ArticulationRuntime`` / ``ModelSet`` surfaces.

Prefix policy is backend-owned: the default is no prefix
(``compose.auto_prefix: false``) so names reflect the compiled asset; an
explicit ``auto_prefix: true`` attaches every asset-bearing articulation
with the short ``<articulation>_`` prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "MjArticulationBinding",
    "MjComposition",
    "MjControlBinding",
    "MjNameMap",
]


def _strip(prefix: str | None, name: str) -> str:
    """Strip *prefix* from *name* when present."""
    if prefix and name.startswith(prefix):
        return name[len(prefix):]
    return name


@dataclass(frozen=True)
class MjNameMap:
    """``(unigym articulation, unigym name)`` <-> compiled backend name.

    Attributes:
        prefixes: ``{articulation: prefix-or-None}``; ``None`` means the
            articulation's compiled names are its unigym names verbatim.
        joints: ``{articulation: tuple[unigym asset joint names]}`` in asset
            order (compiled order).
        bodies: ``{articulation: tuple[unigym body names]}``.
        actuators: ``{articulation: tuple[unigym actuator names]}``.
    """

    prefixes: Mapping[str, str | None] = field(default_factory=dict)
    joints: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    bodies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    actuators: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "prefixes", dict(self.prefixes))
        object.__setattr__(self, "joints", {
            key: tuple(value) for key, value in self.joints.items()})
        object.__setattr__(self, "bodies", {
            key: tuple(value) for key, value in self.bodies.items()})
        object.__setattr__(self, "actuators", {
            key: tuple(value) for key, value in self.actuators.items()})

    # ------------------------------------------------------------------ #
    # articulation-scoped lookups
    # ------------------------------------------------------------------ #
    def prefix(self, articulation: str) -> str | None:
        """The backend prefix of *articulation* (``None`` for identity)."""
        return self.prefixes.get(articulation)

    def backend_name(self, articulation: str, unigym_joint: str) -> str:
        """The compiled joint name of one unigym joint."""
        prefix = self.prefixes.get(articulation)
        if prefix and not unigym_joint.startswith(prefix):
            return f"{prefix}{unigym_joint}"
        return unigym_joint

    def unigym_joint(self, articulation: str, backend_name: str) -> str:
        """The unigym joint name of one compiled joint."""
        return _strip(self.prefixes.get(articulation), backend_name)

    def unigym_joint_names(self, articulation: str) -> tuple[str, ...]:
        """Every unigym joint name of *articulation* (asset order)."""
        return tuple(self.joints.get(articulation, ()))

    def backend_joint_names(self, articulation: str) -> tuple[str, ...]:
        """Every compiled joint name of *articulation* (asset order)."""
        return tuple(self.backend_name(articulation, name)
                     for name in self.unigym_joint_names(articulation))

    def backend_body(self, articulation: str, unigym_body: str) -> str:
        """The compiled body name of one unigym body."""
        prefix = self.prefixes.get(articulation)
        if prefix and not unigym_body.startswith(prefix):
            return f"{prefix}{unigym_body}"
        return unigym_body

    def unigym_body(self, articulation: str, backend_name: str) -> str:
        """The unigym body name of one compiled body."""
        return _strip(self.prefixes.get(articulation), backend_name)

    def backend_actuator(self, articulation: str, unigym_name: str) -> str:
        """The compiled actuator name of one unigym actuator."""
        prefix = self.prefixes.get(articulation)
        if prefix and not unigym_name.startswith(prefix):
            return f"{prefix}{unigym_name}"
        return unigym_name

    # ------------------------------------------------------------------ #
    # whole-scene translation
    # ------------------------------------------------------------------ #
    def unigym_name(self, backend_name: str) -> str:
        """Strip the owning articulation prefix (identity when unowned)."""
        # longest prefix first: articulation names may be prefixes of one
        # another (``g1`` / ``g1b``).
        for prefix in sorted(
                (p for p in self.prefixes.values() if p),
                key=len, reverse=True):
            if backend_name.startswith(prefix):
                return backend_name[len(prefix):]
        return backend_name

    def is_articulation_name(self, backend_name: str) -> bool:
        """True when *backend_name* belongs to a composed articulation."""
        return any(
            backend_name in set(self.backend_joint_names(art))
            or backend_name in {
                self.backend_body(art, name)
                for name in self.bodies.get(art, ())}
            or backend_name in {
                self.backend_actuator(art, name)
                for name in self.actuators.get(art, ())}
            for art in self.prefixes)

    @classmethod
    def identity(cls) -> "MjNameMap":
        """The empty identity map (fused/single-asset historical path)."""
        return cls()


@dataclass(frozen=True)
class MjComposition:
    """One composed ``MjSpec`` plus its backend-private name map."""

    spec: object
    name_map: MjNameMap
    applied_prefixes: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "applied_prefixes",
                           dict(self.applied_prefixes))


@dataclass(frozen=True)
class MjControlBinding:
    """Backend-private binding of one control face (v105.3 §6.4).

    ``ControlRuntime.joint_names`` stays the neutral ``unigym_name`` tuple;
    this row carries the compiled names and addresses the step loop writes.
    """

    unigym_joint_names: tuple[str, ...]
    backend_joint_names: tuple[str, ...]
    jadr: tuple[int, ...] = ()
    jdr: tuple[int, ...] = ()
    act: tuple[int, ...] = ()
    perm_il2pol: tuple[int, ...] = ()

    def __post_init__(self):
        for key in ("unigym_joint_names", "backend_joint_names", "jadr",
                    "jdr", "act", "perm_il2pol"):
            object.__setattr__(self, key, tuple(getattr(self, key) or ()))


@dataclass(frozen=True)
class MjArticulationBinding:
    """Backend-private articulation binding (names + compiled ids)."""

    unigym_joint_names: tuple[str, ...]
    backend_joint_names: tuple[str, ...]
    unigym_base_body: str = ""
    backend_base_body: str = ""
    unigym_feet: tuple[str, ...] = ()
    backend_feet: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ("unigym_joint_names", "backend_joint_names",
                    "unigym_feet", "backend_feet"):
            object.__setattr__(self, key, tuple(getattr(self, key) or ()))
