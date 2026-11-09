"""The neutral per-articulation model set (design v103.4).

A ``ModelSet`` groups one :class:`~unigym.model.ir.ModelIR` per policy
articulation (keyed by articulation name) plus an optional whole-scene
``merged`` IR.  There is **no first/primary articulation**: ``merged`` is a
whole-scene view, never "the first binding".  A backend that cannot merge
several assets losslessly (e.g. isaaclab2 with duplicate bare body names)
reports ``merged=None`` and callers read a named per-articulation IR.

The convenience table proxies (``joints`` / ``bodies`` / ...) read the
``merged`` IR; when it is absent they raise :class:`ModelSetError` naming
the per-articulation alternative instead of silently choosing one.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ModelSetError

if TYPE_CHECKING:
    from .ir import Actuator, Body, Camera, Geom, Joint, Material, ModelIR, Site
    from .table import Table

__all__ = ["ModelSet"]

_TABLE_FACES = (
    "bodies",
    "joints",
    "geoms",
    "sites",
    "cameras",
    "actuators",
    "materials",
)


@dataclass(frozen=True)
class ModelSet:
    """Per-articulation IRs plus an optional whole-scene merge.

    Attributes:
        articulations: ``{articulation name: ModelIR}``.  Every entry is an
            equal citizen; lookup/iteration never implies a primary.
        merged: The whole compiled IR, or ``None`` when no honest lossless
            merge exists (multi-asset backends with colliding bare names).
    """

    articulations: Mapping[str, ModelIR]
    merged: ModelIR | None = None

    def __post_init__(self) -> None:
        """Freeze the articulation mapping into a plain dictionary."""
        object.__setattr__(self, "articulations", dict(self.articulations))

    def __getitem__(self, name: str) -> ModelIR:
        """Return one articulation's IR by name (``KeyError`` when absent)."""
        return self.articulations[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate articulation names in declaration order."""
        return iter(self.articulations)

    def __len__(self) -> int:
        """Return the number of articulations."""
        return len(self.articulations)

    def __contains__(self, name: object) -> bool:
        """Membership test for an articulation name."""
        return name in self.articulations

    def names(self) -> list[str]:
        """Return every articulation name in declaration order."""
        return list(self.articulations)

    def _require_merged(self, face: str) -> ModelIR:
        """Return the merged IR or raise honestly when there is no merge."""
        if self.merged is None:
            names = ", ".join(self.articulations) or "<none>"
            raise ModelSetError(
                f"ModelSet.{face}: this set has no merged whole-scene IR "
                f"(the backend cannot merge its articulations losslessly); "
                f"read the per-articulation IR instead: "
                f"model.articulations[name].{face} for name in "
                f"[{names}]")
        return self.merged

    @property
    def bodies(self) -> Table[Body]:
        """The merged body table (per-articulation alternative when absent)."""
        return self._require_merged("bodies").bodies

    @property
    def joints(self) -> Table[Joint]:
        """The merged joint table (per-articulation alternative when absent)."""
        return self._require_merged("joints").joints

    @property
    def geoms(self) -> Table[Geom]:
        """The merged geom table (per-articulation alternative when absent)."""
        return self._require_merged("geoms").geoms

    @property
    def sites(self) -> Table[Site]:
        """The merged site table (per-articulation alternative when absent)."""
        return self._require_merged("sites").sites

    @property
    def cameras(self) -> Table[Camera]:
        """The merged camera table (per-articulation alternative when absent)."""
        return self._require_merged("cameras").cameras

    @property
    def actuators(self) -> Table[Actuator]:
        """The merged actuator table (per-articulation alternative)."""
        return self._require_merged("actuators").actuators

    @property
    def materials(self) -> Table[Material]:
        """The merged material table (per-articulation alternative)."""
        return self._require_merged("materials").materials

    @property
    def meta(self) -> Mapping[str, object]:
        """The merged IR meta (per-articulation alternative when absent)."""
        return self._require_merged("meta").meta

    def __repr__(self) -> str:
        """``ModelSet(names=[...], merged=<bool>)``."""
        merged = "none" if self.merged is None else repr(self.merged.name)
        return f"ModelSet(names={self.names()!r}, merged={merged})"
