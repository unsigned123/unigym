"""The experimental observation record of the simulation facade.

This is the minimal first slice: it carries the tick that produced the state,
the live state mapping, the exact step-result observation copy and the typed
projection tables.  The empty articulation/camera/contact mappings are the
honest placeholder for the typed projection slice; no typed field is claimed
here yet.

``torch`` is not imported by this module: every value is intentionally kept
as an opaque object until the typed projection owns its shape contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .handles import ArticulationHandle, CameraHandle

__all__ = ["Observation"]


@dataclass(frozen=True, eq=False)
class Observation:
    """One read-only view of the state produced by tick :attr:`t`.

    Attributes:
        t: the policy tick that produced the cached state (``0`` at spawn).
        raw: the live state mapping (the step ``extra`` without the
            ``reset_env_ids`` / ``should_reset`` bookkeeping keys).
        obs_cache: an exact copy of the producing ``StepResult.obs``.
        articulations: handle-keyed typed projections (empty in this slice).
        cameras: handle-keyed camera frames (empty in this slice).
        contacts: contact-group projections (empty in this slice).
    """

    t: int
    raw: Mapping[str, object]
    obs_cache: Mapping[str, object]
    articulations: Mapping[ArticulationHandle, object]
    cameras: Mapping[CameraHandle, object]
    contacts: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))
        object.__setattr__(self, "obs_cache", MappingProxyType(dict(self.obs_cache)))
        object.__setattr__(self, "articulations", MappingProxyType(dict(self.articulations)))
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))
        object.__setattr__(self, "contacts", MappingProxyType(dict(self.contacts)))
