"""Live projection of one pack's currently registered providers.

This module is a read-only diagnostic view over the process-wide registries,
not a second pack model.  Declared items live in ``Pack.items``; the records
returned here are live :class:`~unigym.provenance.ProviderRef` values whose
origins were recorded by activation.  Nothing here activates a pack, mutates a
registry or synthesizes a :class:`~unigym.packs.schema.PackItem` for a dynamic
registration.

Registry imports are lazy so importing this helper never pulls a registry or
its optional dependencies into the base package import path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from unigym.provenance import ProviderRef

if TYPE_CHECKING:
    from .schema import Pack

_REGISTRY_ORDER: tuple[str, ...] = (
    "adapters",
    "robots",
    "scene_providers",
    "scene_content_fillers",
    "prop_templates",
    "terminations",
    "initializers",
    "actuator_laws",
)
"""Canonical render order shared by the text and JSON inventory views."""

_ATTRIBUTED_KINDS = frozenset({"manifest", "pack-code"})
"""Origin kinds that name a pack; ``core``/``inline``/``unknown`` do not."""


def runtime_registrations(pack: Pack) -> Mapping[str, tuple[ProviderRef, ...]]:
    """Return this pack's live registry records grouped by registry label.

    Only records whose origin names *pack* with kind ``manifest`` or
    ``pack-code`` are included.  Core, inline, unknown and other packs'
    registrations are excluded.  Registries with no matching record are
    omitted; the remaining keys follow :data:`_REGISTRY_ORDER`.

    The returned mapping (and its tuples) are read-only; querying it never
    mutates a registry.
    """
    from unigym.actuator_law import actuator_law_providers
    from unigym.adapters import adapter_providers
    from unigym.initializers.base import initializer_providers
    from unigym.props.templates import prop_template_providers
    from unigym.robots import robot_providers
    from unigym.scenes.pipeline import scene_content_filler_providers
    from unigym.scenes.providers import scene_provider_providers
    from unigym.terminations import termination_providers

    queries: Mapping[str, Callable[[], tuple[ProviderRef, ...]]] = {
        "adapters": adapter_providers,
        "robots": robot_providers,
        "scene_providers": scene_provider_providers,
        "scene_content_fillers": scene_content_filler_providers,
        "prop_templates": prop_template_providers,
        "terminations": termination_providers,
        "initializers": initializer_providers,
        "actuator_laws": actuator_law_providers,
    }

    projected: dict[str, tuple[ProviderRef, ...]] = {}
    for label in _REGISTRY_ORDER:
        records = tuple(
            ref
            for ref in queries[label]()
            if ref.origin.pack == pack.name and ref.origin.kind in _ATTRIBUTED_KINDS
        )
        if records:
            projected[label] = records
    return MappingProxyType(projected)


__all__ = ["runtime_registrations"]
