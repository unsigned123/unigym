"""Pure marker policy for the layered pytest harness.

This module owns the directory-to-marker mapping and the derived ``fast`` lane
decision.  ``tests/conftest.py`` is a thin caller: it resolves an item's path
against the ``tests`` root, applies the markers returned here, and adds the
manual-marker-dependent ``backend_agnostic`` fallback.

The module imports only the standard library so the collection hook and the
marker-registry meta tests can use it without pulling in a heavy backend.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

LAYER_MARKERS: dict[str, str] = {
    "smoke": "layer_smoke",
    "unit": "layer_unit",
    "contract": "layer_contract",
    "conformance": "layer_conformance",
    "backends": "layer_backend",
    "integration": "layer_integration",
    "meta": "layer_meta",
}

UNIT_DOMAIN_MARKERS: dict[str, str] = {
    "core": "domain_core",
    "model": "domain_model",
    "scenes": "domain_scene",
    "runtime": "domain_runtime",
    "run": "domain_run",
    "contacts": "domain_contacts",
    "terminations": "domain_terminations",
    "filters": "domain_filters",
    "robots": "domain_robots",
    "actuators": "domain_actuators",
    "adapters": "domain_adapters",
    "assets": "domain_assets",
    "backends": "domain_backends",
}

BACKEND_MARKERS: dict[str, str] = {
    "mujoco": "backend_mujoco",
    "mjwarp": "backend_mjwarp",
    "isaaclab2": "backend_isaaclab2",
    "isaaclab3": "backend_isaaclab3",
}

RESOURCE_MARKERS: frozenset[str] = frozenset(
    {
        "slow",
        "gpu",
        "sim",
        "subprocess",
        "network",
        "serial",
        "requires_capabilities",
        "known_deviation",
        "requires_test_packs",
    }
)

FAST_LAYERS: frozenset[str] = frozenset(
    {"layer_smoke", "layer_unit", "layer_contract", "layer_meta"}
)

BACKEND_MARKER_PREFIX = "backend_"


class UnknownTestPathError(ValueError):
    """A collected test path does not fit the declared harness layout."""


def _path_text(parts: Sequence[str]) -> str:
    return "/".join(parts) if parts else "<empty>"


def layer_marker(parts: Sequence[str]) -> str:
    """Return the layer marker for a path relative to the ``tests`` root."""
    layer_dir = parts[0] if parts else ""
    if layer_dir not in LAYER_MARKERS:
        raise UnknownTestPathError(
            f"unknown test layer {layer_dir!r} in path {_path_text(parts)!r}; "
            f"known layers: {sorted(LAYER_MARKERS)}"
        )
    return LAYER_MARKERS[layer_dir]


def unit_domain_marker(parts: Sequence[str]) -> str | None:
    """Return the unit-domain marker, or ``None`` when not a unit test path."""
    if not parts or parts[0] != "unit":
        return None
    domain = parts[1] if len(parts) > 1 else ""
    if domain not in UNIT_DOMAIN_MARKERS:
        raise UnknownTestPathError(
            f"unknown unit domain {domain!r} in path {_path_text(parts)!r}; "
            f"known unit domains: {sorted(UNIT_DOMAIN_MARKERS)}"
        )
    return UNIT_DOMAIN_MARKERS[domain]


def backend_marker(parts: Sequence[str]) -> str | None:
    """Return the backend marker, or ``None`` when not a backend test path."""
    if not parts or parts[0] != "backends":
        return None
    backend = parts[1] if len(parts) > 1 else ""
    if backend not in BACKEND_MARKERS:
        raise UnknownTestPathError(
            f"unknown backend {backend!r} in path {_path_text(parts)!r}; "
            f"known backends: {sorted(BACKEND_MARKERS)}"
        )
    return BACKEND_MARKERS[backend]


def markers_for_path(parts: Sequence[str]) -> frozenset[str]:
    """Return every directory-derived marker for a path relative to ``tests``.

    Integration tests are always ``slow``; manual resource markers on other
    layers are added by test authors, not here.
    """
    layer = layer_marker(parts)
    markers = {layer}

    domain = unit_domain_marker(parts)
    if domain is not None:
        markers.add(domain)

    backend = backend_marker(parts)
    if backend is not None:
        markers.add(backend)

    if layer == LAYER_MARKERS["integration"]:
        markers.add("slow")

    return frozenset(markers)


def is_fast_lane(layer: str, marker_names: Collection[str]) -> bool:
    """Return whether ``layer`` is fast-eligible with no resource marker set."""
    return layer in FAST_LAYERS and RESOURCE_MARKERS.isdisjoint(marker_names)
