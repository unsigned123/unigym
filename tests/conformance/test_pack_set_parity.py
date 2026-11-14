"""Conformance: installed discovery and explicit roots yield the same pack set.

``unigym.packs.installed()`` and ``unigym.packs.discover_set(roots)`` are one
discovery rule with two entry points.  For the same roots they must yield the
same packs, names, counts, handles and resolved resource paths; only the
``discovered_by`` label differs (configured source versus ``explicit-root``).
``require(*names)`` is the filtered subset of the installed view.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from unigym.packs import PackInfo, PackSet, discover_set, installed, require
from unigym.provenance import ProviderOrigin

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"
ROOTS = (CONTENT_ROOT, CODE_ROOT)


def _pack_projection(pack: PackInfo) -> tuple[object, ...]:
    """Every observable pack fact except the discovery label."""
    return (
        pack.name,
        pack.version,
        pack.api,
        pack.trust,
        str(pack.root),
        tuple(pack.face_counts.items()),
        tuple((ref.id, ref.kind, ref.resource, ref.entrypoint) for ref in pack.robots()),
        tuple((ref.id, ref.kind, ref.resource) for ref in pack.assets()),
        tuple((ref.id, ref.resource) for ref in pack.scene_presets()),
        tuple(
            (ref.id, ref.entrypoint, _schema_projection(ref.params_schema))
            for ref in pack.scene_providers()
        ),
        tuple(
            (info.name, info.kind, info.face, info.resource, info.entrypoint, info.active)
            for info in pack.providers()
        ),
        tuple(
            (info.face, info.id, info.kind, info.resource, str(info.path))
            for info in pack.resources()
        ),
    )


def _schema_projection(schema) -> object:
    """Normalize an optional provider parameter schema for comparison."""
    if schema is None:
        return None
    return tuple((name, repr(spec)) for name, spec in schema.items())


def _set_projection(view: PackSet) -> tuple[object, ...]:
    """The ordered per-pack projection of one pack set."""
    return tuple(_pack_projection(pack) for pack in view.packs())


def _cross_projection(view: PackSet) -> tuple[object, ...]:
    """The cross-pack aggregate views of one pack set."""
    return (
        tuple((ref.pack, ref.id, ref.resource) for ref in view.robots()),
        tuple((ref.pack, ref.id, ref.resource) for ref in view.scene_presets()),
        tuple((ref.pack, ref.id, ref.entrypoint) for ref in view.scene_providers()),
        tuple((ref.pack, ref.id, ref.resource) for ref in view.assets()),
        tuple((info.name, info.kind, info.pack, info.face) for info in view.providers()),
        tuple((info.face, info.id, str(info.path)) for info in view.resources()),
    )


def _without_discovery_label(origin: ProviderOrigin) -> ProviderOrigin:
    """Drop the one intended difference between the two discovery entry points."""
    return replace(origin, discovered_by=None, location_imported=False)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point configured discovery at the shared fixture packs only."""
    monkeypatch.setenv("UNIGYM_PACKS", os.pathsep.join(str(root) for root in ROOTS))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", "")


def test_installed_matches_explicit_discover_set(configured: None) -> None:
    """Configured discovery and explicit roots agree on every catalog fact."""
    configured_view = installed()
    explicit_view = discover_set(ROOTS)

    assert [pack.name for pack in configured_view.packs()] == ["acme_content", "acme_code"]
    assert [pack.name for pack in explicit_view.packs()] == ["acme_content", "acme_code"]
    assert _set_projection(configured_view) == _set_projection(explicit_view)
    assert _cross_projection(configured_view) == _cross_projection(explicit_view)

    assert [origin.discovered_by for origin in configured_view.origins()] == [
        "env:UNIGYM_PACKS",
        "env:UNIGYM_PACKS",
    ]
    assert [origin.discovered_by for origin in explicit_view.origins()] == [
        "explicit-root",
        "explicit-root",
    ]
    assert [_without_discovery_label(origin) for origin in configured_view.origins()] == [
        _without_discovery_label(origin) for origin in explicit_view.origins()
    ]


def test_require_is_the_filtered_installed_subset(configured: None) -> None:
    """``require`` keeps exactly the named packs in discovery order."""
    whole = installed()

    selected = require("acme_content")
    assert [pack.name for pack in selected.packs()] == ["acme_content"]
    assert _set_projection(selected) == (_pack_projection(whole.pack("acme_content")),)
    assert _cross_projection(selected) == _cross_projection(discover_set([CONTENT_ROOT]))

    reversed_names = require("acme_code", "acme_content")
    assert [pack.name for pack in reversed_names.packs()] == ["acme_content", "acme_code"]
    assert _set_projection(reversed_names) == _set_projection(whole)
