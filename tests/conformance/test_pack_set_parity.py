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

from unigym.packs import Pack, PackSet, discover_set, installed, require
from unigym.provenance import ProviderOrigin

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"
ROOTS = (CONTENT_ROOT, CODE_ROOT)


def _pack_projection(pack: Pack) -> tuple[object, ...]:
    """Every observable pack fact except the discovery label."""
    return (
        pack.name,
        pack.version,
        pack.api,
        pack.trust,
        str(pack.root),
        tuple((face.value, count) for face, count in pack.face_counts.items()),
        tuple(
            (item.face.value, item.name, item.kind, item.resource, item.entrypoint)
            for item in pack.items
        ),
    )


def _set_projection(view: PackSet) -> tuple[object, ...]:
    """The ordered per-pack projection of one pack set."""
    return tuple(_pack_projection(pack) for pack in view.packs())


def _cross_projection(view: PackSet) -> tuple[object, ...]:
    """The cross-pack aggregate item and origin views of one pack set."""
    return (
        tuple(
            (item.pack, item.face.value, item.name, item.kind, item.resource, item.entrypoint)
            for item in view.items()
        ),
        tuple((origin.kind, origin.pack) for origin in view.origins()),
        tuple((item.pack, item.name) for item in view.providers()),
        tuple((item.pack, item.name) for item in view.resources()),
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
