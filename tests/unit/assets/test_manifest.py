"""L0 asset manifests reject L1 scene vocabulary at any depth (R3)."""

from __future__ import annotations

import pytest

from unigym.assets.manifest import ManifestError, validate_manifest


def _manifest(**extra: object) -> dict:
    """One minimal valid ``unigym.asset/1`` manifest plus *extra* keys."""
    doc: dict = {
        "schema": "unigym.asset/1",
        "id": "pack:acme/objects/cup",
        "kind": "model",
        "files": {"model": "cup.xml"},
        "capabilities": {},
        "license": "CC-BY-4.0",
        "provenance": {"origin": "registry acme"},
    }
    doc.update(extra)
    return doc


def test_minimal_manifest_validates() -> None:
    """The fixture itself is a valid manifest (the negative cases are real)."""
    assert validate_manifest(_manifest())["id"] == "pack:acme/objects/cup"


@pytest.mark.parametrize("key", ["scene_preset", "scene_provider"])
def test_asset_manifest_rejects_forbidden_scene_keys(key: str) -> None:
    """A top-level preset/provider key smuggles L1 scene knowledge into L0."""
    with pytest.raises(ManifestError, match="forbidden at any depth"):
        validate_manifest(_manifest(**{key: "acme.tabletop"}))


@pytest.mark.parametrize("key", ["scene_preset", "scene_provider"])
def test_asset_manifest_rejects_forbidden_scene_keys_at_any_depth(key: str) -> None:
    """The forbidden-key scan is recursive, not top-level only."""
    doc = _manifest()
    doc["capabilities"] = {key: "acme.tabletop"}

    with pytest.raises(ManifestError, match=f"key '{key}'"):
        validate_manifest(doc)
