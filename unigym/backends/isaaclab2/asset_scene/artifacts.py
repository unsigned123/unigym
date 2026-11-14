"""Deterministic artifact capture for an IsaacLab asset scene.

The neutral writers in :mod:`unigym.scenes.artifacts` own the file format;
this module only builds the pipeline views from the composed product and
hands them over.  It writes the five build artifacts an asset-scene run
records -- ``scene.json`` / ``resolution.json`` / ``payload_index.json`` /
``compose.txt`` / ``describe.json`` -- and nothing that needs Kit:

* ``compose.txt`` is the engine's ``to_xml`` face (canonical text; IsaacLab
  has no scene XML, and the JSON never lands in ``composed.xml``);
* ``describe.json`` is the canonical ``engine.describe`` mapping, so no
  resolved absolute path can enter it;
* the asset rows/entries in ``resolution.json`` and ``payload_index.json``
  may carry the composed ``resolved`` path (artifacts only).

Every write is a pure function of the inputs: sorted keys, two-space indent
and no timestamps, so two runs over the same scene are byte-identical.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from unigym._typing import JsonValue
from unigym.scenes.artifacts import (
    asset_instance_rows,
    write_composed_text,
    write_describe,
    write_payload_index,
    write_resolution,
    write_scene_json,
)
from unigym.scenes.pipeline import PayloadSet, Resolution
from unigym.scenes.scene_ir import Scene

from .engine import AssetSceneEngine, ComposedAssetScene
from .names import asset_field

__all__ = ["write_asset_scene_artifacts"]


def _formats_and_paths(
    scene: Scene,
    composed: ComposedAssetScene,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the selected format and resolved path of every asset instance.

    Args:
        scene: The composed neutral scene.
        composed: The composed IsaacLab product.

    Returns:
        ``(formats, resolved)`` keyed by declared asset-instance id.

    Raises:
        ValueError: A declared asset instance has no composed field.
    """
    by_field = {spec.field: spec for spec in composed.fields}
    formats: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for entry in scene.asset_instances:
        spec = by_field.get(asset_field(scene.name, entry.id))
        if spec is None:
            raise ValueError(
                f"asset scene {scene.name!r}: asset instance {entry.id!r} has no "
                f"composed field -- the artifact face and the composed product "
                f"disagree"
            )
        formats[entry.id] = spec.format
        resolved[entry.id] = spec.resolved_path
    return formats, resolved


def write_asset_scene_artifacts(
    out_dir: str | os.PathLike[str],
    scene: Scene,
    composed: ComposedAssetScene,
    engine: AssetSceneEngine,
) -> dict[str, str]:
    """Write the five build artifacts for one composed asset scene.

    Args:
        out_dir: Artifact directory (created when absent).
        scene: The neutral scene the engine composed.
        composed: The composed IsaacLab product.
        engine: The engine that built *composed* (``to_xml`` / ``describe``).

    Returns:
        A mapping of artifact name to written path.

    Raises:
        TypeError: *scene* is not a Scene.
        ValueError: *composed* is not a ComposedAssetScene.
    """
    if not isinstance(scene, Scene):
        raise TypeError(f"asset-scene artifacts expect a Scene, got {type(scene).__name__}")
    if not isinstance(composed, ComposedAssetScene):
        raise ValueError(
            f"asset-scene artifacts expect a ComposedAssetScene, got " f"{type(composed).__name__}"
        )
    formats, resolved = _formats_and_paths(scene, composed)
    rows: tuple[dict[str, JsonValue], ...] = asset_instance_rows(scene, formats)
    asset_rows: tuple[dict[str, JsonValue], ...] = tuple(
        {**row, "resolved": resolved[str(row["id"])]} for row in rows
    )
    resolution = Resolution(
        instances=tuple(
            {"id": item.id, "generator": item.generator, "source": item.source}
            for item in sorted(scene.instances, key=lambda entry: entry.id)
        ),
        regions=tuple(
            {"name": item.name, "kind": "region", "source": item.source}
            for item in sorted(scene.regions, key=lambda entry: entry.name)
        ),
        slots=tuple(
            {"id": item.id, "kind": "slot", "source": item.source}
            for item in sorted(scene.slots, key=lambda entry: entry.id)
        ),
        materials=tuple(
            {"name": item.name, "kind": "material"}
            for item in sorted(scene.materials, key=lambda entry: entry.name)
        ),
        asset_instances=asset_rows,
    )
    instance_entries: tuple[Mapping[str, JsonValue], ...] = tuple(
        {
            "id": item.id,
            "generator": item.generator,
            "source": item.source,
            "kind": item.role,
        }
        for item in sorted(scene.instances, key=lambda entry: entry.id)
    )
    asset_entries: tuple[Mapping[str, JsonValue], ...] = tuple(
        {
            "id": str(row["id"]),
            "kind": str(row["kind"]),
            "format": str(row["format"]),
            "asset": row["asset"],
            "source": str(row["source"]),
            "resolved": resolved[str(row["id"])],
        }
        for row in asset_rows
    )
    entries = instance_entries + asset_entries
    payloads = PayloadSet(entries=entries)
    return {
        "scene": write_scene_json(out_dir, scene),
        "resolution": write_resolution(out_dir, resolution),
        "payload_index": write_payload_index(out_dir, payloads),
        "compose": write_composed_text(out_dir, engine.to_xml(composed)),
        "describe": write_describe(out_dir, engine.describe(composed)),
    }
