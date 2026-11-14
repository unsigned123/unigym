"""Deterministic scene artifact writers (neutral, stdlib-only dependencies).

The run-side half of the scene bridge: every build stage already produces
a JSON-ready view (:class:`~unigym.scenes.pipeline.Draw`,
:class:`~unigym.scenes.pipeline.Resolution`,
:class:`~unigym.scenes.pipeline.PayloadSet`,
:class:`~unigym.scenes.pipeline.CompiledScene`), and this module turns each
view into one file under the run's artifact directory.

Every file is a pure function of the view it is handed: sorted object keys,
two-space indent, UTF-8 with a trailing newline, no timestamps and no
absolute paths added by the writer.  Two calls with the same view are
byte-identical, so a run's artifacts can be compared directly.

:func:`asset_instance_rows` is the neutral, stdlib-only row shape both
engines put in their ``describe`` slice: hex floats, no resolved paths
(authored asset refs are carried verbatim) and no engine vocabulary.

The engine (S5) owns the compiled scene XML: it captures the composed text
before compilation and passes it to :func:`write_compile_report`, which
writes ``composed.xml`` next to ``compile.json``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import TypeGuard

from unigym._typing import JsonValue
from unigym.assets.asset_ref import ASSET_FORMATS

from .scene_ir import Scene

#: digest tag for the neutral file-asset scene slice both engines share.
ASSET_SCENE_DIGEST_VERSION = "unigym.scene.asset/1"

__all__ = [
    "ASSET_SCENE_DIGEST_VERSION",
    "asset_instance_rows",
    "write_build_artifacts",
    "write_compile_report",
    "write_composed_text",
    "write_composed_xml",
    "write_describe",
    "write_draw",
    "write_payload_index",
    "write_resolution",
    "write_scene_json",
]


def _is_json_mapping(value: object) -> TypeGuard[Mapping[str, JsonValue]]:
    """Narrow a runtime value to a JSON mapping (leaf shape trusted)."""
    return isinstance(value, Mapping)


def _artifact_mapping(value: object) -> Mapping[str, JsonValue]:
    """Return the JSON-ready mapping a writer should persist.

    Args:
        value: A DTO exposing ``to_mapping()``, a plain mapping, or a
            dataclass (the neutral scene IR container).

    Returns:
        A plain mapping.

    Raises:
        TypeError: The value carries no mapping view; the message names
            the accepted shapes.
    """
    to_mapping = getattr(value, "to_mapping", None)
    if callable(to_mapping):
        result = to_mapping()
        if not _is_json_mapping(result):
            raise TypeError(
                f"artifact value {type(value).__name__}.to_mapping() returned "
                f"{type(result).__name__}, expected a mapping"
            )
        return dict(result)
    if _is_json_mapping(value):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(
        f"artifact value {type(value).__name__} is not a DTO with "
        f"to_mapping(), a mapping, or a dataclass -- pass the build-stage "
        f"view produced by the pipeline"
    )


def _write_json(
    out_dir: str | os.PathLike[str],
    filename: str,
    payload: Mapping[str, JsonValue],
) -> str:
    """Write one deterministic JSON file and return its path.

    Args:
        out_dir: Artifact directory (created when absent).
        filename: Leaf filename inside *out_dir*.
        payload: JSON-ready mapping.

    Returns:
        The written path.
    """
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def asset_instance_rows(
    scene: Scene,
    formats: Mapping[str, str],
) -> tuple[dict[str, JsonValue], ...]:
    """Return the deterministic neutral asset rows shared by both engines.

    Args:
        scene: The scene whose file-backed asset instances are described.
        formats: The engine-selected format per asset-instance id.

    Returns:
        One row per asset instance, sorted by id, with hex-float poses,
        no resolved paths, and the declared asset refs carried verbatim.

    Raises:
        TypeError: *scene* is not a Scene or *formats* is not a mapping.
        ValueError: An id has no selected format or a format the asset does
            not declare.
    """
    if not isinstance(scene, Scene):
        raise TypeError(f"asset_instance_rows expects a Scene, got {type(scene).__name__}")
    if not isinstance(formats, Mapping):
        raise TypeError(
            f"asset_instance_rows: formats must be a mapping of asset-instance id "
            f"to format, got {type(formats).__name__}"
        )
    declared_ids = sorted(entry.id for entry in scene.asset_instances)
    unknown = sorted(set(formats) - set(declared_ids))
    if unknown:
        raise ValueError(
            f"asset_instance_rows: formats names undeclared asset instance(s) "
            f"{unknown}; declared ids are {declared_ids}"
        )
    rows: list[dict[str, JsonValue]] = []
    for entry in sorted(scene.asset_instances, key=lambda item: item.id):
        selected = formats.get(entry.id)
        if selected is None:
            raise ValueError(
                f"asset instance {entry.id!r}: no format was selected -- declare "
                f"exactly one of {list(ASSET_FORMATS)} in formats"
            )
        if selected not in ASSET_FORMATS:
            raise ValueError(
                f"asset instance {entry.id!r}: format {selected!r} must be one of "
                f"{list(ASSET_FORMATS)}"
            )
        if not getattr(entry.asset, selected):
            declared = sorted(fmt for fmt in ASSET_FORMATS if getattr(entry.asset, fmt))
            raise ValueError(
                f"asset instance {entry.id!r}: asset declares format(s) "
                f"{declared} but format {selected!r} was selected -- no "
                f"compatible format"
            )
        rows.append(
            {
                "id": entry.id,
                "kind": entry.kind,
                "format": selected,
                "asset": {key: value for key, value in entry.asset.to_dict().items()},
                "pose": {
                    "xyz": [value.hex() for value in entry.pose.xyz],
                    "quat_wxyz": [value.hex() for value in entry.pose.quat_wxyz],
                },
                "collision": entry.collision,
                "source": entry.source,
            }
        )
    return tuple(rows)


def write_draw(out_dir: str | os.PathLike[str], draw: object) -> str:
    """Write ``draw.json`` (what the pipeline decided to draw).

    Args:
        out_dir: Artifact directory.
        draw: A :class:`~unigym.scenes.pipeline.Draw` or equivalent
            mapping.

    Returns:
        The written path.
    """
    return _write_json(out_dir, "draw.json", _artifact_mapping(draw))


def write_resolution(out_dir: str | os.PathLike[str], resolution: object) -> str:
    """Write ``resolution.json`` (the id-level view of the built scene).

    Args:
        out_dir: Artifact directory.
        resolution: A :class:`~unigym.scenes.pipeline.Resolution` or
            equivalent mapping.

    Returns:
        The written path.
    """
    return _write_json(out_dir, "resolution.json", _artifact_mapping(resolution))


def write_scene_json(out_dir: str | os.PathLike[str], scene: object) -> str:
    """Write ``scene.json`` (the neutral scene container).

    Args:
        out_dir: Artifact directory.
        scene: The neutral scene IR container or an equivalent mapping.

    Returns:
        The written path.
    """
    return _write_json(out_dir, "scene.json", _artifact_mapping(scene))


def write_payload_index(out_dir: str | os.PathLike[str], payload_set: object) -> str:
    """Write ``payload_index.json`` (instance -> payload view).

    Args:
        out_dir: Artifact directory.
        payload_set: A :class:`~unigym.scenes.pipeline.PayloadSet` or
            equivalent mapping.

    Returns:
        The written path.
    """
    return _write_json(out_dir, "payload_index.json", _artifact_mapping(payload_set))


def write_composed_xml(out_dir: str | os.PathLike[str], composed_xml: str) -> str:
    """Write ``composed.xml`` (the pre-compile composed scene text).

    Args:
        out_dir: Artifact directory.
        composed_xml: The composed XML text captured before compilation.

    Returns:
        The written path.

    Raises:
        TypeError: *composed_xml* is not text.
    """
    if not isinstance(composed_xml, str):
        raise TypeError(
            f"composed scene text must be a string, got " f"{type(composed_xml).__name__}"
        )
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "composed.xml")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(composed_xml)
    return path


def write_composed_text(out_dir: str | os.PathLike[str], text: str) -> str:
    """Write ``compose.txt`` (the composed product the engine serialized).

    IsaacLab has no scene XML; the engine's ``to_xml`` face returns canonical
    text for artifact capture and A/B comparison.  The neutral writer keeps
    that text out of ``composed.xml`` so the two artifact names never alias.

    Args:
        out_dir: Artifact directory.
        text: The composed text produced by the engine.

    Returns:
        The written path.

    Raises:
        TypeError: *text* is not text.
    """
    if not isinstance(text, str):
        raise TypeError(f"composed scene text must be a string, got {type(text).__name__}")
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "compose.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def write_describe(out_dir: str | os.PathLike[str], describe: object) -> str:
    """Write ``describe.json`` (the engine's neutral description mapping).

    Args:
        out_dir: Artifact directory.
        describe: The mapping returned by an engine's ``describe`` face (a
            DTO, a plain mapping, or a dataclass).

    Returns:
        The written path.
    """
    return _write_json(out_dir, "describe.json", _artifact_mapping(describe))


def write_compile_report(
    out_dir: str | os.PathLike[str],
    compiled: object,
    composed_xml: str | None = None,
) -> str:
    """Write ``compile.json`` and, when given, ``composed.xml``.

    Args:
        out_dir: Artifact directory.
        compiled: A :class:`~unigym.scenes.pipeline.CompiledScene` (or
            equivalent mapping) carrying the post-compile facts.
        composed_xml: Optional pre-compile composed scene text; when
            present it is written beside the report.

    Returns:
        The written ``compile.json`` path.
    """
    if composed_xml is not None:
        write_composed_xml(out_dir, composed_xml)
    return _write_json(out_dir, "compile.json", _artifact_mapping(compiled))


def write_build_artifacts(
    out_dir: str | os.PathLike[str],
    scene: object,
    draw: object,
    resolution: object,
    payloads: object,
) -> dict[str, str]:
    """Write the four build-stage artifacts in one call.

    Args:
        out_dir: Artifact directory.
        scene: The neutral scene IR container.
        draw: The pipeline draw view.
        resolution: The pipeline resolution view.
        payloads: The pipeline payload-set view.

    Returns:
        A mapping of artifact name to written path.
    """
    return {
        "draw": write_draw(out_dir, draw),
        "resolution": write_resolution(out_dir, resolution),
        "scene": write_scene_json(out_dir, scene),
        "payload_index": write_payload_index(out_dir, payloads),
    }
