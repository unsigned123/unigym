"""Deterministic scene artifact writers (neutral, standard library only).

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

The engine (S5) owns the compiled scene XML: it captures the composed text
before compilation and passes it to :func:`write_compile_report`, which
writes ``composed.xml`` next to ``compile.json``.
"""
from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping

__all__ = [
    "write_build_artifacts",
    "write_compile_report",
    "write_composed_xml",
    "write_draw",
    "write_payload_index",
    "write_resolution",
    "write_scene_json",
]


def _artifact_mapping(value):
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
        return to_mapping()
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(
        f"artifact value {type(value).__name__} is not a DTO with "
        f"to_mapping(), a mapping, or a dataclass -- pass the build-stage "
        f"view produced by the pipeline"
    )


def _write_json(out_dir, filename: str, payload) -> str:
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
    text = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False
    ) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def write_draw(out_dir, draw) -> str:
    """Write ``draw.json`` (what the pipeline decided to draw).

    Args:
        out_dir: Artifact directory.
        draw: A :class:`~unigym.scenes.pipeline.Draw` or equivalent
            mapping.

    Returns:
        The written path.
    """
    return _write_json(out_dir, "draw.json", _artifact_mapping(draw))


def write_resolution(out_dir, resolution) -> str:
    """Write ``resolution.json`` (the id-level view of the built scene).

    Args:
        out_dir: Artifact directory.
        resolution: A :class:`~unigym.scenes.pipeline.Resolution` or
            equivalent mapping.

    Returns:
        The written path.
    """
    return _write_json(
        out_dir, "resolution.json", _artifact_mapping(resolution)
    )


def write_scene_json(out_dir, scene) -> str:
    """Write ``scene.json`` (the neutral scene container).

    Args:
        out_dir: Artifact directory.
        scene: The neutral scene IR container or an equivalent mapping.

    Returns:
        The written path.
    """
    return _write_json(out_dir, "scene.json", _artifact_mapping(scene))


def write_payload_index(out_dir, payload_set) -> str:
    """Write ``payload_index.json`` (instance -> payload view).

    Args:
        out_dir: Artifact directory.
        payload_set: A :class:`~unigym.scenes.pipeline.PayloadSet` or
            equivalent mapping.

    Returns:
        The written path.
    """
    return _write_json(
        out_dir, "payload_index.json", _artifact_mapping(payload_set)
    )


def write_composed_xml(out_dir, composed_xml: str) -> str:
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
            f"composed scene text must be a string, got "
            f"{type(composed_xml).__name__}"
        )
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "composed.xml")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(composed_xml)
    return path


def write_compile_report(out_dir, compiled, composed_xml=None) -> str:
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
    return _write_json(
        out_dir, "compile.json", _artifact_mapping(compiled)
    )


def write_build_artifacts(out_dir, scene, draw, resolution, payloads) -> dict:
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
