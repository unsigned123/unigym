"""Scene artifact capture for the MuJoCo backend.

A run that composes a scene records six build artifacts: ``draw.json`` /
``resolution.json`` / ``scene.json`` / ``payload_index.json`` /
``composed.xml`` / ``compile.json``.  This module owns the capture half --
the pipeline DTO views, the pre-compile composed XML text and the
post-compile fact report -- so ``env.py`` only calls in at the two
construction points it already has (before and after ``MjSpec.compile``).

The writers themselves live in :mod:`unigym.scenes.artifacts` and stay
DTO-only; every function here is a pure transformation of the inputs it is
handed (the model/``MjSpec`` are read, never mutated).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

from unigym.scenes.artifacts import write_build_artifacts, write_compile_report
from unigym.scenes.pipeline import CompiledScene, Draw, PayloadSet, Resolution

__all__ = [
    "build_views",
    "capture_composed_xml",
    "compile_report",
    "write_fill_record",
    "write_scene_artifacts",
]


def _scene_generator_exists(name: str) -> bool:
    """Whether *name* resolves in the core generator or prop-template registry.

    The scene composer dispatches primitive generators through
    :mod:`unigym.scenes.generators` and asset instances through the
    prop-template registry, so the binding check must cover both.
    """
    from unigym.props.templates import prop_template_metadata
    from unigym.scenes.generators import registered_generators

    if name in registered_generators():
        return True
    try:
        prop_template_metadata(name)
    except ValueError:
        return False
    return True


def _require_content_scene(spec, scene_name) -> None:
    """Fail closed when a content spec has no composed scene to fill."""
    if spec is not None and scene_name is None:
        raise ValueError(
            "mujoco backend: env.params.content needs a composed scene "
            "(env.params.mj_scene) to fill -- declare the scene or "
            "drop the content spec"
        )


def _fill_scene_content(scene, spec, asset_plan):
    """Fill *scene* with the declared content spec and merge the instances.

    The catalog is loaded from the provider factory's declared
    ``catalog_ref`` (shipped with the pack; never rebuilt per run) and resolved
    through the run's explicit :class:`~unigym.assets.plan.AssetPlan`.  The
    build-time fill uses ``spec.seed`` (default 0); a per-episode refill is
    reserved for the reset stage and is not wired here yet.

    Args:
        scene: The built neutral scene the provider fills.
        spec: The content spec, or ``None`` (the scene passes through).
        asset_plan: The composition boundary that resolves the provider's
            ``catalog_ref``; ``None`` when the run declared no pack runtime.

    Returns:
        The scene with the provider's instances appended, plus the fill
        record (``None`` when no spec was declared or the provider
        declined to record).

    Raises:
        ValueError: The spec is not a mapping, the provider is unknown or
            carries no ``catalog_ref``, no asset plan is available, or a
            binding rule fails.
    """
    from dataclasses import replace

    from unigym.assets.catalog import load_catalog
    from unigym.scenes.pipeline import content_provider, validate_binding

    if spec is None:
        return scene, None
    if not isinstance(spec, Mapping):
        raise ValueError(f"content spec must be a mapping, got {type(spec).__name__}")
    name = spec.get("provider", "robocasa")
    factory = content_provider(name)
    catalog_ref = getattr(factory, "catalog_ref", None)
    if not isinstance(catalog_ref, str) or not catalog_ref:
        raise ValueError(
            f"content provider {name!r} declares no catalog_ref -- the "
            f"runtime loads the shipped catalog from the pack"
        )
    if asset_plan is None:
        raise ValueError(
            f"content provider {name!r} catalog_ref {catalog_ref!r} needs an "
            f"explicit AssetPlan; build the run with a pack runtime"
        )
    provider = factory(
        catalog=load_catalog(asset_plan.resolve(catalog_ref)),
        generator_exists=_scene_generator_exists,
    )
    fill_spec = dict(spec)
    fill_spec.pop("provider", None)
    if fill_spec.get("seed") is None:
        fill_spec["seed"] = 0
    added = provider.fill(scene, fill_spec, random.Random(fill_spec["seed"]))
    merged = replace(scene, instances=scene.instances + tuple(added))
    validate_binding(merged, _scene_generator_exists)
    return merged, provider.last_record


def _views_for_scene(scene, pipeline: str | None, slots):
    """Build the draw/resolution/payload views for one neutral scene."""
    params = dict(slots or {})
    draw = Draw(
        pipeline=str(pipeline),
        seed=params.get("seed"),
        recipe_path="",
        params=params,
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
    )
    payloads = PayloadSet(
        entries=tuple(
            {
                "id": item.id,
                "generator": item.generator,
                "source": item.source,
                "kind": item.role,
            }
            for item in sorted(scene.instances, key=lambda entry: entry.id)
        )
    )
    return draw, resolution, payloads


def build_views(scene, pipeline: str, slots):
    """Build the pipeline's draw/resolution/payload views for one scene.

    Args:
        scene: The validated :class:`~unigym.scenes.scene_ir.Scene` the
            env composed.
        pipeline: The registered scene-provider name (``env.params.mj_scene``).
        slots: The provider slot dict (``env.params.mj_scene_params``).

    Returns:
        The ``(Draw, Resolution, PayloadSet)`` trio the writers consume.
    """
    return _views_for_scene(scene, pipeline, slots)


def capture_composed_xml(mjspec, out_dir=None, scene=None):
    """Return the pre-compile composed scene text of *mjspec*.

    Args:
        mjspec: The ``MjSpec`` captured before the camera/prop/director
            layers join it.
        out_dir: Artifact directory; ``None`` means the artifact face is
            off and no text is captured.
        scene: The composed scene; ``None`` (no composed scene) also
            turns the capture off.

    Returns:
        The composed XML text, or ``None`` when the artifact face is off.
    """
    if out_dir is None or scene is None:
        return None
    return mjspec.to_xml()


def compile_report(model) -> CompiledScene:
    """Build the post-compile structural fact view of *model*."""
    facts = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "nsite": int(model.nsite),
        "timestep": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "cone": int(model.opt.cone),
        "iterations": int(model.opt.iterations),
        "impratio": float(model.opt.impratio),
        "noslip_iterations": int(model.opt.noslip_iterations),
    }
    return CompiledScene(facts)


def write_fill_record(out_dir, record) -> str:
    """Write ``fill.json`` from one content fill record.

    Args:
        out_dir: Artifact directory.
        record: A record exposing ``write(path)`` (the content provider's
            ``FillRecord``).

    Returns:
        The written path (as text).
    """
    return str(record.write(Path(out_dir) / "fill.json"))


def write_scene_artifacts(
    out_dir, scene, pipeline: str | None, model, slots, composed_xml=None, fill_record=None
) -> dict:
    """Write the six build artifacts (plus ``fill.json`` when content ran).

    Args:
        out_dir: Artifact directory (``None`` disables the write).
        scene: The validated :class:`~unigym.scenes.scene_ir.Scene` the
            env composed.
        pipeline: The registered scene-provider name.
        model: The compiled ``MjModel`` carrying the structural facts.
        slots: The provider slot dict.
        composed_xml: The pre-compile composed scene text; ``None`` means
            the artifact face is off and nothing is written.
        fill_record: The content fill record; ``None`` (no content spec)
            leaves ``fill.json`` absent.

    Returns:
        A mapping of artifact name to written path (empty when off).
    """
    if composed_xml is None or out_dir is None:
        return {}
    draw, resolution, payloads = _views_for_scene(scene, pipeline, slots)
    paths = write_build_artifacts(out_dir, scene, draw, resolution, payloads)
    paths["compile"] = write_compile_report(
        out_dir, compile_report(model), composed_xml=composed_xml
    )
    if fill_record is not None:
        paths["fill"] = write_fill_record(out_dir, fill_record)
    return paths
