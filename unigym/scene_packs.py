"""Scene bundle resolution through the pack runtime ``scenes`` face.

A scene bundle answers "what is this world": the MJ model, the base joint
semantics, the dynamic props and the reset initializer.  A run config
references one by its exact ``scenes`` catalog id (``scene:
acme.tabletop``); :func:`apply_scene_bundle` resolves that id through
:meth:`unigym.packs.runtime.PackRuntime.resource_for`, parses the declared
YAML document and routes the bundle's blocks into the run config's existing
consumption points BEFORE placeholder expansion (the pack's
``${UNIGYM_REPO}`` tokens ride the same single expansion point as the run
yaml).  Actuation is not a scene-bundle block: the only physical gain source
is the run-level ``ArticulationSpec.actuation`` IR.

=======================  =============================================
``backend.mujoco``       routed: ``world_base`` -> the run config TOP
                         LEVEL (the world declaration); the rest
                         (``mj_scene``, ``mj_scene_params``, ``base``) ->
                         ``env.params`` -- backend
                         content lives under ``backend: <name>:`` blocks
                         (per-backend hole; ``mj_scene`` names a REGISTERED
                         scene definition the mujoco backend composes +
                         attaches into the ``world_base.mjcf`` base spec;
                         ``mj_scene_params`` rides the provider
                         dispatch verbatim)
``objects``              the run-level content block (run wins); the
                         internal name stays ``props``
``initializer``          routed into ``env.params.initializer``
=======================  =============================================

Scene refs are exact catalog ids, never directory scans: a reference without
a runtime is a teaching error and an unknown id fails with the catalog's
known-id list.  Merge precedence is the three-layer rule (scene defaults ->
task overrides -> run overrides): a run-level value always wins; mappings
merge key-wise (so a run can override a single initializer param, e.g. the
per-seed stream seed, without restating the table).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import yaml

from unigym._typing import JsonValue, is_json_object
from unigym.scenes.recipe import normalize_content_keys

if TYPE_CHECKING:
    from unigym.packs.runtime import PackRuntime


def load_scene_bundle(
    ref: str,
    runtime: PackRuntime,
) -> dict[str, JsonValue]:
    """Resolve one exact ``scenes`` catalog id to its parsed bundle.

    Args:
        ref: The catalog id declared by the run config's ``scene:`` key.
        runtime: The pack runtime owning the ``scenes`` catalog.

    Returns:
        The parsed YAML document (a string-keyed mapping).

    Raises:
        ValueError: The ref is not a non-empty string, or the declared
            resource is not valid YAML or not a mapping.
        unigym.packs.PackError: The id is unknown on the ``scenes`` face.
    """
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"scene ref must be a non-empty pack catalog id string, got {ref!r}")
    if runtime is None:
        raise ValueError(
            f"scene ref {ref!r} requires a pack runtime: pass runtime=... | "
            f"hint: scene refs are exact PackRuntime 'scenes' catalog ids"
        )
    path = runtime.resource_for("scenes", ref)
    try:
        raw_doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ValueError(f"scene resource {ref!r} at {path} is not readable: {err}") from err
    except yaml.YAMLError as err:
        raise ValueError(f"scene resource {ref!r} at {path} is not valid YAML: {err}") from err
    doc: object = {} if raw_doc is None else raw_doc
    if not is_json_object(doc):
        raise ValueError(
            f"scene resource {ref!r} at {path} must be a mapping, got {type(doc).__name__}"
        )
    return doc


def _merge(scene, run):
    """Three-layer merge primitive: run wins; mappings merge key-wise.

    A ``None`` run value counts as absent (the run layer simply did not
    declare it) -- the scene default stands.
    """
    if run is None:
        return scene
    if isinstance(scene, dict) and isinstance(run, dict):
        out = dict(scene)
        for key, val in run.items():
            out[key] = _merge(scene[key], val) if key in scene else val
        return out
    return run


def apply_scene_bundle(
    cfg: dict[str, JsonValue],
    runtime: PackRuntime | None = None,
) -> dict[str, JsonValue]:
    """Resolve the run config's ``scene:`` catalog id (deep-copy merge).

    Returns the config with the bundle's blocks routed in and the consumed
    ``scene:`` ref removed; configs without a ``scene:`` key pass through
    untouched and need no runtime.  Run-level values win at every layer;
    mapping sections (``initializer.params``, ``base``) merge key-wise so a
    run can carry a single param override (e.g. the per-seed stream seed)
    without restating the table.
    Only ``backend.mujoco`` blocks route this phase; other backend blocks are
    left for their backends.
    """
    ref = cfg.get("scene")
    if ref is None:
        return cfg
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"scene ref must be a non-empty pack catalog id string, got {ref!r}")
    if runtime is None:
        raise ValueError(
            f"scene ref {ref!r} requires a pack runtime: pass runtime=... | "
            f"hint: scene refs are exact PackRuntime 'scenes' catalog ids"
        )
    doc = normalize_content_keys(load_scene_bundle(ref, runtime), source=f"scene bundle {ref!r}")
    cfg = copy.deepcopy(cfg)
    # A direct caller (tests, the API) may hand us an un-normalized run
    # config; the collector's single load point already migrated its own
    # copy, so this branch only fires on the surface spelling and can never
    # re-warn a run config that point already normalized.
    if "objects" in cfg:
        cfg = normalize_content_keys(cfg, source=f"run config for scene {ref!r}")
    env_obj = cfg.setdefault("env", {})
    if not isinstance(env_obj, dict):
        raise ValueError(f"scene bundle {ref!r}: env must be a mapping")
    params_obj = env_obj.setdefault("params", {})
    if not isinstance(params_obj, dict):
        raise ValueError(f"scene bundle {ref!r}: env.params must be a mapping")
    params = params_obj
    backend_obj = doc.get("backend") or {}
    if not isinstance(backend_obj, dict):
        raise ValueError(
            f"scene bundle {ref!r}: backend must be a mapping, " f"got {type(backend_obj).__name__}"
        )
    mujoco_block = backend_obj.get("mujoco") or {}
    if not isinstance(mujoco_block, dict):
        raise ValueError(
            f"scene bundle {ref!r}: backend.mujoco must be a mapping, "
            f"got {type(mujoco_block).__name__}"
        )
    if "mj_model" in mujoco_block:
        raise ValueError(
            f"scene bundle {ref!r}: backend.mujoco.mj_model was removed in "
            f"v101 A1; declare `world_base: {{mjcf: <path>}}` there instead "
            f"(it routes to the run config top level) -- no alias/fallback"
        )
    # the world asset is run-level, not a backend param.
    if "world_base" in mujoco_block:
        cfg["world_base"] = _merge(mujoco_block["world_base"], cfg.get("world_base"))
    for key in ("mj_scene", "mj_scene_params", "base"):
        if key in mujoco_block:
            params[key] = _merge(mujoco_block[key], params.get(key))
    if "props" in doc:
        cfg["props"] = _merge(doc["props"], cfg.get("props"))
    if "initializer" in doc:
        params["initializer"] = _merge(doc["initializer"], params.get("initializer"))
    # The ref is a resolution directive, not a RunSpec.scene mapping.  Consume
    # it once the bundle has been merged so build_spec sees no leftover string.
    cfg.pop("scene", None)
    return cfg
