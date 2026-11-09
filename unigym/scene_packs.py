"""Scene packs: discovery + the run-config ``scene:`` reference merge.

A scene bundle answers "what is this world": the MJ model, the base joint semantics,
the dynamic props and the reset initializer.  Run configs reference one
by name (``scene: <family>/<name>``) and the merge routes the bundle's
blocks into the run config's existing consumption points BEFORE
placeholder expansion (the pack's ``${UNIGYM_REPO}`` tokens ride the
same single expansion point as the run yaml).  The actuator laws are
part of the world declaration: a bundle WITHOUT its ``backend.mujoco.actuators``
block leaves the seam on its legacy default path, and a run config
that dropped its ``env.params.actuators`` block on the migration
intent must therefore fall through to the scene declaration, never
silently to ``standard_pd``.

=======================  =============================================
``backend.mujoco``       routed: ``world_base`` -> the run config TOP
                         LEVEL (the world declaration); the rest
                         (``mj_scene``, ``mj_scene_params``, ``base``,
                         ``actuators``) -> ``env.params`` -- backend
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

Discovery mirrors the benchmark_packs mechanism: the ``scenes`` face of
the mounted packs (the checkout's own ``scene_packs/`` is the implicit
``repo`` mount's face) is enumerated for ``scene.yaml``; the
reference's family is the bundle directory, the name the yaml's
top-level ``name:`` key.  Merge precedence is the three-layer rule (scene defaults -> task
overrides -> run overrides): a run-level value always wins; mappings
merge key-wise (so a run can override a single initializer param, e.g.
the per-seed stream seed, without restating the table).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

from unigym._typing import JsonValue, is_json_object
from unigym.core.asset_root import repo_root
from unigym.packs import find, repo_pack
from unigym.scenes.recipe import normalize_content_keys

SCENES_DIR = "scene_packs"
"""Repo-mount face directory holding the scene bundle families."""
SCENE_FILE = "scene.yaml"
"""The bundle definition file every discovered family owns."""


def scenes_root() -> Path:
    """The repo mount's ``scenes`` face (this checkout's ``scene_packs/``)."""
    face = repo_pack().faces.get("scenes")
    return face if face is not None else Path(repo_root()) / SCENES_DIR


def _scene_hits(root: Optional[Path]) -> list[tuple[Path, Path]]:
    """``(face_root, scene.yaml)`` pairs for *root* or the mounted faces."""
    if root is not None:
        base = Path(root)
        return [(base, path) for path in sorted(base.rglob(SCENE_FILE))]
    hits: list[tuple[Path, Path]] = []
    for pack, path in find("scenes", suffix=SCENE_FILE):
        face = pack.faces.get("scenes")
        if face is not None:
            hits.append((face, path))
    return hits


class ScenePackIndex(dict):
    """The scene index plus the cross-mount shadow records.

    ``shadowed`` holds ``(key, shadowing_base, kept_path, dropped_path)``
    tuples: during the v081 transition an external pack and the repo can
    both carry the same bundle, and the pack wins (the same rule the run
    configs follow).
    """

    def __init__(self) -> None:
        super().__init__()
        self.shadowed: list = []


def discover_scene_packs(root: Optional[Path] = None) -> dict:
    """``{"<family>/<name>": path}`` over the recursive whitelist scan.

    A ``scene.yaml`` counts when it sits inside a family directory (at
    least one level below the root); the index key joins the family
    directory name with the yaml's top-level ``name:``.  Duplicate keys
    and name-less bundles fail fast (a scene without a declared name is
    a typo, not a scene).  With ``root`` given the scan keeps its
    literal ``rglob`` semantics; ``root=None`` enumerates the mounted
    packs' ``scenes`` faces through the resolver.
    """
    index: ScenePackIndex = ScenePackIndex()
    origin: dict = {}
    for base, path in _scene_hits(root):
        rel = path.relative_to(base)
        if len(rel.parts) < 2:
            continue               # a bare root-level scene.yaml: no family
        family = rel.parts[0]
        doc = yaml.safe_load(path.read_text()) or {}
        name = str(doc.get("name") or "").strip()
        if not name:
            raise ValueError(
                f"{path}: scene bundle must declare a top-level 'name' "
                f"(referenced as '{family}/<name>')")
        key = f"{family}/{name}"
        if key in index:
            if origin[key] == base:
                raise ValueError(
                    f"duplicate scene bundle {key!r}: {index[key]} and "
                    f"{path}")
            # cross-mount duplicate: the earlier mount (external pack)
            # wins; the repo copy is shadowed during the transition
            index.shadowed.append((key, base, index[key], path))
            continue
        index[key] = path
        origin[key] = base
    return index


def load_scene_bundle(
    ref: str, root: Optional[Path] = None,
) -> dict[str, JsonValue]:
    """Resolve one ``<family>/<name>`` reference to its parsed bundle."""
    index = discover_scene_packs(root)
    ref = str(ref)
    path = index.get(ref)
    if path is None:
        raise ValueError(
            f"unknown scene bundle {ref!r} | available: {sorted(index)} "
            f"| hint: scene refs resolve against <repo>/"
            f"{SCENES_DIR}/<family>/{SCENE_FILE} (the yaml's top-level "
            f"'name:' key is the second reference component)")
    doc = yaml.safe_load(path.read_text()) or {}
    if not is_json_object(doc):
        raise ValueError(f"{path}: scene bundle must be a mapping")
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
    cfg: dict[str, JsonValue], root: Optional[Path] = None,
) -> dict[str, JsonValue]:
    """Resolve the run config's ``scene:`` reference (deep-copy merge).

    Returns the config with the bundle's blocks routed in; configs
    without a ``scene:`` key pass through untouched (the legacy face,
    zero behavior difference).  Run-level values win at every layer;
    mapping sections (``initializer.params``, ``base``, ``actuators``)
    merge key-wise so a run can carry a single param override (e.g. the
    per-seed stream seed) or one actuator group override without
    restating the table.  Only ``backend.mujoco`` blocks route this
    phase; other backend blocks are left for their backends.
    """
    ref = cfg.get("scene")
    if ref is None:
        return cfg
    doc = normalize_content_keys(
        load_scene_bundle(str(ref), root), source=f"scene bundle {ref!r}")
    cfg = copy.deepcopy(cfg)
    # A direct caller (tests, the API) may hand us an un-normalized run
    # config; the collector's single load point already migrated its own
    # copy, so this branch only fires on the surface spelling and can never
    # re-warn a run config that point already normalized.
    if "objects" in cfg:
        cfg = normalize_content_keys(
            cfg, source=f"run config for scene {ref!r}")
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
            f"scene bundle {ref!r}: backend must be a mapping, "
            f"got {type(backend_obj).__name__}")
    mujoco_block = backend_obj.get("mujoco") or {}
    if not isinstance(mujoco_block, dict):
        raise ValueError(
            f"scene bundle {ref!r}: backend.mujoco must be a mapping, "
            f"got {type(mujoco_block).__name__}")
    if "mj_model" in mujoco_block:
        raise ValueError(
            f"scene bundle {ref!r}: backend.mujoco.mj_model was removed in "
            f"v101 A1; declare `world_base: {{mjcf: <path>}}` there instead "
            f"(it routes to the run config top level) -- no alias/fallback")
    # the world asset is run-level, not a backend param.
    if "world_base" in mujoco_block:
        cfg["world_base"] = _merge(mujoco_block["world_base"],
                                   cfg.get("world_base"))
    for key in ("mj_scene", "mj_scene_params", "base", "actuators"):
        if key in mujoco_block:
            params[key] = _merge(mujoco_block[key], params.get(key))
    if "props" in doc:
        cfg["props"] = _merge(doc["props"], cfg.get("props"))
    if "initializer" in doc:
        params["initializer"] = _merge(doc["initializer"],
                                       params.get("initializer"))
    return cfg
