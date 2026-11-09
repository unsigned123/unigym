"""RunSpec: the pure-data construction intent of one run.

``build_spec(cfg, args)`` performs the pre-engine half of the old inline
``collect_cli.main`` path -- content-key normalization, scene-bundle
application, placeholder expansion, label/override validation and the CLI
overrides -- and freezes the result into a :class:`RunSpec`.  It never
builds an engine object: profiles, adapters, managers, envs, sinks and the
bridge are all constructed by :func:`unigym.run.build.build_instance`.

The input ``cfg`` dict is deep-copied first, so callers may reuse their
parsed yaml document.
"""
from __future__ import annotations

import copy
import json
import os
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from unigym._typing import (
    JsonValue,
    is_json_list,
    is_json_object,
    is_pair,
)
from unigym.backends.registry import backend_info, validate_backend_options
from unigym.core.asset_root import expand_placeholders
from unigym.scene_packs import apply_scene_bundle
from unigym.runtime.controls import (
    ControlError,
    ControlGroup,
    normalize_controls,
    validate_controls,
)
from unigym.scenes.articulation import (
    ArticulationError,
    ArticulationSpec,
    normalize_articulations,
)
from unigym.scenes.recipe import normalize_content_keys

if TYPE_CHECKING:
    from unigym.scenes.terrain import Terrain


@dataclass(frozen=True)
class ReplaySpec:
    """Trace replay options (toolkit B; docs/contracts/trace_contract.md §7)."""

    path: str | None = None
    mode: str = "action"
    key: str | None = None
    strict: bool = False
    freeze_rng: bool = False


@dataclass(frozen=True)
class OutputSpec:
    """Resolved output/artifacts wiring (lifecycle owns the final paths)."""

    out_dir: str | None = None
    artifacts_dir: str | None = None
    output_root: str = "data_out"
    artifacts: bool = True


@dataclass(frozen=True)
class RunSpec:
    """Frozen construction intent: data only, no live engine resources."""

    cfg: dict[str, JsonValue]
    config_path: str
    run_name: str
    seed: int
    mode: str
    backend: str
    backend_params: dict[str, JsonValue]
    #: backend-private options selected from ``env.backend_options.<backend>``
    #: and validated against the registry's ``options_schema``.
    backend_options: dict[str, JsonValue]
    #: the run-level world declaration (``{mjcf: <path>}``), kept OUT of
    #: ``backend_params``.  Required for mujoco-family backends.
    world_base: JsonValue
    #: the normalized articulation list (the old ``robot:`` block
    #: was removed).  Articulations are pure authoring data (no control mode).
    articulations: tuple[ArticulationSpec, ...]
    #: the explicit control groups: every driven joint is covered by
    #: exactly one group; undriven joints are declared in passive_joints.
    controls: tuple[ControlGroup, ...]
    adapter: dict[str, JsonValue]
    adapter_name: str | None
    adapter_params: dict[str, JsonValue]
    adapter_overrides: tuple[str, ...]
    scene: dict[str, JsonValue] | None
    props: JsonValue
    cameras: JsonValue
    camera_depth: JsonValue
    contacts: JsonValue
    director: dict[str, JsonValue] | None
    recorder: dict[str, JsonValue]
    bridges: dict[str, JsonValue]
    benchmark: dict[str, JsonValue]
    terminations: JsonValue
    runner: dict[str, JsonValue]
    output: OutputSpec
    sinks: str
    overrides: tuple[dict[str, JsonValue], ...]
    label: str | None
    direct: bool
    watch: str | None
    enable_cameras: bool
    replay: ReplaySpec
    num_envs: int
    total_steps: int
    device: str
    #: optional native terrain IR (programmatic ``Run(world=...)``);
    #: not serialized into the yaml config today.
    terrain: JsonValue | Terrain | None = None
    #: static primitive declarations (``World.add``, programmatic runs) the
    #: isaaclab backend maps to native static colliders; empty for
    #: config-backed runs.  Not serialized into the yaml config today.
    static_shapes: tuple[JsonValue, ...] = ()

def _route_content(
    cfg: dict[str, JsonValue], env_cfg: dict[str, JsonValue],
    backend_label: str,
) -> None:
    """Route the run's content spec into ``env.params.content`` (mujoco).

    The spec was split out of the run's ``objects:`` block by
    :func:`unigym.scenes.recipe.normalize_content_keys`; the mujoco scene
    build consumes it.  Other backends have no scene fill, so a declared
    spec fails closed instead of being silently dropped.
    """
    spec = cfg.get("content")
    if spec is None:
        return
    info = backend_info(backend_label)
    if info.content_router is None:
        raise SystemExit(
            f"[collect] error: the 'objects:' content spec "
            f"(sets/demands) is not supported by the {backend_label!r} "
            f"backend | hint: declare a backend with a content fill or "
            f"drop the content keys")
    info.content_router()(spec, env_cfg)


_STATIC_SHAPE_HINT = (
    "inline static shapes are mujoco-only; use dynamic=True (props path) "
    "or a static prop/terrain for isaaclab")


def _guard_mujoco_only_scene(
    env_cfg: dict[str, JsonValue], backend_label: str,
) -> None:
    """Fail fast when a run declares a scene fill its backend cannot consume.

    ``env.params.mj_scene`` names a registered/inline MuJoCo scene composer.
    The isaaclab backend has no scene-IR consumer, so without this gate the
    geometry would silently disappear.  Programmatic ``World.add`` static
    primitives are mapped to native static colliders before this point
    (the run layer strips ``mj_scene``); anything still carrying it here is
    an unmapped inline scene or a registered mujoco scene.
    """
    if backend_info(backend_label).mj_scene_consumer:
        return   # the backend has a scene-IR / mj_scene consumer
    params = env_cfg.get("params")
    scene_name = params.get("mj_scene") if isinstance(params, dict) else None
    if scene_name is None:
        return
    from unigym.errors import ConfigError

    raise ConfigError(
        f"env.params.mj_scene={scene_name!r} is a mujoco-only scene fill "
        f"({backend_label!r} backend has no scene-IR consumer) | hint: "
        f"{_STATIC_SHAPE_HINT}")


def _validate_world_base(
    cfg: dict[str, JsonValue], env_cfg: dict[str, JsonValue],
    backend_label: str,
) -> JsonValue:
    """Resolve/validate the run-level ``world_base`` declaration (A1).

    The declaration lives at the run config TOP LEVEL (scene bundles put
    it under ``backend.mujoco.world_base`` and :func:`apply_scene_bundle`
    routes it here).  ``env.params.mj_model`` is deleted -- no alias, no
    fallback -- and a mujoco-family run without ``world_base.mjcf`` fails
    fast.  Other backends keep the declaration untouched (they do not
    consume it today).
    """
    from unigym.errors import ConfigError

    params = env_cfg.get("params") or {}
    if isinstance(params, dict) and "mj_model" in params:
        raise ConfigError(
            "env.params.mj_model was removed in v101 A1; declare the "
            "run-level `world_base: {mjcf: <path>}` instead (scene "
            "bundles: `backend.mujoco.world_base`) -- there is no alias "
            "and no fallback")
    world_base = cfg.get("world_base")
    if not backend_info(backend_label).world_base_required:
        return world_base
    if world_base is None:
        raise ConfigError(
            f"world_base is required for the {backend_label!r} backend: "
            "declare `world_base: {mjcf: <path>}` at the run config top "
            "level (or `backend.mujoco.world_base` in a scene bundle); "
            "env.params.mj_model was removed and there is no default model")
    if not isinstance(world_base, dict) or not world_base.get("mjcf"):
        raise ConfigError(
            "world_base must be a mapping declaring `mjcf: <path>` for the "
            f"{backend_label!r} backend; got {world_base!r}")
    return {"mjcf": str(world_base["mjcf"])}


def _normalize_articulations(
    cfg: dict[str, JsonValue], source: str,
) -> tuple[ArticulationSpec, ...]:
    """Normalize ``cfg['articulations']`` and reject the removed ``robot:``.

    No compatibility alias.  A config still carrying
    the old top-level ``robot:`` block fails fast with the new shape; the
    ``articulations[]`` list is normalized into the frozen neutral IR.  The
    articulation-level ``control:`` block is rejected by the
    articulation normalizer (drives live in ``controls[]``).
    """
    from unigym.errors import ConfigError

    if "robot" in cfg:
        raise ConfigError(
            "the top-level `robot:` block was removed in v101 A2; declare "
            "`articulations: [{name: <robot>, params: ..., "
            "body_parts: ...}]` plus a top-level `controls:` list instead "
            "-- there is no alias")
    try:
        return normalize_articulations(cfg.get("articulations"))
    except ArticulationError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


def _normalize_controls(
    cfg: dict[str, JsonValue], articulations: tuple[ArticulationSpec, ...],
    source: str,
) -> tuple[ControlGroup, ...]:
    """Normalize/validate ``cfg['controls']`` against the articulations.

    A run that declares articulations must declare controls.  Every
    driven joint belongs to exactly one group; undriven joints are listed in
    the owning articulation's ``passive_joints`` (validated here).
    """
    from unigym.errors import ConfigError

    entries = cfg.get("controls")
    if not entries or not isinstance(entries, (list, tuple, Mapping)):
        raise ConfigError(
            f"{source}: no controls declared; the v104 shape requires a "
            f"top-level `controls:` list (one group per driven joint set): "
            f"`controls: [{{name: <face>, adapter: <name>, articulation: "
            f"<name>, drive: position_pd, target: position}}]`")
    parsed_entries: list[ControlGroup | Mapping[str, JsonValue]]
    if isinstance(entries, Mapping):
        entry_source: (
            Mapping[str, JsonValue] | list[ControlGroup | Mapping[str, JsonValue]]
        ) = entries
    else:
        parsed_entries = []
        for entry in entries:
            if isinstance(entry, (ControlGroup, Mapping)):
                parsed_entries.append(entry)
            else:
                raise ConfigError(
                    f"{source}: controls entries must be mappings, "
                    f"got {entry!r}")
        entry_source = parsed_entries
    try:
        controls = normalize_controls(entry_source)
        validate_controls(articulations, controls)
    except ControlError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    return controls


def _require_config(value: object, what: str) -> dict[str, JsonValue]:
    """Narrow a config boundary to a string-keyed JSON mapping."""
    from unigym.errors import ConfigError

    if not is_json_object(value):
        raise ConfigError(f"{what} must be a mapping, got {value!r}")
    return value


def _require_str(value: object, what: str) -> str:
    """Narrow a config scalar to a string."""
    from unigym.errors import ConfigError

    if not isinstance(value, str):
        raise ConfigError(f"{what} must be a string, got {value!r}")
    return value


def _as_int(value: object, what: str) -> int:
    """Coerce a config scalar to an int (bool/int/float/str only)."""
    from unigym.errors import ConfigError

    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return int(value)
    raise ConfigError(f"{what} must be an integer, got {value!r}")


def _override_result(
    value: object,
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    """Unpack the ``apply_overrides`` result at its untyped boundary."""
    from unigym.errors import ConfigError

    if is_pair(value):
        cfg_part, diffs_part = value
        if is_json_object(cfg_part) and is_json_list(diffs_part):
            diffs: list[dict[str, JsonValue]] = []
            for item in diffs_part:
                if is_json_object(item):
                    diffs.append(item)
            return cfg_part, diffs
    raise ConfigError(f"config overrides returned {type(value).__name__}")


def _adapter_params_result(
    value: object,
) -> tuple[dict[str, JsonValue], list[str]]:
    """Unpack the ``apply_spawn_params`` result at its untyped boundary."""
    from unigym.errors import ConfigError

    if is_pair(value):
        section_part, overrides_part = value
        if is_json_object(section_part) and is_json_list(overrides_part):
            overrides = [str(item) for item in overrides_part]
            return section_part, overrides
    raise ConfigError(f"adapter params returned {type(value).__name__}")


def build_spec(
    cfg: dict[str, JsonValue], args: Namespace | SimpleNamespace,
) -> RunSpec:
    """Validate/normalize one run config + CLI overrides into a RunSpec."""
    raw_cfg = copy.deepcopy(cfg)
    # The content key surface ('objects:') normalizes onto the
    # internal name BEFORE the scene merge, so a run config and the bundle
    # it references both migrate through the one function.
    cfg = _require_config(
        normalize_content_keys(raw_cfg, source=args.config), "content keys")
    # scene bundle: the `scene:` reference resolves BEFORE placeholder expansion
    # so the pack's ${UNIGYM_REPO} tokens ride the same single
    # expansion point as the run yaml
    cfg = _require_config(apply_scene_bundle(cfg), "scene bundle")
    cfg = _require_config(expand_placeholders(cfg), "placeholders")
    # run label (display metadata): fail-fast teaching gate BEFORE any
    # build work -- same rule the daemon's RunSpec.validate enforces.
    from unigym.core.run_ops import check_label
    label_err = check_label(getattr(args, "label", None))
    if label_err:
        raise SystemExit(f"[collect] error: --label rejected: {label_err} "
                         f"| hint: shorten the label -- it is display "
                         f"metadata, never a path component")
    # -s key=val overrides (before any engine-side read so the
    # whole tree sees the effective config; diff -> meta.json)
    from unigym.scripts.config_overrides import apply_overrides, diff_lines, \
        parse_set_args
    overrides = parse_set_args(getattr(args, "set", None))
    cfg, override_diffs = _override_result(apply_overrides(cfg, overrides))
    for line in diff_lines(override_diffs):
        print(f"[collect] {line}", flush=True)
    # spawn adapter params (most specific layer of the adapter section:
    # defaults < config yaml < spawn adapter_params).  The overlay merges
    # into cfg["adapter"] ONLY -- env/benchmark/cameras are structurally
    # out of its reach.
    from unigym.adapters import AdapterParamsError, apply_spawn_params
    adapter_params: dict[str, JsonValue] = {}
    adapter_overrides: list[str] = []
    raw_params_arg = getattr(args, "adapter_params", None)
    if raw_params_arg:
        try:
            raw_params = json.loads(raw_params_arg)
        except ValueError as e:
            raise SystemExit(
                "[collect] error: --adapter-params is not valid JSON: "
                f"{e} | hint: pass a JSON object of scalar / scalar-list "
                "values, e.g. '{\"name\": \"manual\", "
                "\"control_dt\": 0.02}'") from e
        try:
            adapter_section, adapter_overrides = _adapter_params_result(
                apply_spawn_params(
                    _require_config(cfg.get("adapter") or {}, "adapter"),
                    raw_params))
        except AdapterParamsError as e:
            raise SystemExit(
                f"[collect] error: --adapter-params rejected: {e} | "
                f"details: {json.dumps(e.doc, ensure_ascii=False)}") from e
        cfg["adapter"] = adapter_section
        adapter_params = _require_config(raw_params, "adapter params")
        if adapter_overrides:
            print(f"[collect] adapter params override: "
                  f"{adapter_overrides}", flush=True)
    mode_value = cfg.get("mode", "collect")
    if not isinstance(mode_value, str) \
            or mode_value not in ("collect", "benchmark", "both"):
        raise ValueError(
            f"mode '{mode_value}' is not implemented "
            "(one of: collect | benchmark | both)")
    mode = mode_value
    env_cfg = _require_config(cfg["env"], "env")
    col_cfg = _require_config(cfg["collect"], "collect")
    adapter_cfg = _require_config(cfg.get("adapter") or {}, "adapter")
    if getattr(args, "num_envs", None) is not None:
        env_cfg["num_envs"] = args.num_envs
    if getattr(args, "steps", None) is not None:
        col_cfg["steps"] = args.steps
    if getattr(args, "episode_length_s", None) is not None:
        env_cfg["episode_length_s"] = args.episode_length_s
    if getattr(args, "ckpt_dir", None) is not None:
        adapter_cfg["ckpt_dir"] = args.ckpt_dir
        cfg["adapter"] = adapter_cfg
    if getattr(args, "instruction", None) is not None:
        adapter_cfg["instruction"] = args.instruction
        cfg["adapter"] = adapter_cfg
    seed_value = (args.seed if getattr(args, "seed", None) is not None
                  else cfg.get("seed", 0))
    seed = _as_int(seed_value, "seed")

    # torch device follows --device (AppLauncher); the env backend also sets
    # cfg.sim.device so SimulationContext lands on the same GPU.
    device = args.device if getattr(args, "device", "cpu") != "cpu" else "cpu"
    # The run's ``objects:`` content spec (split out by the
    # normalizer) rides env.params into the mujoco scene build.
    backend_label = str(env_cfg.get("backend", "isaaclab"))
    from unigym.errors import ConfigError
    raw_backend_options = env_cfg.get("backend_options")
    if raw_backend_options is not None \
            and not is_json_object(raw_backend_options):
        raise ConfigError(
            f"{args.config}: env.backend_options must be a mapping")
    try:
        backend_options = _require_config(
            validate_backend_options(backend_label, raw_backend_options),
            "backend options")
    except ValueError as exc:
        raise ConfigError(f"{args.config}: {exc}") from exc
    _guard_mujoco_only_scene(env_cfg, backend_label)
    _route_content(cfg, env_cfg, backend_label)
    world_base = _validate_world_base(cfg, env_cfg, backend_label)
    articulations = _normalize_articulations(cfg, args.config)
    controls = _normalize_controls(cfg, articulations, args.config)
    num_envs = _as_int(env_cfg["num_envs"], "env.num_envs")
    total_steps = _as_int(col_cfg["steps"], "collect.steps")
    # reset policy (docs/contracts/runner_contract.md §5): clock due by default;
    # reset_on_termination merges terminated rows.  runner.reset_mask_fn is
    # a reserved key (not deserialized here; inject via SyncRunner).
    runner_cfg = _require_config(cfg.get("runner") or {}, "runner")
    output_root_value = cfg.get("output_root", "data_out")
    output_root = (output_root_value if isinstance(output_root_value, str)
                   else "data_out")
    run_name = _require_str(cfg["run_name"], "run_name")
    requested_out = getattr(args, "output_dir", None)
    if requested_out is None:
        requested_out = os.path.join(output_root, run_name)
    scene_value = cfg.get("scene")
    scene = (None if scene_value is None
             else _require_config(scene_value, "scene"))
    director_value = cfg.get("director")
    director = (None if director_value is None
                else _require_config(director_value, "director"))
    adapter_name_value = adapter_cfg.get("name")
    adapter_name = (None if adapter_name_value is None
                    else str(adapter_name_value))
    return RunSpec(
        cfg=cfg,
        config_path=args.config,
        run_name=run_name,
        seed=seed,
        mode=mode,
        backend=backend_label,
        backend_params=_require_config(
            env_cfg.get("params") or {}, "env.params"),
        backend_options=backend_options,
        world_base=world_base,
        articulations=articulations,
        controls=controls,
        adapter=adapter_cfg,
        adapter_name=adapter_name,
        adapter_params=adapter_params,
        adapter_overrides=tuple(adapter_overrides),
        scene=scene,
        props=cfg.get("props"),
        cameras=cfg.get("cameras"),
        camera_depth=cfg.get("camera_depth"),
        contacts=cfg.get("contacts"),
        director=director,
        recorder=_require_config(cfg.get("recorder") or {}, "recorder"),
        bridges={
            "serve_camera": getattr(args, "serve_camera", None),
            "serve_frame_mb": getattr(args, "serve_frame_mb", 4.0),
            "serve_state_dir": getattr(args, "serve_state_dir", None),
            "run_log": getattr(args, "run_log", None),
            "serve_pause": bool(getattr(args, "serve_pause", False)),
        },
        benchmark=_require_config(
            cfg.get("benchmark") or {}, "benchmark"),
        terminations=cfg.get("terminations"),
        runner=runner_cfg,
        output=OutputSpec(
            out_dir=requested_out,
            artifacts_dir=None,
            output_root=output_root,
            artifacts=not bool(getattr(args, "no_artifacts", False)),
        ),
        sinks=mode,
        overrides=tuple(override_diffs),
        label=getattr(args, "label", None),
        direct=bool(getattr(args, "direct", False)),
        watch=getattr(args, "watch", None),
        enable_cameras=bool(getattr(args, "enable_cameras", False)),
        replay=ReplaySpec(
            path=getattr(args, "replay", None),
            mode=getattr(args, "replay_mode", "action"),
            key=getattr(args, "replay_key", None),
            strict=bool(getattr(args, "replay_strict", False)),
            freeze_rng=bool(getattr(args, "replay_freeze_rng", False)),
        ),
        num_envs=num_envs,
        total_steps=total_steps,
        device=device,
    )
