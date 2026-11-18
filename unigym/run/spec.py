"""RunSpec: the pure-data construction intent of one run.

``build_spec(cfg, args)`` performs the pre-engine half of the old inline
``collect_cli.main`` path -- content-key normalization, scene-preset
resolution, placeholder expansion, label/override validation and the CLI
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
from unigym.assets.asset_ref import AssetRef, AssetRefError, select_asset_format
from unigym.backends.registry import (
    backend_info,
    known_backends,
    validate_backend_options,
)
from unigym.core.paths import expand_placeholders
from unigym.env.obs import ObservationConfig, ObsConfigError
from unigym.runtime.actions import (
    OrderedActionSpec,
    PolicyBinding,
)
from unigym.runtime.controllers import validate_policy_bindings
from unigym.runtime.controls import ControlError
from unigym.scenes.articulation import (
    ArticulationError,
    ArticulationSpec,
    normalize_articulations,
)
from unigym.scenes.recipe import normalize_content_keys
from unigym.scene_presets import (
    BackendBinding,
    apply_scene_preset,
    reject_legacy_scene_keys,
)

if TYPE_CHECKING:
    from unigym.robots.base import RobotProfile

#: Placeholder spellings that make a config an asset-bearing document.  They
#: are resolved (or rejected as removed) by ``expand_placeholders``; the
#: composition boundary must supply an explicit runtime/context for them.
_ASSET_REF_MARKERS = ("${UNIGYM_ASSET_ROOT}", "${UNIGYM_ASSET_PACK}")

#: Supported termination ownership modes.  ``core`` builds the termination
#: group from the config ``terminations:`` section (today's behavior);
#: ``external`` disables core terminations so an external task layer (the RL
#: TaskEnv) owns every terminated/truncated mask.
_TERMINATION_MODES = ("core", "external")

if TYPE_CHECKING:
    from unigym.packs.runtime import PackRuntime
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
    #: the ordered action groups: every driven joint is covered by exactly
    #: one group; undriven joints are declared in passive_joints.
    action_spec: OrderedActionSpec
    #: the policy bindings: which adapter owns which ``(articulation, group)``
    #: action group, plus optional control_dt/agent metadata.
    policies: tuple[PolicyBinding, ...]
    adapter: dict[str, JsonValue]
    adapter_name: str | None
    adapter_params: dict[str, JsonValue]
    adapter_overrides: tuple[str, ...]
    props: JsonValue
    cameras: JsonValue
    contacts: JsonValue
    #: the parsed ``observations:`` declaration (config-declared sources).
    observations: ObservationConfig
    director: dict[str, JsonValue] | None
    recorder: dict[str, JsonValue]
    bridges: dict[str, JsonValue]
    benchmark: dict[str, JsonValue]
    terminations: JsonValue
    #: termination ownership mode: ``core`` (default; the ``terminations:``
    #: section above is the source) or ``external`` (core builds an empty
    #: never-firing group and the task layer owns termination).
    termination_mode: str
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
    #: the caller's pack runtime (non-serialized): benchmark pack discovery
    #: reads its ``benchmarks`` catalog face.  Never written to cfg/ledger.
    pack_runtime: PackRuntime | None = None
    #: complete RobotProfile objects injected by the programmatic profile
    #: path, keyed by articulation name.  Non-serialized: these bypass the
    #: registry factory lookup only.
    robot_profiles: Mapping[str, RobotProfile] | None = None


def _route_content(
    cfg: dict[str, JsonValue],
    env_cfg: dict[str, JsonValue],
    backend_label: str,
) -> None:
    """Route the run's content spec into ``env.params.content``.

    The spec was split out of the run's ``objects:`` block by
    :func:`unigym.scenes.recipe.normalize_content_keys`; a backend with a
    ``content_router`` consumes it.  A fill always needs a composed scene
    (``env.params.scene_provider``, written only by
    :func:`unigym.scene_presets.apply_scene_preset`) and a backend with a
    ``content_router``; either missing fails closed here, before any env is
    constructed, instead of being silently dropped.
    """
    from unigym.errors import ConfigError

    spec = cfg.get("content")
    if spec is None:
        return
    params = env_cfg.get("params")
    scene_provider = params.get("scene_provider") if isinstance(params, dict) else None
    if scene_provider is None:
        raise ConfigError(
            "the 'objects:' content fill (filler/sets/demands) needs a composed "
            "scene | hint: declare 'scene_preset:' with a provider or drop the "
            "content keys (objects.filler)"
        )
    info = backend_info(backend_label)
    if info.content_router is None:
        raise ConfigError(
            f"the 'objects:' content spec is not supported by the {backend_label!r} "
            f"backend | hint: declare a backend with a content fill or drop the "
            f"content keys (objects.filler)"
        )
    info.content_router()(spec, env_cfg)


_STATIC_SHAPE_HINT = (
    "inline static shapes are mujoco-only; use dynamic=True (props path) "
    "or a static prop/terrain for isaaclab"
)


def _guard_scene_provider_support(
    env_cfg: dict[str, JsonValue],
    backend_label: str,
) -> None:
    """Fail fast when a run declares a scene provider its backend cannot consume."""
    if backend_info(backend_label).scene_provider_consumer:
        return
    params = env_cfg.get("params")
    scene_name = params.get("scene_provider") if isinstance(params, dict) else None
    if scene_name is None:
        return
    from unigym.errors import ConfigError

    raise ConfigError(
        f"env.params.scene_provider={scene_name!r} is not supported by the "
        f"{backend_label!r} backend (no scene-provider consumer) | hint: "
        f"{_STATIC_SHAPE_HINT}"
    )


def _validate_world_base(
    cfg: dict[str, JsonValue],
    env_cfg: dict[str, JsonValue],
    backend_label: str,
) -> JsonValue:
    """Resolve/validate the run-level format-neutral ``world_base``.

    ``world_base`` is one ``AssetRef`` mapping (``mjcf`` / ``urdf`` / ``usd``).
    The run layer selects the first format the backend supports; there is no
    format fallback and no bare-path form in a yaml document.  Backends that
    do not declare ``world_base_consumer`` reject a declared world base.
    """
    from unigym.errors import ConfigError

    params = env_cfg.get("params") or {}
    if isinstance(params, dict) and "mj_model" in params:
        raise ConfigError(
            "env.params.mj_model was removed; declare the run-level "
            "`world_base: {mjcf: <path>}` or a scene preset `world:` instead "
            "-- there is no alias and no fallback"
        )
    info = backend_info(backend_label)
    world_base = cfg.get("world_base")
    if world_base is None:
        if info.world_base_consumer:
            raise ConfigError(
                f"world_base is required for the {backend_label!r} backend: "
                "declare `world_base: {mjcf: <path>}` at the run config top "
                "level or a scene preset `world:`; env.params.mj_model was "
                "removed and there is no default model"
            )
        return None
    if not info.world_base_consumer:
        raise ConfigError(
            f"the {backend_label!r} backend does not consume a world base; "
            f"got world_base={world_base!r} | hint: declare a native terrain "
            "or drop the world_base/scene preset world"
        )
    if not isinstance(world_base, dict) or not any(
        isinstance(world_base.get(key), str) and world_base.get(key)
        for key in ("mjcf", "urdf", "usd")
    ):
        raise ConfigError(
            "world_base must be a mapping declaring at least one non-empty "
            f"`mjcf` / `urdf` / `usd` path for the {backend_label!r} backend; "
            f"got {world_base!r}"
        )
    try:
        ref = AssetRef.from_mapping(world_base, "world_base")
        world_format = select_asset_format(ref, info.asset_formats)
    except AssetRefError as err:
        raise ConfigError(f"world_base: {err}") from err
    return {world_format: str(getattr(ref, world_format))}


def _normalize_articulations(
    cfg: dict[str, JsonValue],
    source: str,
) -> tuple[ArticulationSpec, ...]:
    """Normalize ``cfg['articulations']`` and reject the removed ``robot:``.

    No compatibility alias.  A config still carrying
    the old top-level ``robot:`` block fails fast with the new shape; the
    ``articulations[]`` list is normalized into the frozen neutral IR.  The
    articulation-level ``control:`` block is rejected by the
    articulation normalizer (drives live in ``action_spec`` + ``policies``).
    """
    from unigym.errors import ConfigError

    if "robot" in cfg:
        raise ConfigError(
            "the top-level `robot:` block was removed in v101 A2; declare "
            "`articulations: [{name: <robot>, params: ...}]` plus top-level "
            "`action_spec:` groups and `policies:` bindings instead -- there "
            "is no alias"
        )
    try:
        return normalize_articulations(cfg.get("articulations"))
    except ArticulationError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


def _normalize_action_authoring(
    cfg: dict[str, JsonValue],
    articulations: tuple[ArticulationSpec, ...],
    source: str,
) -> tuple[OrderedActionSpec, tuple[PolicyBinding, ...]]:
    """Normalize/validate ``cfg['action_spec']`` + ``cfg['policies']``.

    The split authoring surface is mandatory: ``action_spec`` declares every
    ordered action group (explicit joints) and ``policies`` binds those groups
    to the adapter that produces them.  The removed ``controls:`` block is
    rejected with a teaching error (no alias).  Every action group must name a
    declared articulation, every policy binding must reference a declared
    group, and no group may be owned by two policies.
    """
    from unigym.errors import ConfigError

    if "controls" in cfg:
        raise ConfigError(
            f"{source}: the top-level `controls:` block was removed; declare "
            f"`action_spec:` action groups (name/articulation/joints/target/"
            f"role) plus `policies:` bindings (adapter/groups[/control_dt/"
            f"agent]) instead -- there is no alias"
        )
    raw_spec = cfg.get("action_spec")
    if isinstance(raw_spec, OrderedActionSpec):
        action_spec = raw_spec
    else:
        if raw_spec is None or not isinstance(raw_spec, (list, tuple)) or not raw_spec:
            raise ConfigError(
                f"{source}: no action_spec declared; the authoring shape requires "
                f"a top-level `action_spec:` list, e.g. `action_spec: [{{name: "
                f"legs, articulation: go1, joints: [...], target: position, "
                f"role: body}}]`"
            )
        try:
            action_spec = OrderedActionSpec.from_mapping(raw_spec)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}: action_spec: {exc}") from exc
    declared_articulations = {art.name for art in articulations}
    for group in action_spec.groups:
        if group.articulation not in declared_articulations:
            raise ConfigError(
                f"{source}: action group {group.name!r} names articulation "
                f"{group.articulation!r}, which is not declared; declared "
                f"articulations: {sorted(declared_articulations)}"
            )
    raw_policies = cfg.get("policies")
    if raw_policies is None or not isinstance(raw_policies, (list, tuple)) or not raw_policies:
        raise ConfigError(
            f"{source}: no policies declared; the authoring shape requires a "
            f"top-level `policies:` list, e.g. `policies: [{{adapter: manual, "
            f"groups: [[go1, legs]]}}]`"
        )
    policies: list[PolicyBinding] = []
    for index, entry in enumerate(raw_policies):
        if isinstance(entry, PolicyBinding):
            policies.append(entry)
        elif isinstance(entry, Mapping):
            try:
                policies.append(PolicyBinding.from_mapping(entry))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{source}: policies[{index}]: {exc}") from exc
        else:
            raise ConfigError(f"{source}: policies[{index}] must be a mapping, got {entry!r}")
    try:
        validate_policy_bindings(action_spec.groups, policies)
    except ControlError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    declared_adapter = _declared_adapter_name(cfg)
    if declared_adapter is not None:
        bound = sorted({binding.adapter for binding in policies})
        if declared_adapter not in bound:
            raise ConfigError(
                f"{source}: adapter name {declared_adapter!r} owns no policy "
                f"binding; policies name {bound} | hint: the single-adapter "
                f"contract requires `policies[*].adapter` to include "
                f"`adapter.name`"
            )
    return action_spec, tuple(policies)


def _contains_asset_ref(value: JsonValue) -> bool:
    """True when *value* carries a pack reference or a removed token.

    ``release:`` references are deliberately excluded: they are resolved by
    the ``unigym.asset_release`` provider (or the ``unigym_robots`` fallback)
    and never need a pack runtime.
    """
    if isinstance(value, str):
        return value.startswith("pack:") or any(marker in value for marker in _ASSET_REF_MARKERS)
    if isinstance(value, dict):
        return any(_contains_asset_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_asset_ref(item) for item in value)
    return False


def _declares_prop_template(value: JsonValue) -> bool:
    """True when a content block selects a registered prop template."""
    if isinstance(value, dict):
        if isinstance(value.get("template"), str) and value["template"]:
            return True
        return any(_declares_prop_template(item) for item in value.values())
    if isinstance(value, list):
        return any(_declares_prop_template(item) for item in value)
    return False


def _scene_preset_needs_runtime(value: JsonValue) -> bool:
    """True when a ``scene_preset`` value needs a pack runtime to resolve."""
    if isinstance(value, str):
        return True
    if not isinstance(value, dict):
        return False
    if value.get("id") is not None:
        return True
    provider = value.get("provider")
    if not isinstance(provider, str) or not provider:
        return False
    if provider.startswith("_inline_"):
        return False
    from unigym.scenes.providers import registered_scene_providers

    return provider not in registered_scene_providers()


def _declared_adapter_name(cfg: dict[str, JsonValue]) -> str | None:
    """The adapter name declared by the config's ``adapter:`` mapping."""
    adapter = cfg.get("adapter")
    if not isinstance(adapter, Mapping):
        return None
    name = adapter.get("name")
    return name if isinstance(name, str) else None


def requires_pack_runtime(cfg: dict[str, JsonValue], mode: str | None = None) -> bool:
    """True when a run config needs an explicit runtime at its boundary.

    A catalog ``scene_preset`` (or an unregistered inline provider), a
    benchmark/both run, an adapter other than the built-in ``manual``, a
    ``pack:`` reference (or asset placeholder token), or a prop-template
    selection needs the runtime.  A ``release:`` reference does not: it is
    resolved by the ``unigym.asset_release`` provider or the ``unigym_robots``
    fallback.  The helper is the one composition-boundary decision shared by
    ``Run``, ``collect_cli`` and ``describe_cli``; it does not build or cache a
    runtime.
    """
    if "scene_preset" in cfg and _scene_preset_needs_runtime(cfg.get("scene_preset")):
        return True
    run_mode = cfg.get("mode", "collect") if mode is None else mode
    if run_mode in ("benchmark", "both"):
        return True
    adapter_name = _declared_adapter_name(cfg)
    if adapter_name is not None and adapter_name != "manual":
        return True
    for section in ("objects", "props"):
        block = cfg.get(section)
        if isinstance(block, dict) and _declares_prop_template(block.get("types")):
            return True
    return _contains_asset_ref(cfg)


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
    """Coerce a config scalar to an int (bool is rejected)."""
    from unigym.errors import ConfigError

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{what} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(f"{what} must be an integer, got {value!r}") from None


def _as_positive_int(value: object, what: str) -> int:
    """Coerce a config scalar to a strictly positive integer."""
    from unigym.errors import ConfigError

    number = _as_int(value, what)
    if number <= 0:
        raise ConfigError(f"{what} must be positive, got {number}")
    return number


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


def _parse_termination_mode(
    cfg: dict[str, JsonValue],
    runner_cfg: dict[str, JsonValue],
    config_path: str,
) -> str:
    """Validate the run's termination ownership mode.

    ``core`` (the default) leaves today's behavior untouched: the core builds
    its termination group from ``terminations:``.  ``external`` disables core
    terminations for an RL task layer, so it rejects any core termination
    source and requires a caller-owned reset policy.
    """
    from unigym.errors import ConfigError

    value = cfg.get("termination_mode", "core")
    if not isinstance(value, str) or value not in _TERMINATION_MODES:
        raise ConfigError(
            f"{config_path}: termination_mode={value!r} is not supported "
            f"(one of: {' | '.join(_TERMINATION_MODES)}) | hint: omit the key "
            "for the default 'core' mode"
        )
    if value == "core":
        return value
    terminations = cfg.get("terminations")
    if terminations is not None and not (is_json_list(terminations) and not terminations):
        raise ConfigError(
            f"{config_path}: termination_mode: external owns termination in "
            f"the task layer; 'terminations:' must be absent or empty, got "
            f"{terminations!r} | hint: remove the core terminations or use "
            "termination_mode: core"
        )
    auto_reset = runner_cfg.get("auto_reset")
    if auto_reset is not False:
        raise ConfigError(
            f"{config_path}: termination_mode: external requires an explicit "
            "runner.auto_reset: false so core never resets a terminated row; "
            f"got runner.auto_reset={auto_reset!r} | hint: set "
            "runner.auto_reset: false"
        )
    reset_on_termination = runner_cfg.get("reset_on_termination", False)
    if reset_on_termination is not False:
        raise ConfigError(
            f"{config_path}: termination_mode: external requires "
            "runner.reset_on_termination: false (the caller owns terminated "
            f"row resets); got runner.reset_on_termination="
            f"{reset_on_termination!r}"
        )
    return value


def _validate_episode_clock(
    env_cfg: dict[str, JsonValue],
    termination_mode: str,
    config_path: str,
) -> None:
    """Reject ambiguous episode-clock ownership.

    ``episode_length_s: null`` disables the core clock and is only meaningful
    when the task layer owns time-based truncation.  A numeric duration in
    external mode would create two independent clocks, so it is rejected
    rather than silently competing with the task's ``max_steps``.
    """
    from unigym.errors import ConfigError

    value = env_cfg.get("episode_length_s")
    if termination_mode == "core":
        if "episode_length_s" in env_cfg and value is None:
            raise ConfigError(
                f"{config_path}: episode_length_s: null disables the core "
                "episode clock, so it requires termination_mode: external | "
                "hint: give a positive numeric duration for core mode, or "
                "switch to termination_mode: external and let the task's "
                "max_steps own truncation"
            )
        return
    if value is not None:
        raise ConfigError(
            f"{config_path}: termination_mode: external with a numeric "
            f"episode_length_s={value!r} would create two independent clocks "
            "(core duration and the task's own budget) | hint: set "
            "episode_length_s: null and let the task max_steps own "
            "truncation, or use termination_mode: core"
        )


def build_spec(
    cfg: dict[str, JsonValue],
    args: Namespace | SimpleNamespace,
    *,
    runtime: PackRuntime | None = None,
    robot_profiles: Mapping[str, RobotProfile] | None = None,
) -> RunSpec:
    """Validate/normalize one run config + CLI overrides into a RunSpec.

    The run's ``scene_preset`` resolves through *runtime* after the backend is
    selected (its ``BackendBinding`` drives world-format and backend-param
    selection).  A ``pack:`` / asset reference expands through an
    :class:`~unigym.packs.runtime.AssetContext` built from the same runtime.
    A ``benchmark`` / ``both`` run stores *runtime* on the spec for benchmark
    pack discovery.  A legacy ``benchmark.packs`` list is rejected: packs are
    declared on the runtime ``benchmarks`` face.
    """
    from unigym.errors import ConfigError

    raw_cfg = copy.deepcopy(cfg)
    reject_legacy_scene_keys(raw_cfg, source=args.config)
    # The content key surface ('objects:') normalizes onto the internal name
    # BEFORE preset resolution; an ``objects.provider`` rejection is a config
    # error at this boundary.
    try:
        cfg = _require_config(normalize_content_keys(raw_cfg, source=args.config), "content keys")
    except ValueError as exc:
        raise ConfigError(f"{args.config}: {exc}") from exc
    early_env = _require_config(cfg.get("env"), "env")
    backend_label = str(early_env.get("backend", "isaaclab"))
    info = backend_info(backend_label)
    binding = BackendBinding(
        name=info.name,
        asset_formats=info.asset_formats,
        backend_param_keys=info.backend_param_keys,
        known_backends=tuple(known_backends()),
    )
    # The preset resolves BEFORE placeholder expansion so the pack's
    # ${UNIGYM_REPO} tokens ride the same single expansion point as the run yaml.
    cfg = _require_config(
        apply_scene_preset(cfg, runtime=runtime, binding=binding, source=args.config),
        "scene preset",
    )
    context = None
    if runtime is not None:
        from unigym.packs.runtime import AssetContext

        context = AssetContext(runtime)
    cfg = _require_config(expand_placeholders(cfg, context), "placeholders")
    # run label (display metadata): fail-fast teaching gate BEFORE any
    # build work -- same rule the daemon's RunSpec.validate enforces.
    from unigym.core.run_ops import check_label

    label_err = check_label(getattr(args, "label", None))
    if label_err:
        raise SystemExit(
            f"[collect] error: --label rejected: {label_err} "
            f"| hint: shorten the label -- it is display "
            f"metadata, never a path component"
        )
    # -s key=val overrides (before any engine-side read so the
    # whole tree sees the effective config; diff -> meta.json)
    from unigym.scripts.config_overrides import apply_overrides, diff_lines, parse_set_args

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
                'values, e.g. \'{"name": "manual", '
                '"control_dt": 0.02}\''
            ) from e
        try:
            adapter_section, adapter_overrides = _adapter_params_result(
                apply_spawn_params(_require_config(cfg.get("adapter") or {}, "adapter"), raw_params)
            )
        except AdapterParamsError as e:
            raise SystemExit(
                f"[collect] error: --adapter-params rejected: {e} | "
                f"details: {json.dumps(e.doc, ensure_ascii=False)}"
            ) from e
        cfg["adapter"] = adapter_section
        adapter_params = _require_config(raw_params, "adapter params")
        if adapter_overrides:
            print(f"[collect] adapter params override: " f"{adapter_overrides}", flush=True)
    mode_value = cfg.get("mode", "collect")
    if not isinstance(mode_value, str) or mode_value not in ("collect", "benchmark", "both"):
        raise ValueError(
            f"mode '{mode_value}' is not implemented " "(one of: collect | benchmark | both)"
        )
    mode = mode_value
    benchmark_cfg = _require_config(cfg.get("benchmark") or {}, "benchmark")
    if "camera_depth" in cfg:
        raise ConfigError(
            f"{args.config}: top-level `camera_depth:` was removed; depth is "
            "served only by an explicitly declared `cameras:` entry named "
            "'depth' (declare the camera geometry explicitly -- the classic "
            "depth-encoder recipe lives in docs/pack_api.md)"
        )
    if "packs" in benchmark_cfg:
        raise ConfigError(
            f"{args.config}: benchmark.packs is removed; benchmark packs are "
            "declared on the PackRuntime 'benchmarks' face "
            "(provides: benchmarks: [{id: <id>, resource: <pack dir>}]) "
            "-- build the runtime and pass it to build_spec instead"
        )
    env_cfg = _require_config(cfg.get("env"), "env")
    col_cfg = _require_config(cfg.get("collect"), "collect")
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
    seed_value = args.seed if getattr(args, "seed", None) is not None else cfg.get("seed", 0)
    seed = _as_int(seed_value, "seed")

    # torch device follows --device (AppLauncher); the env backend also sets
    # cfg.sim.device so SimulationContext lands on the same GPU.
    device = args.device if getattr(args, "device", "cpu") != "cpu" else "cpu"
    # The run's content spec rides env.params into a backend with a content
    # router; the scene provider keys were routed by apply_scene_preset.
    env_cfg = _require_config(cfg.get("env"), "env")

    raw_backend_options = env_cfg.get("backend_options")
    if raw_backend_options is not None and not is_json_object(raw_backend_options):
        raise ConfigError(f"{args.config}: env.backend_options must be a mapping")
    try:
        backend_options = _require_config(
            validate_backend_options(backend_label, raw_backend_options), "backend options"
        )
    except ValueError as exc:
        raise ConfigError(f"{args.config}: {exc}") from exc
    _guard_scene_provider_support(env_cfg, backend_label)
    _route_content(cfg, env_cfg, backend_label)
    world_base = _validate_world_base(cfg, env_cfg, backend_label)
    articulations = _normalize_articulations(cfg, args.config)
    try:
        observations = ObservationConfig.from_mapping(cfg.get("observations"))
    except ObsConfigError as exc:
        raise ConfigError(f"{args.config}: {exc}") from exc
    action_spec, policies = _normalize_action_authoring(cfg, articulations, args.config)
    num_envs = _as_positive_int(env_cfg.get("num_envs"), "env.num_envs")
    total_steps = _as_positive_int(col_cfg.get("steps"), "collect.steps")
    # reset policy (docs/contracts/runner_contract.md §5): clock due by default;
    # reset_on_termination merges terminated rows.  runner.reset_mask_fn is
    # a reserved key (not deserialized here; inject via SyncRunner).
    runner_cfg = _require_config(cfg.get("runner") or {}, "runner")
    termination_mode = _parse_termination_mode(cfg, runner_cfg, args.config)
    _validate_episode_clock(env_cfg, termination_mode, args.config)
    output_root_value = cfg.get("output_root", "data_out")
    output_root = output_root_value if isinstance(output_root_value, str) else "data_out"
    run_name = _require_str(cfg.get("run_name"), "run_name")
    requested_out = getattr(args, "output_dir", None)
    if requested_out is None:
        requested_out = os.path.join(output_root, run_name)
    director_value = cfg.get("director")
    director = None if director_value is None else _require_config(director_value, "director")
    adapter_name_value = adapter_cfg.get("name")
    adapter_name = None if adapter_name_value is None else str(adapter_name_value)
    return RunSpec(
        cfg=cfg,
        config_path=args.config,
        run_name=run_name,
        seed=seed,
        mode=mode,
        backend=backend_label,
        backend_params=_require_config(env_cfg.get("params") or {}, "env.params"),
        backend_options=backend_options,
        world_base=world_base,
        articulations=articulations,
        action_spec=action_spec,
        policies=policies,
        adapter=adapter_cfg,
        adapter_name=adapter_name,
        adapter_params=adapter_params,
        adapter_overrides=tuple(adapter_overrides),
        props=cfg.get("props"),
        cameras=cfg.get("cameras"),
        contacts=cfg.get("contacts"),
        observations=observations,
        director=director,
        recorder=_require_config(cfg.get("recorder") or {}, "recorder"),
        bridges={
            "serve_camera": getattr(args, "serve_camera", None),
            "serve_frame_mb": getattr(args, "serve_frame_mb", 4.0),
            "serve_state_dir": getattr(args, "serve_state_dir", None),
            "run_log": getattr(args, "run_log", None),
            "serve_pause": bool(getattr(args, "serve_pause", False)),
        },
        benchmark=benchmark_cfg,
        terminations=cfg.get("terminations"),
        termination_mode=termination_mode,
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
        pack_runtime=runtime,
        robot_profiles=robot_profiles,
    )
