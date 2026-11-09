"""build_instance: the ONE construction point for a run (v097 R1).

Turns a frozen :class:`~unigym.run.spec.RunSpec` into a live
:class:`~unigym.run.instance.RunInstance` -- robot profile + adapter,
prop/camera/contact managers, env compile, recorder/director/timeline/
bridge, sink composition, rewind providers and run metadata.  It never
spawns processes, writes pid markers or parses argv; the CLI/lifecycle owns
those (docs/designs/v097-run-construction-execution-split.md §6/§7).

Construction order is the pre-split ``collect_cli.main`` order, moved
verbatim: prop -> adapter -> camera -> contact -> env ->
writer/logger -> director/timeline -> sinks -> replay -> bridge -> rewind
providers.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard, TypedDict, runtime_checkable

import numpy as np

from unigym._typing import JsonValue, is_json_object
from unigym.backends.registry import BackendInfo, backend_info
from unigym.benchmarks import build_tasks
from unigym.env.backends import build_env_backend
from unigym.filters import build_filter_chain
from unigym.logger import DataCollectLogger
from unigym.recorder import build_recorder
from unigym.robots import build_robot_recipe
from unigym.runtime.controllers import (
    build_control_plan,
    build_control_runtimes,
)
from unigym.terminations import (
    build_termination_group,
)

from .instance import RunInstance, Sink
from .spec import ReplaySpec, RunSpec

if TYPE_CHECKING:
    from unigym.adapters.base import Adapter
    from unigym.cameras.manager import CameraManager
    from unigym.commands.base import CommandSampler
    from unigym.contacts.manager import ContactManager
    from unigym.director import Director
    from unigym.env.backends.base import EnvBackend
    from unigym.filters.base import EpisodeFilter
    from unigym.io.writer import ShardWriter
    from unigym.language.base import LanguageGenerator
    from unigym.props.base import PropManager
    from unigym.recorder import Recorder
    from unigym.robots.base import RobotRecipe
    from unigym.runner.sinks.benchmark import BenchmarkSink
    from unigym.runtime.bridge import Bridge
    from unigym.runtime.controllers import ControlRuntime
    from unigym.runtime.rewind_providers import RewindStateProvider
    from unigym.scenes.articulation import ArticulationSpec
    from unigym.scenes.terrain import Terrain
    from unigym.trace.replay import TraceReplay
    from unigym.watchers.timeline import EventLog, TimelineSink

    from .instance import RunEnv


@runtime_checkable
class _EnvKwargsBuilder(Protocol):
    """Backend ``env_kwargs`` builder (registry factory -> opaque callable)."""

    def __call__(
        self,
        env_cfg: dict[str, JsonValue],
        artifacts_dir: Path | None,
        world_base: JsonValue,
        spec: RunSpec,
        *,
        controls: Sequence[ControlRuntime] | None,
        recipes: Mapping[str, RobotRecipe] | None,
        backend_options: dict[str, JsonValue],
    ) -> dict[str, object]: ...


@runtime_checkable
class _TerrainBuilder(Protocol):
    """Backend terrain factory (registry factory -> opaque callable)."""

    def __call__(
        self,
        cfg: dict[str, JsonValue],
        num_envs: int,
        env_spacing: float,
        terrain: JsonValue | Terrain | None,
    ) -> object: ...


@runtime_checkable
class _ActuatorOverride(Protocol):
    """Backend actuator override applier (registry factory -> callable)."""

    def __call__(self, profile: RobotRecipe, section: JsonValue) -> None: ...


@runtime_checkable
class _EventsBuilder(Protocol):
    """Backend DR events factory (registry factory -> callable)."""

    def __call__(self, section: JsonValue) -> object: ...


class _RobotBundle(TypedDict):
    profile: RobotRecipe
    profiles: dict[str, RobotRecipe]
    terrain: object | None
    """Backend-built terrain object; opaque to the run layer."""
    prop_manager: PropManager | None
    adapter: Adapter
    generates_command: bool


class _SensingBundle(TypedDict):
    camera_manager: CameraManager | None
    contact_manager: ContactManager | None
    effective_lighting: JsonValue
    director_cfg: dict[str, JsonValue] | None
    events_cfg: object | None
    """Backend-built DR events object; opaque to the run layer."""
    episode_length_s: JsonValue
    recorder: Recorder


class _EnvBundle(TypedDict):
    env: RunEnv
    artifacts_dir: Path | None
    out_dir: str
    sampler: CommandSampler | None
    command_dim: int
    policy_dt: float
    language_gen: LanguageGenerator | None
    lang_interval: int


class _TaskMeta(TypedDict, total=False):
    run_name: str
    seed: int
    num_envs: int
    policy_dt: float
    articulations: dict[str, list[str]]
    has_command: bool
    backend: str
    joint_names: list[str]


class _SinkBundle(TypedDict):
    director: Director | None
    sink: Sink
    timeline: TimelineSink
    event_log: EventLog
    bench_sink: BenchmarkSink | None
    stop_when: Callable[[], bool] | None
    watch_specs: list[dict[str, JsonValue]] | None


class _WiringBundle(TypedDict):
    bridge: Bridge
    sink: Sink
    replay: TraceReplay | None
    rewind_providers: list[RewindStateProvider]
    stop_when: Callable[[], bool]


class _RunMeta(TypedDict, total=False):
    framework: str
    created: str
    run_name: str
    config_file: str
    config: dict[str, JsonValue]
    seed: int
    num_envs: int
    steps: int
    policy_dt: float | None
    episode_length_s: JsonValue
    episode_length_range: list[float]
    schema_version: int
    articulations: dict[str, dict[str, JsonValue]]
    command_dim: int
    obs_collect_overridden: bool
    body_parts: Mapping[str, JsonValue] | None
    language_generator: str | None
    language_interval_steps: int
    filters: JsonValue
    filter_active: bool
    terminations: JsonValue
    init_pose: JsonValue
    camera_enabled: bool
    director_enabled: bool
    watch_file: str | None
    overrides: list[dict[str, JsonValue]]
    label: str
    adapter_params: dict[str, JsonValue]
    adapter_overrides: list[str]
    obs_catalog: dict[str, JsonValue] | None


def _config_int(value: JsonValue) -> int:
    """Narrow a config scalar to an int (bool/int/float/str only)."""
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"expected an integer config value, got {value!r}")


def _config_float(value: JsonValue) -> float:
    """Narrow a config scalar to a float (bool/int/float/str only)."""
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected a numeric config value, got {value!r}")


def _config_list(value: JsonValue) -> list[JsonValue] | None:
    """Narrow a config value to a list section (None when absent)."""
    return value if isinstance(value, list) else None


def _config_dict(value: JsonValue) -> dict[str, JsonValue] | None:
    """Narrow a config value to a mapping section (None when absent)."""
    return value if isinstance(value, dict) else None


def _config_dict_list(value: JsonValue) -> list[dict[str, JsonValue]]:
    """Narrow a config value to a list of mapping sections (fail-fast)."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(
            f"expected a list of config sections, got {type(value).__name__}")
    out: list[dict[str, JsonValue]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(
                f"expected a config section mapping, got {type(item).__name__}")
        out.append(item)
    return out


def _config_str(value: JsonValue, default: str | None = None) -> str | None:
    """Narrow a config scalar to a string (default when absent/other)."""
    return value if isinstance(value, str) else default


def _config_bool(value: JsonValue, default: bool = False) -> bool:
    """Narrow a config scalar to a bool (default when absent/other)."""
    return value if isinstance(value, bool) else default


def _is_str_iterable(value: object) -> TypeGuard[Iterable[str]]:
    """True for a non-string iterable of names (catalog-style faces)."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _profile_body_parts(profile: RobotRecipe) -> dict[str, JsonValue] | None:
    """The recipe's body-parts mapping as a JSON object (None when absent)."""
    value: object = profile.body_parts
    return value if is_json_object(value) else None


def _load_watch_specs(path: str) -> list[dict[str, JsonValue]]:
    """Load a watcher pack file as JSON-object specs (fail-fast upstream)."""
    from unigym.watchers import load_watch_yaml

    raw: object = load_watch_yaml(path)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if is_json_object(item)]


def _catalog_keys(env: EnvBackend) -> list[str] | None:
    """The env's observation catalog keys (None when it publishes none)."""
    catalog: object = getattr(env, "obs_catalog", None)
    return [str(key) for key in catalog] if _is_str_iterable(catalog) else None


def _lighting_json(values: Mapping[str, float]) -> dict[str, JsonValue]:
    """Project a backend lighting default into the JSON config shape."""
    return {str(key): float(value) for key, value in values.items()}


def _policy_hz_of(adapter: Adapter) -> float | None:
    """The v069 wall render scheduler's clock.

    ``1 / adapter.control_dt`` (camera_contract §16); None when the adapter
    exposes no positive control period (probe doubles).
    """
    dt = float(adapter.control_dt or 0.0)
    return (1.0 / dt) if dt > 0 else None


def _adapter_self_command_dim(adapter: Adapter) -> int:
    """The adapter's self-generated command width (0 when undeclared)."""
    value = getattr(adapter, "self_command_dim", 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


def _body_controls(env: EnvBackend) -> list[ControlRuntime]:
    """The body control faces (one per articulation, action face == name)."""
    return [c for c in (env.controls or {}).values()
            if c.action_face == c.articulation]


def _body_control_for(env: EnvBackend, articulation: str) -> ControlRuntime:
    """The body control face of one articulation (or its first control)."""
    controls = list((env.controls or {}).values())
    for control in controls:
        if (control.articulation == articulation
                and control.action_face == articulation):
            return control
    for control in controls:
        if control.articulation == articulation:
            return control
    raise RuntimeError(
        f"articulation {articulation!r} has no control face")


def _build_adapter_groups(spec: RunSpec) -> dict[str, tuple[str, ...]]:
    """``{adapter_name: tuple[group_name, ...]}`` from ``RunSpec.controls``.

    This is the authoritative control partition (v104.3): the config's
    ``controls[]`` names exactly which control group each adapter drives.
    One group cannot be claimed by two adapters; recipe-derived extra faces
    and articulated props are NOT implicitly added here (an adapter that
    emits them must have them declared as control groups).
    """
    groups: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    for group in getattr(spec, "controls", ()) or ():
        name = str(group.name)
        owner = str(group.adapter)
        prev = owners.get(name)
        if prev is not None:
            raise ValueError(
                f"control group {name!r} is claimed by two adapters: "
                f"{prev!r} and {owner!r}")
        owners[name] = owner
        groups.setdefault(owner, []).append(name)
    return {owner: tuple(names) for owner, names in groups.items()}


def _announce_artifacts(path: str | Path | None) -> None:
    """Print the run-artifact destination (or the off switch)."""
    if path is None:
        print("[collect] run artifacts: off", flush=True)
    else:
        print(f"[collect] run artifacts: {path}", flush=True)


def _build_replay(spec: ReplaySpec, num_envs: int) -> TraceReplay | None:
    """Trace replay source for --replay (toolkit B; fail-fast at bind)."""
    if not spec.path:
        return None
    from unigym.trace.replay import ReplayError, TraceReplay
    try:
        replay = TraceReplay(spec.path, mode=spec.mode,
                             strict=spec.strict,
                             key=spec.key)
        info: dict[str, JsonValue] = replay.bind(num_envs)
        if spec.freeze_rng:
            frozen: dict[str, JsonValue] = replay.freeze_rng()
            print(f"[replay] rng frozen: faces={frozen['faces']} "
                  f"seed={frozen['seed']}", flush=True)
        print(f"[replay] bound: mode={info['mode']} every={info['every']} "
              f"coverage={info['coverage']} ticks raw_key={info['raw_key']}",
              flush=True)
        return replay
    except ReplayError as e:
        raise SystemExit(f"[replay] error: {e} | hint: {e.hint}") from e


def _recipe_section(art: ArticulationSpec) -> dict[str, JsonValue]:
    """Project one articulation into the recipe-builder section (v104)."""
    section: dict[str, JsonValue] = {"name": art.name}
    if art.params:
        section["params"] = dict(art.params)
    if art.body_parts:
        section["body_parts"] = dict(art.body_parts)
    return section


def _profile_for(art: ArticulationSpec, with_isaaclab: bool) -> RobotRecipe:
    """Build the :class:`RobotRecipe` of one robot articulation.

    ``params.profile`` is the reserved selector for runs whose articulation
    name is not the robot type (e.g. a second ``g1b`` articulation reuses
    the ``g1`` recipe); it never reaches the recipe factory.
    """
    section = _recipe_section(art)
    raw_params = section.get("params")
    params: dict[str, JsonValue] = (
        dict(raw_params) if isinstance(raw_params, dict) else {})
    selector = params.pop("profile", None)
    if selector is not None:
        section["params"] = params
        section["name"] = str(selector)
    return build_robot_recipe(
        section, with_isaaclab=with_isaaclab)


def _control_articulations(spec: RunSpec) -> list[ArticulationSpec]:
    """The articulation specs named by the run's controls[] (declaration order)."""
    names: list[str] = []
    for group in spec.controls:
        if group.articulation not in names:
            names.append(group.articulation)
    by_name = {art.name: art for art in spec.articulations}
    return [by_name[name] for name in names if name in by_name]


def _control_recipes(
        spec: RunSpec, info: BackendInfo) -> dict[str, RobotRecipe]:
    """Build the RobotRecipe of every recipe-backed control articulation."""
    recipes: dict[str, RobotRecipe] = {}
    for art in _control_articulations(spec):
        groups = [g for g in spec.controls if g.articulation == art.name]
        try:
            recipes[art.name] = _profile_for(
                art, info.recipe_with_isaaclab)
        except (AttributeError, KeyError, NotImplementedError, ValueError):
            # recipe-less articulations are legal only when every control
            # declares its joints/q0 explicitly (e.g. a future prop face);
            # a `joint_set` selector REQUIRES a recipe (v105).
            if any(g.joint_set is not None for g in groups):
                raise ValueError(
                    f"control articulation {art.name!r} has no RobotRecipe "
                    f"(unknown robot type) and at least one control group "
                    f"selects a `joint_set` | hint: use a registered robot "
                    f"articulation or declare every group's explicit "
                    f"`joints` + `q0`"
                ) from None
    return recipes


def _build_robot_and_props(
        spec: RunSpec, cfg: dict[str, JsonValue], info: BackendInfo,
        device: str, num_envs: int, env_spacing: float,
        config_dir: str) -> _RobotBundle:
    """Step 1: policy profile(s) + actuator/terrain/prop + adapter."""
    backend_label = info.name
    # backends that build an ArticulationCfg declare it on their
    # BackendInfo face; the others take the pure-constants profile
    # (articulation_cfg None).
    profiles = _control_recipes(spec, info)
    if not profiles:
        raise ValueError(
            "no recipe-backed control articulation: declare a top-level "
            "`controls:` group over a registered robot articulation "
            "(v104 authoring shape) | hint: e.g. controls: [{name: g1, "
            "adapter: <name>, articulation: g1, drive: position_pd, "
            "target: position}]")
    first_name = next(iter(profiles))
    profile = profiles[first_name]
    # Pluggable actuators: the yaml top-level "actuator" field overrides
    # the profile default (default implicit_pd).
    # The mujoco backend has no articulation_cfg: gains go through
    # env.params (PD lives inside the backend) -- reject explicitly.
    # The pluggable ctrl laws on the mujoco side go through
    # env.params.actuators (the unigym/actuators top-level package,
    # v063 D2/D9: standard_pd / computed_torque_pd / native_servo) --
    # the top-level "actuator" key semantics unchanged (isaaclab-only).
    if "actuator" in cfg:
        if info.build_actuator_override is None \
                or profile.articulation_cfg is None:
            raise ValueError(
                "yaml top-level 'actuator' override is not supported by the "
                f"'{backend_label}' backend; configure gains via env.params")
        override = info.build_actuator_override()
        if not isinstance(override, _ActuatorOverride):
            raise TypeError("backend actuator override is not callable")
        override(profile, cfg["actuator"])
    # terrain preset is a backend construction face; backends without one
    # serve their own ground (e.g. the floor geom of the MJ XML scene).  A
    # programmatic ``Run(world=World.terrain...)`` carries the native
    # Terrain IR on the spec and the backend builds the preset itself.
    terrain: object | None = None
    if info.build_terrain is not None:
        terrain_factory = info.build_terrain()
        if not isinstance(terrain_factory, _TerrainBuilder):
            raise TypeError("backend terrain factory is not callable")
        terrain = terrain_factory(cfg, num_envs, env_spacing, spec.terrain)
    from unigym.adapters import build_adapter
    from unigym.props import build_prop_manager

    prop_manager = build_prop_manager(
        _config_dict(cfg.get("props")), num_envs=num_envs,
        env_spacing=env_spacing, device=device, base_dir=config_dir,
        backend=backend_label)
    if prop_manager is not None:
        print(f"[collect] props enabled: "
              f"{[s.name for s in prop_manager.specs]}", flush=True)
    adapter_section = _config_dict(cfg.get("adapter"))
    if adapter_section is None:
        raise TypeError("run config 'adapter' section must be a mapping")
    adapter = build_adapter(adapter_section, num_envs=num_envs, device=device,
                            body_parts=_profile_body_parts(profile))
    # P9d: hierarchical adapters generate their own command each step ->
    # no command sampler; the self-generated command is recorded instead.
    generates_command = bool(adapter.generates_command)
    return {"profile": profile, "profiles": profiles, "terrain": terrain,
            "prop_manager": prop_manager, "adapter": adapter,
            "generates_command": generates_command}


def _build_sensing(
        spec: RunSpec, cfg: dict[str, JsonValue], info: BackendInfo,
        device: str, num_envs: int, profile: RobotRecipe, adapter: Adapter,
        prop_manager: PropManager | None) -> _SensingBundle:
    """Step 2: camera/contact managers + lighting/director/events/recorder."""
    backend_label = info.name
    # feat/camera-obs: request-driven camera manager, built BEFORE the env
    # (sensors must join the scene cfg prior to cloning; also feeds the
    # backend capability gate).  Requires --enable_cameras (kit camera
    # pipeline); depth alias obs keys get the implicit sugar-shim camera
    # (yaml ``camera_depth`` = verbatim override params for that camera).
    # An enabled recorder's camera_ref is passed as a ref so a video-only
    # rig (not consumed by the adapter) is instantiated too.
    from unigym.cameras import build_camera_manager
    from unigym.contacts import build_contact_manager

    rec_cfg = _config_dict(cfg.get("recorder")) or {}
    rec_ref = rec_cfg.get("camera_ref") if rec_cfg.get("enabled") else None
    camera_manager = build_camera_manager(
        cfg.get("cameras"), adapter.get_obs_requests(),
        enable_cameras=spec.enable_cameras,
        refs=[rec_ref] if rec_ref else None,
        depth_overrides=_config_dict(cfg.get("camera_depth")),
        backend=backend_label,
        policy_hz=_policy_hz_of(adapter),
        num_envs=num_envs)
    if camera_manager is not None:
        print(f"[collect] cameras enabled: {camera_manager.names} "
              f"(keys: {sorted(str(k) for k in camera_manager.camera_keys())})",
              flush=True)
    # feat/contact-manager: request-driven contact manager, built BEFORE the
    # env (its ContactSensorCfgs must join the scene cfg prior to cloning on
    # isaaclab; docs/contracts/contacts_contract.md).  No ``contacts:`` section ->
    # None -> zero code paths in the env (baseline protocol).
    # stage 2: contact groups are FULLY declarative -- sensor_bodies
    # may reference a robot body-parts group (``{body_parts: grasp.left}``)
    # and filter_bodies a prop (``{prop: <name>}``); resolution happens
    # inside build_contact_manager against the profile body_parts + the
    # declared prop names (the auto held groups are retired).
    contact_manager = build_contact_manager(
        cfg.get("contacts"), backend=backend_label, device=device,
        adapter_requests={r.key for r in adapter.get_obs_requests()},
        body_parts=_profile_body_parts(profile),
        prop_names=([s.name for s in prop_manager.specs]
                    if prop_manager is not None else None),
        articulated_props=({s.name for s in prop_manager.specs
                            if s.type == "articulation"}
                           if prop_manager is not None else None))
    # default lights: a textured usd prop on isaaclab without scene
    # fill light renders as a black silhouette -- when the config declares
    # NO env.lighting and a usd prop is present, enable dome + distant by
    # default (an explicit env.lighting section always wins verbatim;
    # dome_intensity: 0 disables the fill light).
    env_section = _config_dict(cfg.get("env")) or {}
    effective_lighting: JsonValue = env_section.get("lighting")
    if effective_lighting is None and info.default_lighting is not None \
            and prop_manager is not None and prop_manager.has_usd_props:
        effective_lighting = _lighting_json(info.default_lighting)
        print("[collect] usd prop default lights: dome 600 + distant "
              "(override via env.lighting)", flush=True)
    if contact_manager is not None:
        print(f"[collect] contacts enabled: "
              f"{[s.name for s in contact_manager.specs]} "
              f"(keys: {sorted(contact_manager.keys())})", flush=True)
    # director mode : the --direct flag is the ONLY opt-in face.
    # director_cfg rides into the env builders so the mujoco spec-side
    # injection (gripper mocap body + weld slots) happens pre-compile;
    # without --direct it stays None and the model is byte-identical.
    director_cfg: dict[str, JsonValue] | None = None
    if spec.direct:
        director_cfg = {"enabled": True}
        gripper = (_config_dict(cfg.get("director")) or {}).get("gripper")
        if gripper:
            director_cfg["gripper"] = gripper
        print("[collect] director mode ON (--direct): direct.* commands "
              "enabled; episodes touched by commands are marked directed",
              flush=True)
    events_cfg: object | None = None
    randomization = _config_dict(cfg.get("randomization")) or {}
    events_section = randomization.get("events")
    if events_section:
        if info.build_events is None:
            raise ValueError(
                f"the '{backend_label}' backend does not implement "
                "randomization.events")
        events_factory = info.build_events()
        if not isinstance(events_factory, _EventsBuilder):
            raise TypeError("backend events factory is not callable")
        events_cfg = events_factory(events_section)
    episode_length_s = env_section.get("episode_length_s", 10.0)
    # the recorder binds an explicitly declared in-scene camera
    # only (recorder.camera_ref REQUIRED; no self-built camera path).  The
    # camera_manager MUST be built before this call (above).
    recorder = build_recorder(rec_cfg, num_envs=num_envs,
                              camera_manager=camera_manager)
    if recorder.enabled:
        print(f"[collect] camera recording enabled "
              f"(ref {recorder.camera_ref}, env {recorder.record_env}, "
              f"interval {recorder.record_interval})", flush=True)
    return {"camera_manager": camera_manager,
            "contact_manager": contact_manager,
            "effective_lighting": effective_lighting,
            "director_cfg": director_cfg, "events_cfg": events_cfg,
            "episode_length_s": episode_length_s, "recorder": recorder}


def _build_env_and_sampler(
        spec: RunSpec, cfg: dict[str, JsonValue],
        env_cfg: dict[str, JsonValue], info: BackendInfo, device: str,
        num_envs: int, profile: RobotRecipe, adapter: Adapter,
        prop_manager: PropManager | None, terrain: object | None,
        camera_manager: CameraManager | None,
        contact_manager: ContactManager | None,
        effective_lighting: JsonValue,
        director_cfg: dict[str, JsonValue] | None,
        events_cfg: object | None, recorder: Recorder,
        episode_length_s: JsonValue,
        controls: Sequence[ControlRuntime] | None = None,
        recipes: Mapping[str, RobotRecipe] | None = None) -> _EnvBundle:
    """Step 3: env compile + sampler/language wiring."""
    # the run output dir resolves BEFORE the backend build: the recorded
    # scene artifacts (design v091 D14) are written during env construction
    # and land under <run output dir>/scene/.  The daemon drops the run.json
    # ledger (as run.json.pending-<mypid>) into out_dir BEFORE the worker
    # starts, so "exists"/"non-empty" is no longer the collision signal --
    # a dir holding ONLY our pending ledger is OURS (the daemon created it
    # moments ago for this very run); any real data (shards/meta/summary/
    # videos or a previous run's ledger) rotates to the timestamped sibling.
    # The pid tag means a shared-name spawn can never clobber the previous
    # run's ledger.
    out_dir = spec.output.out_dir
    if out_dir is None:
        raise RuntimeError("run output directory is unresolved")
    artifacts_dir = (Path(spec.output.artifacts_dir)
                     if spec.output.artifacts_dir is not None else None)
    env_kwargs_factory = info.env_kwargs
    if env_kwargs_factory is None:
        raise TypeError(
            f"backend {info.name!r} declares no env_kwargs factory")
    env_kwargs_builder = env_kwargs_factory()
    if not isinstance(env_kwargs_builder, _EnvKwargsBuilder):
        raise TypeError("backend env_kwargs builder is not callable")
    extra_kwargs = env_kwargs_builder(
        env_cfg, artifacts_dir, spec.world_base, spec,
        controls=controls, recipes=recipes,
        backend_options=spec.backend_options)
    env: RunEnv = build_env_backend(
        info.name,
        robot_profile=profile,
        adapter=adapter,
        terrain_cfg=terrain,
        num_envs=num_envs,
        episode_length_s=episode_length_s,
        env_spacing=_config_float(env_cfg.get("env_spacing", 4.0)),
        decimation=_config_int(env_cfg.get("decimation", 4)),
        sim_dt=_config_float(env_cfg.get("sim_dt", 0.005)),
        events_cfg=events_cfg,
        recorder=recorder,
        prop_manager=prop_manager,
        lighting=effective_lighting,
        init_pose=cfg.get("init_pose"),
        camera_manager=camera_manager,
        contact_manager=contact_manager,
        director_cfg=director_cfg,
        device=device,
        # backend-specific params (yaml env.params; mujoco: spawn_height /
        # contact_force_threshold / kp / kd).  The run-level world_base is
        # an explicit kwarg (v101 A1), never an env.params key.  isaaclab's
        # IsaaclabEnv does not accept this key -> isaaclab-only parity.
        # The artifact dir is an explicit mujoco kwarg (not an
        # env.params key): it is run wiring, never scene configuration.
        **extra_kwargs,
    )
    _announce_artifacts(artifacts_dir)
    from unigym.commands import build_sampler
    from unigym.language import build_generator

    command_section = _config_dict(cfg.get("command"))
    if command_section is None:
        raise TypeError("run config 'command' section must be a mapping")
    sampler = None if bool(adapter.generates_command) \
        else build_sampler(command_section, num_envs=num_envs, device=device)
    command_dim = (sampler.dim if sampler is not None
                   else _adapter_self_command_dim(adapter))
    policy_dt = adapter.control_dt
    language_section = _config_dict(cfg.get("language"))
    language_gen = build_generator(language_section)
    lang_interval = _config_int(
        (language_section or {}).get("interval_steps", 1))
    if language_gen is not None:
        print(f"[collect] language generator: {type(language_gen).__name__} "
              f"(interval={lang_interval})", flush=True)
    return {"env": env, "artifacts_dir": artifacts_dir, "out_dir": out_dir,
            "sampler": sampler, "command_dim": command_dim,
            "policy_dt": policy_dt, "language_gen": language_gen,
            "lang_interval": lang_interval}


def _apply_benchmark_budget(
        cfg: dict[str, JsonValue], mode: str, env: EnvBackend,
        sampler: CommandSampler | None, generates_command: bool, seed: int,
        num_envs: int, policy_dt: float, backend_label: str,
        total_steps: int
) -> tuple[int, dict[str, JsonValue], _TaskMeta | None]:
    """Step 4: benchmark max_steps cap + task metadata."""
    # benchmark wiring (mode: benchmark | both).  Step budget: the benchmark
    # max_steps caps collect.steps (docs/contracts/benchmark_contract.md); the collect
    # path (mode absent / "collect") never enters this branch.
    bench_cfg = _config_dict(cfg.get("benchmark")) or {}
    task_meta: _TaskMeta | None = None
    if mode in ("benchmark", "both"):
        max_steps = _config_int(bench_cfg.get("max_steps", 100000))
        if max_steps < total_steps:
            total_steps = max_steps
            collect_section = _config_dict(cfg.get("collect"))
            if collect_section is None:
                raise TypeError("run config 'collect' section must be a mapping")
            collect_section["steps"] = total_steps
        runtimes = _body_controls(env)
        task_meta = {
            "run_name": _config_str(cfg.get("run_name")) or "",
            "seed": seed,
            "num_envs": num_envs,
            "policy_dt": policy_dt,
            "articulations": {
                rt.articulation: list(rt.joint_names) for rt in runtimes},
            "has_command": bool(sampler is not None or generates_command),
            "backend": backend_label,     # results.v1 face
        }
        if len(runtimes) == 1:
            task_meta["joint_names"] = list(runtimes[0].joint_names)
        task_names = [
            t if isinstance(t, str)
            else (_config_str(t.get("name")) if isinstance(t, dict) else None)
            for t in (_config_list(bench_cfg.get("tasks")) or [])]
        print(f"[benchmark] mode={mode} max_steps={max_steps} "
              f"min_episodes={_config_int(bench_cfg.get('min_episodes', 1))} "
              f"tasks={task_names}", flush=True)
    return total_steps, bench_cfg, task_meta


def _build_sinks(
        spec: RunSpec, cfg: dict[str, JsonValue], mode: str, env: RunEnv,
        adapter: Adapter, writer: ShardWriter, logger: DataCollectLogger,
        filter_chain: list[EpisodeFilter],
        language_gen: LanguageGenerator | None, lang_interval: int,
        recorder: Recorder, num_envs: int, total_steps: int, device: str,
        policy_dt: float, out_dir: str, seed: int,
        episode_length_s: JsonValue, dur_lo: float, dur_hi: float,
        command_dim: int, profile: RobotRecipe,
        bench_cfg: dict[str, JsonValue], task_meta: _TaskMeta | None,
        contact_manager: ContactManager | None) -> _SinkBundle:
    """Step 5: director + timeline + mode sink composition."""
    from unigym.runner.sinks import (
        BenchmarkSink, CollectSink, FanOutSink)
    # director (before the sinks so BenchmarkSink can label
    # episodes directed): status_fn wires in after the bridge exists
    director: Director | None = None
    if spec.direct:
        from unigym.backends.registry import backend_info, build_name
        from unigym.director import Director
        info = backend_info(build_name(spec.backend))
        director = Director(
            env=env, contact_manager=contact_manager,
            cfg=_config_dict(cfg.get("director")) or {}, out_dir=out_dir,
            director_ops=info.director_ops)

    # timeline (§3): the unified events.jsonl stream + shared writer
    # (read-only observers -- zero perturbation of recording/physics)
    from unigym.watchers import EventLog, TimelineSink
    watch_specs: list[dict[str, JsonValue]] | None = None
    if spec.watch:
        watch_specs = _load_watch_specs(spec.watch)   # fail-fast, pre-mount
        print(f"[collect] watchers born-armed: {len(watch_specs)} "
              f"from {spec.watch}", flush=True)
    event_log = EventLog(out_dir)
    timeline = TimelineSink(
        event_log,
        contact_keys=(contact_manager.keys() if contact_manager is not None
                      else []))

    # sink selection by yaml ``mode`` (collect = unchanged default path)
    collect_sink: CollectSink | None = None
    if mode != "benchmark":
        collect_sink = CollectSink(
            writer=writer,
            logger=logger,
            filter_chain=filter_chain,
            language_gen=language_gen,
            lang_interval=lang_interval,
            recorder=recorder,
            adapter=adapter,
            env=env,
            num_envs=num_envs,
            total_steps=total_steps,
            device=device,
            policy_dt=policy_dt,
            out_dir=out_dir,
            seed=seed,
            episode_length_s=episode_length_s,
            dur_lo=dur_lo,
            dur_hi=dur_hi,
            command_dim=command_dim,
            body_parts=_profile_body_parts(profile),
        )
    bench_sink: BenchmarkSink | None = None
    stop_when: Callable[[], bool] | None = None
    if mode in ("benchmark", "both"):
        if task_meta is None:
            raise RuntimeError("benchmark mode requires task metadata")
        # benchmark.packs: extra pack dirs (docs/contracts/benchmark_contract.md §10);
        # builtin registry + <repo>/benchmark_packs are always discovered
        bench_tasks = build_tasks(
            _config_dict_list(bench_cfg.get("tasks")), dict(task_meta),
            packs_cfg=_config_list(bench_cfg.get("packs")))
        bench_sink = BenchmarkSink(
            tasks=bench_tasks,
            meta=dict(task_meta),
            min_episodes=_config_int(bench_cfg.get("min_episodes", 1)),
            max_steps=_config_int(bench_cfg.get("max_steps", 100000)),
            report_path=os.path.join(out_dir, "benchmark_report.json"),
            director=director,
            # benchmark.retain_metric_frames -> metrics.npz
            retain_frames=bool(bench_cfg.get("retain_metric_frames", False)))
        if bench_tasks:  # no tasks -> nothing to wait for
            stop_when = bench_sink.episodes_done
    sink: Sink | None
    if mode == "benchmark":
        sink = bench_sink
    elif mode == "both":
        # CollectSink first, BenchmarkSink second (owner decision 2026-09-03)
        sink = FanOutSink(collect_sink, bench_sink)
    else:
        sink = collect_sink
    # adapter-provided diagnostics sinks (generic hook, engine stays
    # adapter-agnostic): mounted AHEAD of the standard chain so extra keys
    # they write land before the standard sinks buffer the same step
    for extra_sink in adapter.extra_sinks(env, out_dir):
        sink = extra_sink if sink is None else FanOutSink(extra_sink, sink)
    if sink is None:
        raise RuntimeError(f"no sink selected for run mode {mode!r}")
    return {"director": director, "sink": sink, "timeline": timeline,
            "event_log": event_log, "bench_sink": bench_sink,
            "stop_when": stop_when, "watch_specs": watch_specs}


def _build_bridge_and_wiring(
        spec: RunSpec, cfg: dict[str, JsonValue], backend_label: str,
        env: RunEnv, adapter: Adapter,
        camera_manager: CameraManager | None, out_dir: str,
        bench_sink: BenchmarkSink | None, sink: Sink,
        timeline: TimelineSink, director: Director | None,
        event_log: EventLog, watch_specs: list[dict[str, JsonValue]] | None,
        stop_when: Callable[[], bool] | None, num_envs: int) -> _WiringBundle:
    """Step 6: replay + bridge + runtime attach + rewind providers."""
    from unigym.runner.sinks import FanOutSink
    # trace replay source (toolkit B): fail-fast BEFORE the bridge/worker
    # mounts so a bad trace can never leak a zombie SHM segment
    replay = _build_replay(spec.replay, num_envs)
    # runtime worker (run-slimming): EVERY run mounts the SHM
    # worker (status + interval frame publish + command slot) -- engine
    # zero-change, shutdown arrives as bridge.shutdown_requested during
    # sink.on_step and is honoured through the engine's existing stop_when
    # seam (normal break -> on_run_end teardown, no exception plumbing).
    from unigym.runtime.bridge import Bridge, make_run_id
    bridge = Bridge(
        make_run_id(spec.run_name), camera_manager,
        camera=_config_str(spec.bridges["serve_camera"]),
        frame_capacity=int(
            _config_float(spec.bridges["serve_frame_mb"]) * 1024 * 1024),
        config=os.path.abspath(spec.config_path), out_dir=out_dir,
        state_dir=_config_str(spec.bridges["serve_state_dir"]),
        obs_catalog=_catalog_keys(env),
        paused=_config_bool(spec.bridges["serve_pause"]),
        log_path=_config_str(spec.bridges["run_log"]),
        # typed queries: effective config snapshot (startup
        # fail-fast on oversize) + benchmark telemetry feed (report draft,
        # episode rings; None in collect-only runs)
        config_doc=cfg, metrics_source=bench_sink)
    # runtime debug console: step/inspect/save_npz handlers on
    # the bridge dispatch table (docs/contracts/console_contract.md)
    from unigym.console import attach as attach_console
    attach_console(bridge, env=env, adapter=adapter, out_dir=out_dir)
    # scene face: cmd-16 typed query -- curated scene tree
    # (lazy cache) + live state (per-query sampling); zero engine writes
    from unigym.runtime.scene import attach as attach_scene
    robot_name = next(iter(getattr(env, "articulations", {}) or {}), None)
    attach_scene(bridge, env, backend_label, robot_name=robot_name)
    # watchers: cmd-15 handler + per-tick hook (before the step gate);
    # born-armed specs came from --watch (fail-fast parsed above).
    # v068: watcher artifacts live under <runtime_dir>/watch/<run_id>/
    # (watch/ + watchers.json; design §2) -- events.jsonl stays in the
    # run's output dir (the daemon tail-face reads it there).
    from unigym import runtime_dirs
    from unigym.watchers import attach as attach_watchers
    attach_watchers(bridge, env=env, adapter=adapter,
                    out_dir=runtime_dirs.watch_dir(bridge.run_id),
                    event_log=event_log, specs=watch_specs)
    # director: command handler on cmd 14 + live status feed
    if director is not None:
        from unigym.director import attach as attach_director
        director.status_fn = bridge.read_status
        attach_director(bridge, director)
    else:
        # guard gate (contract §0.2/§6): no --direct -> every direct.*
        # command is rejected fail-fast (the attempt still lands in the
        # shared command log)
        from unigym.director import Rejector
        log = (Path(out_dir) / "director" / "commands.jsonl"
               if out_dir else None)
        bridge.register_command("direct", Rejector(log).handle)
    if director is not None:
        sink = FanOutSink(sink, director)      # sinks first, then director
    sink = FanOutSink(sink, timeline)          # timeline, then the bridge
    sink = FanOutSink(sink, bridge)
    base_stop = stop_when

    def _stop_when() -> bool:
        return bool(bridge.shutdown_requested) or (
            base_stop() if base_stop is not None else False)
    stop_when = _stop_when

    # rewind face (Round B B1): the providers are constructed here; the
    # controller attach needs the engine (built by the CLI after this).
    # v057 providers: the sink facilities' open-episode state joins every
    # checkpoint payload and is restored fail-fast with the engine+adapter
    from unigym.runtime.rewind_providers import build_rewind_providers
    initializer: object | None = getattr(env, "_initializer", None)
    rewind_providers = build_rewind_providers(
        sink, initializer=initializer)
    # freeze face (v056 freeze paradigm): skill halt -> world freeze.
    # Engine-thread park hook for adapters that declare it (agent_g1);
    # a no-op mount for every other adapter.
    from unigym.runtime.freeze import attach as attach_freeze
    attach_freeze(bridge, adapter)
    # born-paused truth: a --serve_pause run parks via the gate at
    # construction -- adopt that park as the time proxy's command
    # truth so /health.paused is honest from tick 0 (v056)
    set_paused: object = getattr(adapter, "set_time_truth_paused", None)
    if _config_bool(spec.bridges["serve_pause"]) and callable(set_paused):
        set_paused(True)
    return {"bridge": bridge, "sink": sink, "replay": replay,
            "rewind_providers": rewind_providers, "stop_when": stop_when}


def build_instance(spec: RunSpec) -> RunInstance:
    """Construct every live resource of one run from ``spec`` (no driving)."""
    cfg = spec.cfg
    backend_label = spec.backend
    info = backend_info(backend_label)
    mode = spec.mode
    seed = spec.seed
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = spec.device
    env_value = cfg["env"]
    env_cfg: dict[str, JsonValue] = (
        env_value if isinstance(env_value, dict) else {})
    col_value = cfg["collect"]
    col_cfg: dict[str, JsonValue] = (
        col_value if isinstance(col_value, dict) else {})
    num_envs = _config_int(env_cfg["num_envs"])
    total_steps = _config_int(col_cfg["steps"])
    # reset policy (docs/contracts/runner_contract.md §5): clock due by default;
    # reset_on_termination merges terminated rows.  runner.reset_mask_fn is
    # a reserved key (not deserialized here; inject via SyncRunner).
    reset_on_termination = bool(spec.runner.get("reset_on_termination", False))
    env_spacing = _config_float(env_cfg.get("env_spacing", 4.0))
    config_dir = os.path.dirname(os.path.abspath(spec.config_path))

    robot = _build_robot_and_props(
        spec, cfg, info, device, num_envs, env_spacing, config_dir)
    profile = robot["profile"]
    profiles = robot["profiles"]
    adapter = robot["adapter"]
    prop_manager = robot["prop_manager"]
    generates_command = robot["generates_command"]
    # v104.2/v105.2: RunSpec.controls is authoritative.  Build one
    # ControlRuntime per declared group by expanding `joint_set` from the
    # recipe (or using explicit `joints`) -- once.  The same resolved
    # controls freeze into the read-only plan; no guessing path remains.
    control_runtimes = build_control_runtimes(
        spec.controls, spec.articulations, profiles)
    control_plan = build_control_plan(spec.controls, control_runtimes)
    sensing = _build_sensing(
        spec, cfg, info, device, num_envs, profile, adapter,
        prop_manager)
    built = _build_env_and_sampler(
        spec, cfg, env_cfg, info, device, num_envs, profile, adapter,
        prop_manager, robot["terrain"],
        sensing["camera_manager"], sensing["contact_manager"],
        sensing["effective_lighting"], sensing["director_cfg"],
        sensing["events_cfg"], sensing["recorder"],
        sensing["episode_length_s"],
        controls=control_runtimes, recipes=profiles)
    env = built["env"]
    env.adapter_groups = _build_adapter_groups(spec)
    if spec.adapter_name:
        adapter._unigym_name = spec.adapter_name
        # articulated-prop action faces are declared by ``objects:`` sugar and
        # materialized by the backend as prop control runtimes; the emitting
        # adapter owns them (the config's controls[] cannot name them).
        prop_faces = tuple(c.name for c in (env.controls or {}).values()
                           if c.prop)
        if prop_faces:
            owned = tuple(env.adapter_groups.get(spec.adapter_name, ()))
            env.adapter_groups[spec.adapter_name] = owned + prop_faces
    sampler = built["sampler"]
    policy_dt = built["policy_dt"]
    total_steps, bench_cfg, task_meta = _apply_benchmark_budget(
        cfg, mode, env, sampler, generates_command, seed, num_envs, policy_dt,
        backend_label, total_steps)

    # pluggable filter chain (episode-level)
    articulation_names = list(env.articulations or {})
    filter_chain = build_filter_chain(
        _config_list(cfg.get("filters")), articulations=articulation_names)
    filter_active = len(filter_chain) > 0
    if filter_active:
        names = [type(f).__name__ for f in filter_chain]
        print(f"[collect] filter chain ({len(filter_chain)}): {names}", flush=True)

    # P10: unified termination checks (detection-only; episodes still end by
    # duration).  v106.1: terminations are explicit; no default fall group is
    # synthesised when the config omits the section.
    term_group = build_termination_group(_config_list(cfg.get("terminations")))
    print(f"[collect] terminations: {term_group.names} "
          f"(mode={term_group.mode})", flush=True)

    # label single-writer (designs/v0611): the run process stamps its OWN
    # ledger with the display label (pid-aware, no-op without a ledger
    # -- CLI runs keep the label in meta.json only); failures are
    # swallowed by the mutation face (bookkeeping never kills a run).
    if spec.label:
        from unigym.daemon import runrec as _runrec
        _runrec.set_label(built["out_dir"], os.getpid(), spec.label)
    episode_length_s = sensing["episode_length_s"]
    from unigym.adapters.base import Adapter
    from unigym.env.episode import parse_episode_length

    dur_lo, dur_hi = parse_episode_length(episode_length_s)

    meta: _RunMeta = {
        "framework": "unigym",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": spec.run_name,
        "config_file": os.path.abspath(spec.config_path),
        "config": cfg,
        "seed": seed,
        "num_envs": num_envs,
        "steps": total_steps,
        "policy_dt": policy_dt,
        "episode_length_s": episode_length_s,
        "episode_length_range": [dur_lo, dur_hi],
        # v3 data schema: per-articulation orders replace the flat
        # joint_names_policy_order / foot_names_policy_order keys
        "schema_version": 3,
        "articulations": {
            name: {
                "joint_names": list(control.joint_names),
                "foot_names": list(art.feet),
                "asset_joint_names": list(art.joint_names),
                "action_face": control.action_face,
            }
            for name, art in (env.articulations or {}).items()
            for control in [_body_control_for(env, name)]
        },
        "command_dim": built["command_dim"],
        "obs_collect_overridden": (
            type(adapter).compute_obs_collect
            is not Adapter.compute_obs_collect),
        "body_parts": _profile_body_parts(profile),
        "language_generator": (type(built["language_gen"]).__name__
                               if built["language_gen"] is not None else None),
        "language_interval_steps": built["lang_interval"],
        "filters": cfg.get("filters"),
        "filter_active": filter_active,
        "terminations": cfg.get("terminations"),
        "init_pose": cfg.get("init_pose"),
        "camera_enabled": sensing["recorder"].enabled,
        # director: static startup face (live state lives in
        # director/status.json + commands.jsonl, contract §7)
        "director_enabled": bool(spec.direct),
        # watchers: the born-armed pack (reproducibility, §8.6)
        "watch_file": (os.path.abspath(spec.watch)
                       if spec.watch else None),
    }
    # effective diff (reproducibility; key ABSENT when no
    # --set so old runs' meta.json stays byte-identical)
    if spec.overrides:
        meta["overrides"] = list(spec.overrides)
    # display label (key ABSENT when no --label: old meta.json stays
    # byte-identical -- same convention as overrides/adapter_params)
    if spec.label:
        meta["label"] = spec.label
    # spawn adapter params bookkeeping (audit chain: which keys the
    # spawn overlay replaced/added and what the spawn carried; keys are
    # ABSENT when no --adapter-params was given)
    if spec.adapter_params:
        meta["adapter_params"] = spec.adapter_params
        meta["adapter_overrides"] = list(spec.adapter_overrides)
    # the run's observation catalog snapshot (additive meta key;
    # envs without a catalog -- fakes -- simply omit the key)
    catalog: object = getattr(env, "obs_catalog", None)
    to_meta: object = getattr(catalog, "to_meta", None)
    if isinstance(to_meta, Callable):
        meta["obs_catalog"] = to_meta()
    from unigym.io.writer import ShardWriter

    writer = ShardWriter(
        built["out_dir"],
        shard_steps=_config_int(col_cfg.get("shard_steps", 1000)),
        meta=dict(meta))
    logger = DataCollectLogger(
        cfg, built["out_dir"], num_envs,
        total_steps=total_steps,
        policy_dt=policy_dt,
        log_interval=_config_int(col_cfg.get("log_interval", 500)),
    )
    print(f"[collect] output dir: {built['out_dir']}", flush=True)

    sinks = _build_sinks(
        spec, cfg, mode, env, adapter, writer, logger, filter_chain,
        built["language_gen"], built["lang_interval"], sensing["recorder"],
        num_envs, total_steps, device, policy_dt, built["out_dir"], seed,
        episode_length_s, dur_lo, dur_hi, built["command_dim"], profile,
        bench_cfg, task_meta, sensing["contact_manager"])
    wiring = _build_bridge_and_wiring(
        spec, cfg, backend_label, env, adapter, sensing["camera_manager"],
        built["out_dir"], sinks["bench_sink"], sinks["sink"],
        sinks["timeline"], sinks["director"], sinks["event_log"],
        sinks["watch_specs"], sinks["stop_when"], num_envs)

    return RunInstance(
        spec=spec,
        profile=profile,
        adapter=adapter,
        sampler=sampler,
        term_group=term_group,
        env=env,
        plan=control_plan,
        camera_manager=sensing["camera_manager"],
        prop_manager=prop_manager,
        contact_manager=sensing["contact_manager"],
        recorder=sensing["recorder"],
        director=sinks["director"],
        bridge=wiring["bridge"],
        sink=wiring["sink"],
        writer=writer,
        logger=logger,
        replay=wiring["replay"],
        rewind_providers=wiring["rewind_providers"],
        stop_when=wiring["stop_when"],
        artifacts_dir=built["artifacts_dir"],
        out_dir=built["out_dir"],
        reset_on_termination=reset_on_termination,
    )
