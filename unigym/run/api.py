"""The user-facing ``Run`` facade over spec -> instance -> runner (S1a/S1b).

``Run`` is a lazy builder and session: it owns a :class:`~unigym.run.spec.
RunSpec` from the first call, builds the live :class:`~unigym.run.instance.
RunInstance` on demand (the first live access, or :meth:`Run.build`) and
hands out runners through :func:`unigym.runner.make_runner`.  Static
introspection (``spec`` / ``cfg`` / ``model``) never requires a runner; the
live faces (``step`` / ``steps`` / ``reset``) build one automatically.

Three construction forms share the ONE ``build_spec`` -> ``build_instance``
path (no second config parser):

* ``Run.from_config(path)`` / ``Run(cfg_dict)`` -- an existing run yaml;
* ``Run(world=..., world_base=..., cameras=[...], adapter=..., robot=...,
  ...)`` -- a programmatic assembly whose pieces are builder objects that
  emit the existing config schema (see :mod:`unigym.world` /
  :mod:`unigym.camera`);
* ``Run()`` / ``Run.minimal()`` with neither a cfg nor a world/base raise
  :class:`~unigym.errors.ConfigError`: there is no zero-argument default
  model (v101 A1 removed ``DEFAULT_MJ_MODEL`` and
  ``env.params.mj_model``).

The three introspection levels are deliberately separate surfaces:

* ``run.model`` -- the frozen :class:`~unigym.model.set.ModelSet`: one
  backend-neutral ``ModelIR`` per articulation plus the whole-scene
  ``merged`` IR (``None`` when the backend cannot merge losslessly);
* ``run.controls`` -- the neutral ``{group: ControlRuntime}`` view;
* ``run.articulations`` -- the neutral per-articulation views derived from
  the live runtime (never a duplicated ``ModelIR.meta`` list);
* ``run.cameras`` -- the DECLARED :class:`CameraSet`
  (``run.model.merged.cameras`` is the compiled table);
* ``run.live`` / ``run.live_for(name)`` -- zero-copy dynamic proxies bound
  to the live env (one articulation at a time).
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import os
import tempfile
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from unigym.backends.registry import backend_info
from unigym.run.passive import (
    declare_passive_joints as _declare_passive_joints,
    declare_world_passive_joints as _declare_world_passive_joints,
)
from unigym.camera import CameraSet
from unigym.run.spec import RunSpec, build_spec

if TYPE_CHECKING:
    from unigym.runner.base import Runner

__all__ = ["Run", "load", "make"]

_LOG = logging.getLogger(__name__)

_DEFAULT_STEPS = 10
_DEFAULT_EPISODE_LENGTH_S = 10.0


class BenchmarkSet:
    """The declared benchmark task list with a pre-build ``add``.

    ``add(name)`` accepts a registered task name only.  A callable task
    would need a live task object in the config schema (the benchmark
    section is data), so it is deferred with the exact reason.
    """

    def __init__(self, tasks=(), *, locked: bool = False, on_add=None):
        self._tasks: list = list(tasks or ())
        self._locked = bool(locked)
        self._on_add = on_add

    def add(self, task) -> None:
        """Declare one benchmark task by name (pre-build only)."""
        if self._locked:
            raise RuntimeError(
                "BenchmarkSet.add: the run is already built; benchmarks are "
                "declared pre-build only")
        if callable(task):
            raise NotImplementedError(
                "BenchmarkSet.add(callable) is not representable in the "
                "current schema: benchmark.tasks is data (name / class+"
                "metrics dict) and a live callable has no serialized form; "
                "pass a registered task name, or register the callable as a "
                "benchmark provider and add its name")
        if not isinstance(task, str) or not task:
            raise TypeError(f"BenchmarkSet.add expects a task name string, "
                            f"got {task!r}")
        self._tasks.append(task)
        if self._on_add is not None:
            self._on_add(task)

    def lock(self) -> "BenchmarkSet":
        """Freeze the declared list after build."""
        self._locked = True
        return self

    def names(self) -> list:
        """Declared task names (string entries or dict ``name`` fields)."""
        return [task if isinstance(task, str) else task.get("name")
                for task in self._tasks]

    def to_list(self) -> list:
        """Config-ready copies of the declared task entries."""
        return copy.deepcopy(self._tasks)

    def __iter__(self):
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, item) -> bool:
        return item in self.names()

    def __getitem__(self, index):
        return self.names()[index]

    def __repr__(self) -> str:
        state = "locked" if self._locked else "open"
        return f"BenchmarkSet({self.names()!r}, {state})"


def _runner_kwargs(kind: str, kwargs: dict) -> dict:
    """Validate the runner kind before a build (fail fast, no side effects)."""
    from unigym.runner import _RUNNER_KINDS

    if kind not in _RUNNER_KINDS:
        known = ", ".join(sorted(_RUNNER_KINDS))
        raise ValueError(
            f"unknown runner kind {kind!r} (known: {known})")
    return kwargs


def _adapter_section(adapter) -> dict:
    """Normalize an adapter declaration into the existing ``adapter:`` dict."""
    if adapter is None:
        return {"name": "manual"}
    if isinstance(adapter, str):
        return {"name": adapter}
    if isinstance(adapter, dict):
        return copy.deepcopy(adapter)
    raise NotImplementedError(
        "an adapter INSTANCE is not representable in the current config "
        "schema (adapter is a {name, params} declaration built by "
        "build_adapter). Pass adapter='<registered name>' or None "
        "(ManualAdapter) instead, or register the class under a name")


def _articulation_entry(robot) -> dict:
    """Builder convenience: one ``articulations[]`` entry (v104 shape).

    The public ``Run(robot=...)`` knob is a builder convenience, not a
    config alias: it emits the same authoring shape a yaml file declares
    (``{name, params?, body_parts?}`` -- no articulation-level control).
    """
    if isinstance(robot, str):
        entry = {"name": robot}
    elif isinstance(robot, dict):
        entry = copy.deepcopy(robot)
        entry.pop("control", None)
    else:
        raise NotImplementedError(
            "a RobotRecipe INSTANCE is not representable in the current "
            "config schema (robot is a {name, params, body_parts} "
            "declaration built by build_robot_recipe). Pass robot='g1' "
            "(or another registered name) with params/body_parts in the "
            "config instead")
    return entry


def _controls_entries(robot, adapter) -> list:
    """Builder convenience: the top-level ``controls:`` groups (v105).

    The body group is named after the articulation (the historical action
    face) and selects ``joint_set: body``; recipe-declared extra faces
    (``hand`` / ``right_hand`` / ...) select their same-named joint set.
    Joints are resolved once at build time -- the builder never copies a
    joint list and never infers a set from the group name.
    """
    entry = _articulation_entry(robot)
    name = str(entry["name"])
    adapter_name = str(_adapter_section(adapter).get("name") or "manual")
    body = {
        "name": name,
        "adapter": adapter_name,
        "articulation": name,
        "drive": "position_pd",
        "target": "position",
    }
    groups = [body]
    recipe = None
    try:
        from unigym.robots import build_robot_recipe

        recipe = build_robot_recipe(entry, with_isaaclab=False)
    except (AttributeError, KeyError, NotImplementedError, ValueError):
        recipe = None
    if recipe is not None:
        body["joint_set"] = "body"
        # Only recipe-declared extra faces (``actuate_hands``) are emitted
        # here: an adapter that emits a face the recipe does not declare
        # (e.g. the G1 ``hand`` face) must get it from its config's
        # ``controls[]`` -- the v104 partition is the single authority.
        terms = dict(getattr(recipe, "extra_action_terms", None) or {})
        for face in terms:
            groups.append({
                "name": str(face),
                "adapter": adapter_name,
                "articulation": name,
                "joint_set": str(face),
                "drive": "position_pd",
                "target": "position",
            })
    return groups


def _minimal_cfg(*, run_name: str, seed: int, articulations: list,
                 controls: list, backend: str, num_envs: int, adapter: dict,
                 steps: int, episode_length_s: float) -> dict:
    """The programmatic base config (existing schema only; no YAML).

    The world asset is NOT injected here: callers pass it explicitly via
    ``world_base`` / a :class:`~unigym.world.World` declaration (v101 A1).
    """
    return {
        "run_name": run_name,
        "output_root": "data_out",
        "seed": int(seed),
        "mode": "collect",
        "adapter": copy.deepcopy(adapter),
        "articulations": copy.deepcopy(articulations),
        "controls": copy.deepcopy(controls),
        "env": {
            "backend": backend,
            "num_envs": int(num_envs),
            "episode_length_s": float(episode_length_s),
            "decimation": 4,
            "sim_dt": 0.005,
            "params": {},
        },
        "command": {"name": "fixed", "dim": 0, "values": []},
        "collect": {"steps": int(steps), "shard_steps": 1000},
    }


def _world_base_section(world_base) -> dict:
    """Normalize a ``world_base=`` constructor value into the config shape."""
    if isinstance(world_base, str):
        return {"mjcf": world_base}
    if isinstance(world_base, dict):
        return copy.deepcopy(world_base)
    raise TypeError(
        "world_base= expects a path string or a mapping {mjcf: <path>}, "
        f"got {type(world_base).__name__}")


def _camera_entries(cameras) -> list:
    """Normalize the ``cameras=`` constructor form to a list of entries."""
    if cameras is None:
        return []
    if isinstance(cameras, CameraSet):
        return cameras.to_list()
    if isinstance(cameras, dict):
        return [copy.deepcopy(cameras)]
    if isinstance(cameras, (list, tuple)):
        return [copy.deepcopy(entry) for entry in cameras]
    raise TypeError(
        f"cameras= expects a camera dict / list / CameraSet, got "
        f"{type(cameras).__name__}")


class Run:
    """A lazy run builder + session (v097/v098 S1a/S1b).

    The object is cheap: constructing it only records the config and the
    override knobs.  :meth:`build` (or the first live access) runs the
    existing ``build_spec`` -> ``build_instance`` path exactly once; the
    resulting :class:`~unigym.run.instance.RunInstance` stays owned by this
    ``Run`` and is closed by :meth:`close` (idempotent, also on context
    exit).

    Construction forms:

    * ``Run(cfg_dict)`` / ``Run.from_config(path)`` -- existing configs;
    * ``Run(world=..., world_base=..., cameras=..., adapter=..., robot=...)``
      -- assemble the config from programmatic builders;
    * ``Run()`` / ``Run.minimal()`` with no cfg/world/base raise
      :class:`~unigym.errors.ConfigError` -- the v101 A1 removal of
      ``DEFAULT_MJ_MODEL`` / ``env.params.mj_model`` leaves no implicit
      world.

    Static reads (``spec``, ``cfg``, ``model``) do not need a runner.  Live
    reads (``step``, ``steps``, ``reset``) create the default synchronous
    runner on demand.  ``model`` stays readable after ``close`` because the
    IR is pure data; ``live`` refuses once the run is closed.

    Runner ownership: :meth:`runner` may be called before the first step;
    the chosen ``sync``/``external`` runner becomes the active one and
    :meth:`step` delegates to it.  Changing the runner **kind** after the
    first step raises (one time owner per run).

    Args:
        cfg: An existing run config mapping (``None`` selects the
            programmatic form; a world/base is then required).
        world: A :class:`unigym.world.World` declaration.
        world_base: An explicit world asset (``{mjcf: <path>}`` or a path
            string) merged into the run config top level.
        cameras: Camera-spec dicts (``unigym.Camera.*``) or a ``CameraSet``.
        adapter: ``None`` (ManualAdapter) / a registered adapter name.
        robot: A registered robot name (default ``"g1"``).
        backend: Environment backend (default ``"mujoco"``).
        backend_options: Backend-private options mapping
            (``env.backend_options``; v105.3), e.g.
            ``{"mujoco": {"compose": {"auto_prefix": True}}}``.
        num_envs: Environment count (default 1).
        seed: Run seed (default 0).
        inspect: Build quietly (stdout captured into :attr:`logs`, no
            artifacts, temp output dir).
        **overrides: Extra top-level config keys (existing schema only).
    """

    def __init__(self, cfg: dict | None = None, *, world=None, world_base=None,
                 cameras=None, adapter=None, robot="g1", backend=None,
                 backend_options=None, num_envs=None, seed=None, device=None,
                 steps=None,
                 episode_length_s=None, output_dir=None, label=None, set=(),
                 adapter_params=None, no_artifacts=False, inspect=False,
                 _config_path: str | None = None, **overrides):
        if cfg is not None and not isinstance(cfg, dict):
            raise TypeError(
                f"Run() config must be a mapping, got {type(cfg).__name__}")
        if cfg is None and world is None and world_base is None:
            from unigym.errors import ConfigError

            raise ConfigError(
                "Run() needs an explicit construction source: pass a config "
                "(Run(cfg) / Run.from_config(path)), a World declaration "
                "(Run(world=World.base(mjcf=...))), or world_base=...; "
                "v101 A1 removed the zero-argument default and "
                "env.params.mj_model")
        self._world = world
        self._backend_options = (copy.deepcopy(backend_options)
                                 if backend_options is not None else None)
        self._inspect = bool(inspect)
        self._logs = ""
        if cfg is None:
            raw = self._assemble_minimal(
                world=world, cameras=cameras, adapter=adapter, robot=robot,
                backend=backend, num_envs=num_envs, seed=seed, steps=steps,
                episode_length_s=episode_length_s)
            if world_base is not None:
                raw["world_base"] = _world_base_section(world_base)
            if self._backend_options is not None:
                raw.setdefault("env", {})["backend_options"] = \
                    copy.deepcopy(self._backend_options)
        else:
            raw = copy.deepcopy(cfg)
            if world is not None:
                world.apply(raw)
                _declare_world_passive_joints(raw)
            if world_base is not None:
                raw["world_base"] = _world_base_section(world_base)
            if self._backend_options is not None:
                raw.setdefault("env", {})["backend_options"] = \
                    copy.deepcopy(self._backend_options)
            if cameras is not None:
                raw["cameras"] = _camera_entries(cameras)
            if adapter is not None:
                raw["adapter"] = _adapter_section(adapter)
            if robot != "g1":
                entry = _articulation_entry(robot)
                cfg_controls = _controls_entries(
                    robot, adapter if adapter is not None
                    else raw.get("adapter"))
                _declare_passive_joints(entry, cfg_controls)
                arts = raw.setdefault("articulations", [])
                if arts:
                    arts[0] = entry
                else:
                    arts.append(entry)
                raw["controls"] = cfg_controls
            elif "articulations" not in raw:
                entry = {"name": "g1"}
                cfg_controls = _controls_entries(
                    "g1", adapter if adapter is not None
                    else raw.get("adapter"))
                _declare_passive_joints(entry, cfg_controls)
                raw["articulations"] = [entry]
                raw["controls"] = cfg_controls
        for key, value in overrides.items():
            raw[key] = copy.deepcopy(value)
        self._raw_cfg = raw
        self._config_path = _config_path or "<memory>"
        self._num_envs = num_envs
        self._seed = seed
        self._backend = backend
        self._device = device
        self._steps = steps
        self._episode_length_s = episode_length_s
        self._output_dir = output_dir
        self._label = label
        self._set = tuple(set or ())
        self._adapter_params = adapter_params
        self._no_artifacts = bool(no_artifacts)
        self._camera_set = CameraSet(raw.get("cameras") or [])
        self._benchmark_set = BenchmarkSet(
            self._benchmark_tasks(raw),
            on_add=self._on_benchmark_add)
        self._spec: RunSpec | None = None
        self._instance = None
        self._runner: "Runner | None" = None
        self._runner_kind: str | None = None
        self._stepped = False
        self._closed = False

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _assemble_minimal(self, *, world, cameras, adapter, robot, backend,
                          num_envs, seed, steps, episode_length_s) -> dict:
        """The programmatic base config (existing schema; no world asset)."""
        articulation_entry = _articulation_entry(robot)
        adapter_section = _adapter_section(adapter)
        backend_label = backend or "mujoco"
        minimal_controls = _controls_entries(robot, adapter_section)
        _declare_passive_joints(articulation_entry, minimal_controls)
        raw = _minimal_cfg(
            run_name="minimal_run",
            seed=0 if seed is None else int(seed),
            articulations=[articulation_entry],
            controls=minimal_controls,
            backend=backend_label,
            num_envs=1 if num_envs is None else int(num_envs),
            adapter=adapter_section,
            steps=_DEFAULT_STEPS if steps is None else int(steps),
            episode_length_s=(_DEFAULT_EPISODE_LENGTH_S
                              if episode_length_s is None
                              else float(episode_length_s)),
        )
        if world is not None:
            world.apply(raw)
            _declare_world_passive_joints(raw)
        if cameras is not None:
            raw["cameras"] = _camera_entries(cameras)
        return raw

    @classmethod
    def minimal(cls, **kwargs) -> "Run":
        """Same as ``Run(**kwargs)``; a world/base is required (no default)."""
        return cls(None, **kwargs)

    @classmethod
    def from_config(cls, path, *, num_envs=None, seed=None, backend=None,
                    device=None, steps=None, episode_length_s=None,
                    output_dir=None, label=None, set=(),
                    adapter_params=None, no_artifacts=False,
                    inspect=False) -> "Run":
        """Load a run yaml and return a lazy :class:`Run` for it.

        Args:
            path: Run config path (``.yaml``).
            num_envs: Override ``env.num_envs``.
            seed: Override the run seed.
            backend: Override ``env.backend``.
            device: Torch device label (default ``cpu``).
            steps: Override ``collect.steps``.
            episode_length_s: Override ``env.episode_length_s``.
            output_dir: Override the run output directory.
            label: Display-only run label.
            set: ``key=value`` config overrides (collector ``-s``).
            adapter_params: Adapter-section overlay mapping.
            no_artifacts: Disable per-run scene artifacts.
            inspect: Build quietly with a temp output dir.

        Returns:
            A lazy ``Run`` whose spec is built on first access.
        """
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            raise ValueError(
                f"run config {path!r} must be a yaml mapping, got "
                f"{type(cfg).__name__}")
        return cls(
            cfg, num_envs=num_envs, seed=seed, backend=backend,
            device=device, steps=steps, episode_length_s=episode_length_s,
            output_dir=output_dir, label=label, set=set,
            adapter_params=adapter_params, no_artifacts=no_artifacts,
            inspect=inspect,
            _config_path=os.path.abspath(os.fspath(path)))

    def _args(self) -> Namespace:
        """Build the minimal collector-CLI namespace ``build_spec`` reads."""
        adapter_params = self._adapter_params
        if isinstance(adapter_params, dict):
            adapter_params = json.dumps(adapter_params)
        return Namespace(
            config=self._config_path,
            num_envs=self._num_envs,
            steps=self._steps,
            episode_length_s=self._episode_length_s,
            output_dir=self._output_dir,
            no_artifacts=self._no_artifacts,
            ckpt_dir=None,
            instruction=None,
            seed=self._seed,
            set=list(self._set) or None,
            adapter_params=adapter_params,
            label=self._label,
            serve_camera=None,
            serve_frame_mb=4.0,
            serve_state_dir=None,
            run_log=None,
            serve_pause=False,
            replay=None,
            replay_mode="action",
            replay_key=None,
            replay_strict=False,
            replay_freeze_rng=False,
            direct=False,
            watch=None,
            enable_cameras=False,
            device=self._device or "cpu",
        )

    def _build_spec(self) -> RunSpec:
        """Run the one existing normalization path (no second parser)."""
        cfg = self._raw_cfg
        # Pre-build CameraSet/BenchmarkSet edits are declarations too:
        # fold them into the config at the single normalization point so an
        # add() made after a spec read still reaches build_spec.
        if len(self._camera_set):
            cfg = copy.deepcopy(cfg)
            cfg["cameras"] = self._camera_set.to_list()
        if len(self._benchmark_set):
            cfg = copy.deepcopy(cfg)
            cfg.setdefault("benchmark", {})["tasks"] = \
                self._benchmark_set.to_list()
        if self._backend is not None:
            cfg = copy.deepcopy(cfg)
            cfg.setdefault("env", {})["backend"] = self._backend
        backend = str((cfg.get("env") or {}).get("backend", "isaaclab"))
        static_shapes: tuple = ()
        if self._world is not None \
                and not backend_info(backend).inline_static_scene:
            if not self._world.unmapped_static_scene:
                static_shapes = tuple(self._world.static_declarations())
                if static_shapes:
                    # The inline world scene is the mujoco-only fill of these
                    # SAME primitives.  The isaaclab run receives them as
                    # native static colliders instead (no silent drop).
                    cfg = copy.deepcopy(cfg)
                    params = cfg.setdefault("env", {}).setdefault("params", {})
                    params.pop("mj_scene", None)
                    params.pop("mj_scene_params", None)
        spec = build_spec(cfg, self._args())
        if self._world is not None:
            terrain = self._world.terrain_value()
            if terrain is not None:
                spec = replace(spec, terrain=terrain)
            if static_shapes:
                spec = replace(spec, static_shapes=static_shapes)
        return spec

    def _prepare_inspect(self) -> None:
        """Force the quiet build knobs (temp output, no artifacts)."""
        self._no_artifacts = True
        self._output_dir = tempfile.mkdtemp(prefix="unigym_inspect_")
        self._spec = None

    def build(self, inspect: bool = False):
        """Build (once) and return the live :class:`RunInstance`.

        Args:
            inspect: Build quietly for inspection: ``[collect]`` /
                ``[mujoco_env]`` / ``[bridge]`` stdout is captured into
                :attr:`logs`, artifacts are disabled and the output dir is a
                temp directory.  The model still compiles (``run.model``).
        """
        if self._instance is not None:
            return self._instance
        if self._closed:
            raise RuntimeError(
                "Run is closed; create a new Run to build again")
        quiet = bool(inspect or self._inspect)
        if quiet:
            self._prepare_inspect()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self._instance = self._build_now()
            self._logs += buffer.getvalue()
        else:
            self._instance = self._build_now()
        self._camera_set.lock()
        self._benchmark_set.lock()
        if self._world is not None:
            self._world._mark_built()
        return self._instance

    def _build_now(self):
        """Run the one existing construction path (``build_instance``)."""
        _LOG.debug("Run(%s): building instance lazily", self._display_name())
        from unigym.run.build import build_instance

        return build_instance(self.spec)

    def close(self) -> None:
        """Close the runner/instance once; repeated calls are no-ops."""
        if self._closed:
            return
        self._closed = True
        if self._runner is not None:
            try:
                self._runner.close()
            finally:
                self._runner = None
        if self._instance is not None:
            self._instance.close()

    def __enter__(self) -> "Run":
        """Enter the session (the run builds on first live access)."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the session, closing the owned instance."""
        self.close()

    # ------------------------------------------------------------------ #
    # representation
    # ------------------------------------------------------------------ #
    def _display_name(self) -> str:
        """The declared run name without forcing a spec build."""
        return str(self._raw_cfg.get("run_name") or "<unnamed>")

    def __repr__(self) -> str:
        """``<Run name=... backend=... num_envs=... built=...>``."""
        env = self._raw_cfg.get("env") or {}
        backend = self._backend or env.get("backend") or "isaaclab"
        num_envs = (self._num_envs if self._num_envs is not None
                    else env.get("num_envs", "?"))
        return (f"<Run name={self._display_name()!r} backend={backend!r} "
                f"num_envs={num_envs} built={self.built}>")

    def _repr_html_(self) -> str:
        """A small HTML table for notebooks (same facts as ``__repr__``)."""
        env = self._raw_cfg.get("env") or {}
        rows = {
            "name": self._display_name(),
            "backend": self._backend or env.get("backend", "isaaclab"),
            "num_envs": (self._num_envs if self._num_envs is not None
                         else env.get("num_envs", "?")),
            "built": self.built,
        }
        body = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>"
                       for key, value in rows.items())
        return f"<table><tbody>{body}</tbody></table>"

    @property
    def logs(self) -> str:
        """Captured stdout of the last quiet (inspect) build."""
        return self._logs

    @property
    def built(self) -> bool:
        """True once the live instance has been constructed."""
        return self._instance is not None

    # ------------------------------------------------------------------ #
    # introspection (static)
    # ------------------------------------------------------------------ #
    @property
    def spec(self) -> RunSpec:
        """The frozen :class:`RunSpec` (built lazily, no engine objects)."""
        if self._spec is None:
            self._spec = self._build_spec()
        return self._spec

    @property
    def controls(self) -> dict:
        """The neutral control runtime views (``{group: ControlRuntime}``)."""
        return dict(self.build().env.controls)

    @property
    def plan(self):
        """The frozen :class:`ResolvedControlPlan` of this run (read-only).

        Built once with the run instance from the resolved controls; the
        joints/drive/target/q0/provenance are never re-inferred here.
        """
        return self.build().plan

    @property
    def resolved_controls(self) -> tuple:
        """The plan's :class:`ResolvedControl` rows (read-only tuple)."""
        plan = self.plan
        return () if plan is None else tuple(plan.controls)

    @property
    def cfg(self) -> dict:
        """The normalized effective config mapping."""
        return self.spec.cfg

    @property
    def model(self):
        """The env's per-articulation :class:`~unigym.model.set.ModelSet`.

        ``run.model.articulations[name]`` is that articulation's own IR;
        ``run.model.merged`` is the whole-scene IR (``None`` when the
        backend cannot merge losslessly, e.g. isaaclab2 multi).  This reads
        the set the env already assembled, so it stays valid after
        :meth:`close` (pure data, no native state).
        """
        if self._instance is not None:
            return self._instance.model
        return self.build().model

    def live_for(self, name=None):
        """Zero-copy live proxies for one articulation's IR (v103.4).

        Args:
            name: An articulation name; ``None`` binds the single
                articulation of a one-articulation run.

        Raises:
            RuntimeError: The run was closed, or *name* was omitted for a
                multi-articulation run (there is no first-binding fallback).
        """
        if self._closed:
            raise RuntimeError(
                "Run is closed: live proxies are unavailable; static .model "
                "reads remain valid")
        return self.build().live_for(name)

    @property
    def live(self):
        """Zero-copy live proxies over the single articulation (built on demand).

        A one-articulation run binds its only IR; a multi-articulation run
        requires :meth:`live_for` with an explicit name (no first-binding
        fallback).

        Raises:
            RuntimeError: The run was closed or has several articulations.
        """
        return self.live_for()

    @property
    def articulations(self) -> list:
        """The neutral per-articulation views derived from the live runtime.

        One entry per ``env.articulations`` runtime: name/model/root/joints/
        q0/feet/body_parts/spawn plus the owning body control's action face.
        This is NOT a copy of a ``ModelIR.meta`` annotation.
        """
        env = self.build().env
        controls = getattr(env, "controls", {}) or {}
        views = []
        for name, art in (getattr(env, "articulations", {}) or {}).items():
            body = next(
                (c for c in controls.values()
                 if c.articulation == name and c.action_face == name),
                None)
            model = getattr(art, "model", None)
            model_to_dict = getattr(model, "to_dict", None)
            views.append({
                "name": name,
                "role": "articulation",
                "model": (model_to_dict()
                          if callable(model_to_dict) else None),
                "root": getattr(art, "root", None),
                "base_body": getattr(art, "base_body", None),
                "joints": list(getattr(art, "joint_names", ()) or ()),
                "feet": list(getattr(art, "feet", ()) or ()),
                "body_parts": dict(getattr(art, "body_parts", {}) or {}),
                "spawn": dict(getattr(art, "spawn", {}) or {}),
                "action_face": (body.action_face if body is not None else None),
            })
        return views

    @property
    def cameras(self) -> CameraSet:
        """The DECLARED camera set (``run.model.cameras`` = compiled table)."""
        return self._camera_set

    @staticmethod
    def _benchmark_tasks(cfg: dict) -> list:
        """The declared benchmark task list from a config mapping."""
        return list((cfg.get("benchmark") or {}).get("tasks") or [])

    def _on_benchmark_add(self, task: str) -> None:
        """Mirror a pre-build ``benchmarks.add`` into the raw config."""
        bench = self._raw_cfg.setdefault("benchmark", {})
        tasks = bench.setdefault("tasks", [])
        tasks.append(task)

    @property
    def benchmarks(self) -> BenchmarkSet:
        """The declared benchmark set (``.add(name)`` is pre-build only)."""
        return self._benchmark_set

    @property
    def adapter(self):
        """The live adapter of the built instance (builds on first access)."""
        return self.build().adapter

    def to_config(self, path) -> str:
        """Write the normalized effective config as yaml; return its path.

        The world asset serializes as the run-level ``world_base: {mjcf:
        <path>}`` declaration (v101 A1); it is never an ``env.params`` key.

        Args:
            path: Destination file (parent directories are created).
        """
        out = Path(path)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        cfg = self.spec.cfg
        if "props" in cfg:
            # Serialize the content block on its current surface spelling
            # (``objects:``); the long-lived alias reloads it identically.
            cfg = copy.deepcopy(cfg)
            cfg["objects"] = cfg.pop("props")
        out.write_text(
            yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8")
        return str(out)

    # ------------------------------------------------------------------ #
    # live faces
    # ------------------------------------------------------------------ #
    def runner(self, kind: str | None = None, **kwargs) -> "Runner":
        """Build or return this run's active :class:`Runner`.

        Args:
            kind: Runner kind (``sync`` / ``external`` / ``gym`` /
                ``gym_vector``).  ``None`` returns the active runner
                (default ``sync`` when none exists yet).
            **kwargs: Kind-specific options forwarded to ``make_runner``.

        Returns:
            The runner.  The active default used by ``step`` is the most
            recent ``sync``/``external``/``gym`` runner.

        Raises:
            RuntimeError: A DIFFERENT kind was requested after the first
                step (one time owner per run).
        """
        if kind is None:
            if self._runner is not None:
                return self._runner
            kind = "sync"
        _runner_kwargs(kind, kwargs)
        if self._runner is not None:
            if self._stepped and kind != self._runner_kind:
                raise RuntimeError(
                    f"cannot change runner kind after stepping: active "
                    f"{self._runner_kind!r}, requested {kind!r} | hint: one "
                    f"runner owns the tick clock per run; create a new Run "
                    f"to use a different kind")
            if kind == self._runner_kind and not kwargs:
                return self._runner
        from unigym.runner import make_runner

        runner = make_runner(self.build(), kind=kind, **kwargs)
        self._runner = runner
        self._runner_kind = kind
        return runner

    def gym(self, **kwargs):
        """Convenience alias for ``runner("gym", **kwargs)``."""
        return self.runner("gym", **kwargs)

    def external(self, **kwargs):
        """Convenience alias for ``runner("external", **kwargs)``."""
        return self.runner("external", **kwargs)

    def _default_runner(self) -> "Runner":
        """Return the cached active runner, creating sync on first use."""
        runner = self._runner
        if runner is None:
            runner = self.runner("sync")
        return runner

    def reset(self, env_ids=None):
        """Reset the named envs (all when *None*) via the active runner.

        Returns:
            The runner's :class:`~unigym.runner.base.StepResult`.
        """
        return self._default_runner().reset(env_ids)

    def step(self, actions=None):
        """Advance one policy tick and return its ``StepResult``.

        Args:
            actions: Optional ``dict[str, NamedAction]``.  When supplied the
                adapter forward/postprocess is skipped (external-driver
                path); ``None`` reproduces the active runner's normal tick.
        """
        runner = self._default_runner()
        self._stepped = True
        if actions is None:
            return runner.tick()
        return runner.tick(actions_override=actions)

    def steps(self, n=None):
        """Yield ``StepResult`` objects for *n* ticks (remaining by default).

        Args:
            n: Number of ticks.  ``None`` yields until the spec's
                ``total_steps`` budget is exhausted.
        """
        runner = self._default_runner()
        if n is None:
            n = max(int(self.spec.total_steps) - int(runner.ticks_done), 0)
        for _ in range(int(n)):
            yield self.step()


def load(path, **kwargs) -> Run:
    """Load a run config path into a lazy :class:`Run`.

    Args:
        path: Run config path.
        **kwargs: Same override knobs as :meth:`Run.from_config`.

    Returns:
        A lazily built ``Run``.
    """
    return Run.from_config(path, **kwargs)


def make(target, kind: str = "gym", **kwargs):
    """Build a runner facade for a ``Run`` (or a config path).

    Args:
        target: A :class:`Run` or a run config path.
        kind: Runner kind (default ``gym``: the single-env ``gym.Env``
            facade; use ``gym_vector`` for batched runs).
        **kwargs: Kind-specific options forwarded to ``make_runner``.

    Returns:
        The runner produced by :func:`unigym.runner.make_runner`.
    """
    if isinstance(target, Run):
        instance = target.build()
    elif isinstance(target, (str, os.PathLike)):
        instance = load(target).build()
    else:
        raise TypeError(
            "make() expects a Run or a config path, got "
            f"{type(target).__name__}")
    _runner_kwargs(kind, kwargs)
    from unigym.runner import make_runner

    return make_runner(instance, kind=kind, **kwargs)
