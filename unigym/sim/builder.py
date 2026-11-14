"""Declaration-only :class:`SimulationBuilder` for the experimental facade.

The builder owns no engine object, imports no backend module at construction
time and never starts a process.  It validates declarations against one
explicit :class:`~unigym.packs.packset.PackSet` and mints identity-scoped
handles.  ``build()`` is the single transition to live state and belongs to the
next implementation slice; this slice raises :class:`NotImplementedError`
instead of fabricating a partially live simulation.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

from unigym._typing import JsonValue
from unigym.run.passive import declare_passive_joints as _declare_passive_joints

from .control import ControlSpec
from .errors import (
    BuilderSealedError,
    ConfigModeError,
    DeferredFeatureError,
    NoControlError,
    NoWorldError,
    SimulationConfigError,
)
from .handles import (
    ArticulationHandle,
    BodyHandle,
    CameraHandle,
    _check_owner,
)
from .profile import load_profile_for_ref
from .simulation import Simulation

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend
    from unigym.packs.packset import PackRuntime, PackSet, RobotRef
    from unigym.robots.base import RobotProfile
    from unigym.run.api import Run
    from unigym.run.spec import RunSpec
    from unigym.world import World, _TerrainBuilder

__all__ = ["SimulationBuilder"]


def _positive_float(value: object, what: str) -> float:
    """Validate one finite scalar strictly greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationConfigError(f"{what} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise SimulationConfigError(f"{what} must be finite and > 0, got {value!r}")
    return number


def _positive_int(value: object, what: str) -> int:
    """Validate one integer strictly greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SimulationConfigError(f"{what} must be an int >= 1, got {value!r}")
    return value


def _json_limit(value: float | Mapping[str, float]) -> float | dict[str, JsonValue]:
    """A validated gain/limit as the JSON shape the run config accepts."""
    if isinstance(value, Mapping):
        return {str(name): float(amount) for name, amount in value.items()}
    return float(value)


def _json_names(names: Iterable[str]) -> list[JsonValue]:
    """A joint-name sequence as a JSON list."""
    return [name for name in names]


def _json_float_map(values: Mapping[str, float]) -> dict[str, JsonValue]:
    """A finite float mapping as a JSON mapping."""
    return {str(name): float(amount) for name, amount in values.items()}


def _json_objects(entries: list[dict[str, JsonValue]]) -> list[JsonValue]:
    """A list of JSON objects as a JSON list."""
    return [entry for entry in entries]


class SimulationBuilder:
    """Validate facade declarations and mint owned handles.

    Args:
        backend: a registered backend name (a registry lookup, never a build).
        dt: the policy/control period in seconds.
        physics_dt: the engine step; exactly one of this and ``substeps``.
        substeps: the integer decimation; exactly one of this and
            ``physics_dt``.
        seed: the deterministic seed (>= 0).
        num_envs: the environment count (>= 1; the spike accepts ``N == 1``).
        pack_set: the explicit pack scope every robot ref is validated against.
        name: the run label.
        steps: the collect budget (>= 1).
        episode_length_s: the episode duration in seconds (> 0).
        artifacts: whether to write run artifacts.
        device: the backend device string.
    """

    def __init__(
        self,
        *,
        backend: str,
        dt: float,
        physics_dt: float | None = None,
        substeps: int | None = None,
        seed: int = 0,
        num_envs: int = 1,
        pack_set: PackSet,
        name: str = "simulation",
        steps: int = 1000,
        episode_length_s: float = 20.0,
        artifacts: bool = False,
        device: str = "cpu",
    ) -> None:
        if not isinstance(backend, str) or not backend.strip():
            raise SimulationConfigError(
                f"SimulationBuilder backend must be a non-empty string, got {backend!r}"
            )
        from unigym.backends.registry import BackendSelectionError, backend_info

        try:
            info = backend_info(backend)
        except (BackendSelectionError, ValueError) as err:
            raise SimulationConfigError(
                f"SimulationBuilder backend {backend!r} is not registered: {err}"
            ) from err

        policy_dt = _positive_float(dt, "SimulationBuilder dt")
        if (physics_dt is None) == (substeps is None):
            raise SimulationConfigError(
                "SimulationBuilder requires exactly one of physics_dt or substeps "
                f"(got physics_dt={physics_dt!r}, substeps={substeps!r})"
            )
        if physics_dt is not None:
            engine_dt = _positive_float(physics_dt, "SimulationBuilder physics_dt")
            ratio = policy_dt / engine_dt
            decimation = round(ratio)
            if decimation < 1 or not math.isclose(
                policy_dt, engine_dt * decimation, rel_tol=0.0, abs_tol=1e-12
            ):
                raise SimulationConfigError(
                    "SimulationBuilder dt must equal physics_dt * substeps exactly "
                    f"(dt={policy_dt!r}, physics_dt={engine_dt!r}, "
                    f"implied substeps={ratio!r})"
                )
        else:
            decimation = _positive_int(substeps, "SimulationBuilder substeps")
            engine_dt = policy_dt / decimation
            if not math.isclose(policy_dt, engine_dt * decimation, rel_tol=0.0, abs_tol=1e-12):
                raise SimulationConfigError(
                    "SimulationBuilder dt must equal physics_dt * substeps exactly "
                    f"(dt={policy_dt!r}, substeps={decimation}, "
                    f"implied physics_dt={engine_dt!r})"
                )

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SimulationConfigError(f"SimulationBuilder seed must be an int >= 0, got {seed!r}")
        if not isinstance(name, str) or not name.strip():
            raise SimulationConfigError(
                f"SimulationBuilder name must be a non-empty string, got {name!r}"
            )
        if not isinstance(device, str) or not device.strip():
            raise SimulationConfigError(
                f"SimulationBuilder device must be a non-empty string, got {device!r}"
            )
        if not isinstance(artifacts, bool):
            raise SimulationConfigError(
                f"SimulationBuilder artifacts must be a bool, got {artifacts!r}"
            )

        self._backend = info.name
        self._dt = policy_dt
        self._physics_dt = engine_dt
        self._substeps = decimation
        self._seed = seed
        self._num_envs = _positive_int(num_envs, "SimulationBuilder num_envs")
        self._pack_set: PackSet | None = pack_set
        self._name = name
        self._steps = _positive_int(steps, "SimulationBuilder steps")
        self._episode_length_s = _positive_float(
            episode_length_s, "SimulationBuilder episode_length_s"
        )
        self._artifacts = artifacts
        self._device = device
        self._world: World | None = None
        self._articulations: list[ArticulationHandle] = []
        self._cameras: list[CameraHandle] = []
        self._owner = object()
        self._sealed = False
        self._sim: Simulation | None = None
        self._pack_runtime: PackRuntime | None = pack_set.to_runtime()
        self._config_mode = False
        self._run: Run | None = None
        self._spec: RunSpec | None = None

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str],
        *,
        pack_set: PackSet | None = None,
        backend: str | None = None,
        device: str | None = None,
    ) -> SimulationBuilder:
        """Load a run config and return a config-mode builder.

        The document is parsed and normalized by the one existing
        ``Run.from_config`` path.  A config-mode builder declares nothing:
        its handles are re-adopted from the live instance after ``build()``.
        """
        from unigym.run.api import Run

        runtime = None if pack_set is None else pack_set.to_runtime()
        run = Run.from_config(
            path,
            pack_runtime=runtime,
            discover_runtime=False,
            backend=backend,
            device=device,
        )
        spec = run.spec
        manual_controls = [control for control in spec.controls if control.adapter == "manual"]
        if not manual_controls:
            control_adapters = sorted({control.adapter for control in spec.controls})
            adapter_label = spec.adapter_name or (control_adapters[0] if control_adapters else None)
            raise SimulationConfigError(
                "SimulationBuilder.from_config needs at least one manual control face "
                f"to expose an external action table; this config declares adapter "
                f"{adapter_label!r} with control adapter(s) "
                f"{control_adapters or ['<none>']}. Declare a controls[] group with "
                "adapter: manual -- the facade is not a policy-adapter driver."
            )
        env_cfg = spec.cfg.get("env")
        if not isinstance(env_cfg, Mapping):
            raise SimulationConfigError(
                "a config-mode builder needs an 'env' mapping in the run config, "
                f"got {type(env_cfg).__name__}"
            )
        physics_dt = _positive_float(env_cfg.get("sim_dt"), "run config env.sim_dt")
        decimation = _positive_int(env_cfg.get("decimation"), "run config env.decimation")
        dt = _positive_float(physics_dt * decimation, "run config env.sim_dt * env.decimation")
        episode_length_s = _positive_float(
            env_cfg.get("episode_length_s"), "run config env.episode_length_s"
        )
        builder = cls.__new__(cls)
        builder._backend = spec.backend
        builder._dt = dt
        builder._physics_dt = physics_dt
        builder._substeps = decimation
        builder._seed = spec.seed
        builder._num_envs = spec.num_envs
        builder._pack_set = pack_set
        builder._pack_runtime = runtime
        builder._name = spec.run_name
        builder._steps = spec.total_steps
        builder._episode_length_s = episode_length_s
        builder._artifacts = spec.output.artifacts
        builder._device = spec.device
        builder._world = None
        builder._articulations = []
        builder._cameras = []
        builder._owner = object()
        builder._sealed = False
        builder._sim = None
        builder._config_mode = True
        builder._run = run
        builder._spec = spec
        return builder

    @property
    def world(self) -> World | None:
        """The one world declaration owned by this builder."""
        return self._world

    @world.setter
    def world(self, world: World) -> None:
        self._check_declarable("world")
        self._world = world

    @property
    def terrain(self) -> _TerrainBuilder:
        """Sugar for ``builder.world.terrain``; the world owns the terrain."""
        if self._world is None:
            raise NoWorldError("builder.terrain needs builder.world; set builder.world first")
        return self._world.terrain

    @property
    def pack_runtime(self) -> PackRuntime:
        """The activation-capable runtime over this builder's explicit pack set."""
        runtime = self._pack_runtime
        if runtime is None:
            raise SimulationConfigError(
                "this config-mode builder has no pack runtime; pass pack_set= to "
                "SimulationBuilder.from_config for a config that references a pack"
            )
        return runtime

    @property
    def articulations(self) -> tuple[ArticulationHandle, ...]:
        """Declared articulations; config-mode handles arrive after ``build()``."""
        return tuple(self._articulations)

    @property
    def cameras(self) -> tuple[CameraHandle, ...]:
        """Declared cameras; config-mode handles arrive after ``build()``."""
        return tuple(self._cameras)

    def add_articulation(
        self,
        ref: RobotRef,
        *,
        name: str | None = None,
        control: ControlSpec | None = None,
    ) -> ArticulationHandle:
        """Declare one pack robot and resolve every control face against its profile."""
        self._check_declarable("add_articulation")
        if control is None:
            raise NoControlError(
                f"add_articulation(ref={getattr(ref, 'id', ref)!r}) needs control=; "
                f"there is no implicit default control"
            )
        label = ref.id if name is None else name
        if not isinstance(label, str) or not label.strip():
            raise SimulationConfigError(
                f"articulation label must be a non-empty string, got {label!r}"
            )
        if any(handle.name == label for handle in self._articulations):
            raise SimulationConfigError(
                f"articulation label {label!r} is already declared; labels must be unique"
            )
        pack_set = self._pack_set
        if pack_set is None:
            raise SimulationConfigError(
                "a declaration-mode builder needs an explicit pack_set; "
                "there is no ambient pack discovery"
            )
        profile = load_profile_for_ref(pack_set, ref)
        faces = self._resolve_faces(label, profile, control)
        self._check_face_names_available(label, control)
        handle = ArticulationHandle(
            name=label,
            ref=ref,
            profile=profile,
            control=control,
            faces=faces,
            _owner=self._owner,
        )
        self._articulations.append(handle)
        return handle

    def add_camera(
        self,
        entry: Mapping[str, JsonValue],
        *,
        name: str | None = None,
        attach: BodyHandle | None = None,
    ) -> CameraHandle:
        """Declare one camera entry from the existing ``unigym.Camera`` schema."""
        self._check_declarable("add_camera")
        if attach is not None:
            if not isinstance(attach, BodyHandle):
                raise SimulationConfigError(
                    f"add_camera attach must be a BodyHandle, got {type(attach).__name__}"
                )
            _check_owner(attach, self._owner, "camera attach body")
            raise DeferredFeatureError(
                "add_camera(attach=BodyHandle) is deferred; use "
                "Camera.attached(name, mount=<native link name>) for a body camera"
            )
        if not isinstance(entry, Mapping):
            raise SimulationConfigError(
                f"add_camera entry must be a cameras[] mapping, got {type(entry).__name__}"
            )
        value: dict[str, JsonValue] = copy.deepcopy(dict(entry))
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise SimulationConfigError(
                    f"camera name override must be a non-empty string, got {name!r}"
                )
            value["name"] = name
        camera_name = value.get("name")
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise SimulationConfigError(
                f"camera entry needs a non-empty 'name', got {camera_name!r}"
            )
        if any(handle.name == camera_name for handle in self._cameras):
            raise SimulationConfigError(
                f"camera name {camera_name!r} is already declared; names must be unique"
            )
        handle = CameraHandle(name=camera_name, entry=value, _owner=self._owner)
        self._cameras.append(handle)
        return handle

    def _to_cfg(self) -> dict[str, JsonValue]:
        """Assemble the declaration tables of the existing run-config document.

        The builder is the single owner of ``articulations[]`` and
        ``controls[]``; the world contributes only its own fragment, which
        ``Run(cfg, world=world)`` merges exactly once.  The emitted
        ``dt``/``physics_dt`` arithmetic is re-checked here because this
        document is the contract handed to ``Run``.
        """
        world = self._world
        if world is None:
            raise NoWorldError("build() needs builder.world; set builder.world first")
        fragment = world.fragment()
        if not fragment:
            raise NoWorldError(
                "builder.world declares no world asset; set a scene_preset or world_base"
            )
        for key in ("articulations", "controls"):
            if key in fragment:
                raise SimulationConfigError(
                    f"the world fragment must not declare {key!r}; "
                    f"use builder.add_articulation for every articulation and control"
                )
        if not self._articulations:
            raise SimulationConfigError(
                "SimulationBuilder needs at least one articulation; call add_articulation"
            )
        sim_dt = self._physics_dt
        decimation = self._substeps
        if not math.isclose(self._dt, sim_dt * decimation, rel_tol=0.0, abs_tol=1e-12):
            raise SimulationConfigError(
                "SimulationBuilder dt must equal physics_dt * substeps exactly "
                f"(dt={self._dt!r}, physics_dt={sim_dt!r}, substeps={decimation})"
            )

        cfg: dict[str, JsonValue] = {
            "run_name": self._name,
            "output_root": "data_out",
            "seed": self._seed,
            "mode": "collect",
            "adapter": {"name": "manual"},
            "env": {
                "backend": self._backend,
                "num_envs": self._num_envs,
                "episode_length_s": self._episode_length_s,
                "decimation": decimation,
                "sim_dt": sim_dt,
                "params": {},
            },
            "command": {"name": "fixed", "dim": 0, "values": []},
            "collect": {"steps": self._steps, "shard_steps": 1000},
        }
        cfg["articulations"] = _json_objects(self._articulation_entries())
        cfg["controls"] = _json_objects(self._control_entries())
        cameras = self._camera_entries()
        if cameras:
            cfg["cameras"] = _json_objects(cameras)
        return cfg

    def _articulation_entries(self) -> list[dict[str, JsonValue]]:
        """The ``articulations[]`` table with explicit actuation groups."""
        controls = self._control_entries()
        entries: list[dict[str, JsonValue]] = []
        for handle in self._articulations:
            control = handle.control
            if control is None:
                raise SimulationConfigError(
                    f"articulation {handle.name!r} has no control declaration; "
                    f"config-mode builders are not implemented in this slice"
                )
            entry: dict[str, JsonValue] = {"name": handle.name}
            ref = handle.ref
            if ref is not None and handle.name != ref.id:
                entry["params"] = {"profile": ref.id}
            entry["actuation"] = self._actuation_entries(control, handle)
            _declare_passive_joints(entry, controls)
            entries.append(entry)
        return entries

    def _actuation_entries(
        self,
        control: ControlSpec,
        handle: ArticulationHandle,
    ) -> list[JsonValue]:
        """One ``standard_pd`` actuation group per resolved control face."""
        groups: list[JsonValue] = []
        for face_name, face in control.faces.items():
            group: dict[str, JsonValue] = {
                "name": face.name,
                "law": "standard_pd",
                "joints": _json_names(handle.faces[face_name]),
                "gains": {
                    "kp": _json_limit(face.gains["kp"]),
                    "kd": _json_limit(face.gains["kd"]),
                },
            }
            if face.effort_limit is not None:
                group["effort_limit"] = _json_limit(face.effort_limit)
            if face.velocity_limit is not None:
                group["velocity_limit"] = _json_limit(face.velocity_limit)
            groups.append(group)
        return groups

    def _control_entries(self) -> list[dict[str, JsonValue]]:
        """The ``controls[]`` table with explicit joints and q0 overrides."""
        entries: list[dict[str, JsonValue]] = []
        for handle in self._articulations:
            control = handle.control
            if control is None:
                raise SimulationConfigError(
                    f"articulation {handle.name!r} has no control declaration; "
                    f"config-mode builders are not implemented in this slice"
                )
            for face_name, face in control.faces.items():
                group: dict[str, JsonValue] = {
                    "name": face.name,
                    "adapter": "manual",
                    "articulation": handle.name,
                    "joints": _json_names(handle.faces[face_name]),
                    "drive": face.drive,
                    "target": face.target,
                }
                if face.q0:
                    group["q0"] = _json_float_map(face.q0)
                entries.append(group)
        return entries

    def _camera_entries(self) -> list[dict[str, JsonValue]]:
        """The declared ``cameras[]`` entries (deep copies)."""
        return [copy.deepcopy(dict(handle.entry)) for handle in self._cameras]

    def _check_declarable(self, what: str) -> None:
        """Refuse a declaration on a config-mode or sealed builder."""
        if self._config_mode:
            raise ConfigModeError(
                f"this builder was created by from_config(); {what} is a "
                f"declaration-mode operation and cannot mutate a loaded run config"
            )
        self._check_unsealed(what)

    def _check_unsealed(self, what: str) -> None:
        """Refuse a declaration after the first successful build."""
        if self._sealed:
            raise BuilderSealedError(
                f"SimulationBuilder is sealed after build(); {what} is a pre-build declaration"
            )

    def _seal(self) -> None:
        """Seal this builder; the live transition (A3) calls this once."""
        self._sealed = True

    def build(self) -> Simulation:
        """Build the live simulation once and return the cached instance."""
        if self._sim is not None:
            return self._sim
        if self._config_mode:
            return self._build_from_config()
        cfg = self._to_cfg()
        world = cast("World", self._world)
        from unigym.run.api import Run
        from unigym.runner.external import ExternalRunner
        from unigym.runner.state import read_extra

        run = Run(
            cfg,
            world=world,
            pack_runtime=self.pack_runtime,
            no_artifacts=not self._artifacts,
            device=self._device,
        )
        try:
            instance = run.build()
            runner = run.runner("external")
            if not isinstance(runner, ExternalRunner):
                raise RuntimeError(
                    f"SimulationBuilder.runner('external') returned "
                    f"{type(runner).__name__}, expected ExternalRunner"
                )
            state = read_extra(instance.env)
            sim = Simulation(
                backend=self._backend,
                num_envs=self._num_envs,
                world=world,
                articulations=tuple(self._articulations),
                cameras=tuple(self._cameras),
                runner=runner,
                env=instance.env,
                run=run,
                owner=self._owner,
                state=state,
                obs_cache=dict(instance.env.obs_cache),
                t=0,
            )
        except BaseException:
            run.close()
            raise
        self._sim = sim
        self._seal()
        return sim

    def _build_from_config(self) -> Simulation:
        """Build from the stored Run and re-adopt handles from the live env."""
        run = self._run
        spec = self._spec
        if run is None or spec is None:
            raise SimulationConfigError(
                "a config-mode builder needs its stored Run and RunSpec; "
                "create it with SimulationBuilder.from_config(path)"
            )
        from unigym.runner.external import ExternalRunner
        from unigym.runner.state import read_extra
        from unigym.world import World

        try:
            instance = run.build()
            runner = run.runner("external")
            if not isinstance(runner, ExternalRunner):
                raise RuntimeError(
                    f"SimulationBuilder.runner('external') returned "
                    f"{type(runner).__name__}, expected ExternalRunner"
                )
            articulations, face_targets = self._adopt_config_articulations(instance.env)
            cameras = self._adopt_config_cameras(spec)
            try:
                world = World.from_config(spec.cfg, runtime=spec.pack_runtime)
            except ValueError as err:
                raise SimulationConfigError(
                    f"config-mode simulation cannot resolve its world from the run "
                    f"config: {err}"
                ) from err
            sim = Simulation(
                backend=self._backend,
                num_envs=self._num_envs,
                world=world,
                articulations=articulations,
                cameras=cameras,
                runner=runner,
                env=instance.env,
                run=run,
                owner=self._owner,
                state=read_extra(instance.env),
                obs_cache=dict(instance.env.obs_cache),
                t=0,
                face_targets=face_targets,
            )
        except BaseException:
            run.close()
            raise
        self._articulations = list(articulations)
        self._cameras = list(cameras)
        self._sim = sim
        self._seal()
        return sim

    def _adopt_config_articulations(
        self,
        env: EnvBackend,
    ) -> tuple[tuple[ArticulationHandle, ...], dict[str, str]]:
        """Re-adopt config-mode articulation handles from the live runtime."""
        from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime

        articulations = getattr(env, "articulations", None)
        controls = getattr(env, "controls", None)
        if not isinstance(articulations, Mapping) or not articulations:
            raise SimulationConfigError(
                "config-mode build needs a non-empty env.articulations mapping; "
                f"got {type(articulations).__name__}"
            )
        if not isinstance(controls, Mapping) or not controls:
            raise SimulationConfigError(
                "config-mode build needs a non-empty env.controls mapping; "
                f"got {type(controls).__name__}"
            )
        manual_controls: list[ControlRuntime] = []
        for key, runtime in controls.items():
            if not isinstance(runtime, ControlRuntime):
                raise SimulationConfigError(
                    f"env.controls[{key!r}] is {type(runtime).__name__}, "
                    f"expected ControlRuntime"
                )
            if runtime.adapter == "manual" and not runtime.prop:
                manual_controls.append(runtime)

        handles: list[ArticulationHandle] = []
        face_targets: dict[str, str] = {}
        for key, runtime in articulations.items():
            if not isinstance(runtime, ArticulationRuntime):
                raise SimulationConfigError(
                    f"env.articulations[{key!r}] is {type(runtime).__name__}, "
                    f"expected ArticulationRuntime"
                )
            name = str(runtime.name)
            profile = runtime.profile
            if profile is None:
                raise SimulationConfigError(
                    f"config-mode articulation {name!r} has no RobotProfile; "
                    f"the facade re-adopts profile-backed handles only"
                )
            faces: dict[str, tuple[str, ...]] = {}
            for control in manual_controls:
                if control.articulation != name:
                    continue
                face = control.name
                if face in face_targets:
                    raise SimulationConfigError(
                        f"control face {face!r} is declared by more than one "
                        f"manual control group; face names are global"
                    )
                faces[face] = tuple(control.joint_names)
                face_targets[face] = control.target
            handles.append(
                ArticulationHandle(
                    name=name,
                    ref=None,
                    profile=profile,
                    control=None,
                    faces=faces,
                    _owner=self._owner,
                )
            )
        return tuple(handles), face_targets

    def _adopt_config_cameras(self, spec: RunSpec) -> tuple[CameraHandle, ...]:
        """Re-adopt config-mode camera handles from the normalized config table."""
        declared = spec.cameras
        if declared is None:
            return ()
        if not isinstance(declared, (list, tuple)):
            raise SimulationConfigError(
                "a config-mode builder needs 'cameras' to be a list of entries, "
                f"got {type(declared).__name__}"
            )
        handles: list[CameraHandle] = []
        seen: set[str] = set()
        for index, entry in enumerate(declared):
            if not isinstance(entry, Mapping):
                raise SimulationConfigError(
                    f"cameras[{index}] is {type(entry).__name__}, expected a mapping"
                )
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SimulationConfigError(
                    f"cameras[{index}] needs a non-empty 'name', got {name!r}"
                )
            if name in seen:
                raise SimulationConfigError(
                    f"camera name {name!r} is declared twice; camera names must be unique"
                )
            seen.add(name)
            handles.append(
                CameraHandle(
                    name=name,
                    entry=copy.deepcopy(dict(entry)),
                    _owner=self._owner,
                )
            )
        return tuple(handles)

    def _check_face_names_available(self, label: str, control: ControlSpec) -> None:
        """Refuse control-face names another articulation already owns.

        ``controls[].name`` is a global group key, so two articulations that
        both fall back to the same default face name (for example
        ``joint_set="body"`` -> face ``"body"``) would collide in
        ``build_spec``.  Refuse here, while the caller can still name the face.
        """
        declared = {
            face_name: handle.name
            for handle in self._articulations
            if handle.control is not None
            for face_name in handle.control.faces
        }
        overlaps = sorted(face_name for face_name in control.faces if face_name in declared)
        if not overlaps:
            return
        names = ", ".join(repr(face_name) for face_name in overlaps)
        owners = ", ".join(
            f"{face_name!r} (articulation {declared[face_name]!r})" for face_name in overlaps
        )
        raise SimulationConfigError(
            f"articulation {label!r} reuses control face(s) {names} already declared by "
            f"{owners}; control-face names are global, so declare a distinct face= for "
            f"this articulation"
        )

    def _resolve_faces(
        self,
        label: str,
        profile: RobotProfile,
        control: ControlSpec,
    ) -> dict[str, tuple[str, ...]]:
        """Expand each control face against the profile and validate coverage."""
        declared = set(profile.full_joint_names)
        resolved: dict[str, tuple[str, ...]] = {}
        owners: dict[str, str] = {}
        for face_name, face in control.faces.items():
            if face.joint_set is not None:
                joint_set = face.joint_set
                if joint_set not in profile.joint_sets:
                    raise SimulationConfigError(
                        f"articulation {label!r} control face {face_name!r} selects "
                        f"joint_set {joint_set!r}, which the profile does not declare; "
                        f"declared sets: {sorted(profile.joint_sets)}"
                    )
                joints = tuple(profile.joint_sets[joint_set])
            else:
                if face.joints is None:
                    raise SimulationConfigError(
                        f"articulation {label!r} control face {face_name!r} has no joints"
                    )
                joints = face.joints
                unknown = sorted(set(joints) - declared)
                if unknown:
                    raise SimulationConfigError(
                        f"articulation {label!r} control face {face_name!r} names joints "
                        f"{unknown} that the profile does not declare; declared joints: "
                        f"{sorted(declared)}"
                    )
            unknown_q0 = sorted(set(face.q0) - set(joints))
            if unknown_q0:
                raise SimulationConfigError(
                    f"articulation {label!r} control face {face_name!r} q0 names joints "
                    f"{unknown_q0} outside the face joints {list(joints)}"
                )
            missing_q0 = sorted(
                joint
                for joint in joints
                if joint not in face.q0 and joint not in profile.default_joint_pos
            )
            if missing_q0:
                raise SimulationConfigError(
                    f"articulation {label!r} control face {face_name!r} has no q0 for "
                    f"joints {missing_q0}; declare control q0 or extend the profile "
                    f"default_joint_pos"
                )
            overlap = sorted(set(joints) & set(owners))
            if overlap:
                raise SimulationConfigError(
                    f"articulation {label!r} control faces {owners[overlap[0]]!r} and "
                    f"{face_name!r} overlap on joints {overlap}; faces must be disjoint"
                )
            for joint in joints:
                owners[joint] = face_name
            resolved[face_name] = joints
        return resolved
