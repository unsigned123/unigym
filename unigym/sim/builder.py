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
from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from unigym._typing import JsonValue

from .control import ControlSpec
from .errors import (
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

if TYPE_CHECKING:
    from unigym.packs.packset import PackRuntime, PackSet, RobotRef
    from unigym.robots.base import RobotProfile
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
        self._pack_set = pack_set
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

    @property
    def world(self) -> World | None:
        """The one world declaration owned by this builder."""
        return self._world

    @world.setter
    def world(self, world: World) -> None:
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
        return self._pack_set.to_runtime()

    @property
    def articulations(self) -> tuple[ArticulationHandle, ...]:
        """Declared articulations in declaration order."""
        return tuple(self._articulations)

    @property
    def cameras(self) -> tuple[CameraHandle, ...]:
        """Declared cameras in declaration order."""
        return tuple(self._cameras)

    def add_articulation(
        self,
        ref: RobotRef,
        *,
        name: str | None = None,
        control: ControlSpec | None = None,
    ) -> ArticulationHandle:
        """Declare one pack robot and resolve every control face against its profile."""
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
        profile = load_profile_for_ref(self._pack_set, ref)
        faces = self._resolve_faces(label, profile, control)
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

    def build(self) -> NoReturn:
        """Transition to live state (implemented by the next facade slice)."""
        raise NotImplementedError(
            "SimulationBuilder.build is implemented by the simulation slice; "
            "this builder is declaration-only"
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
