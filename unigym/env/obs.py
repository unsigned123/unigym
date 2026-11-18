# ruff: noqa: D102
"""Typed, config-declared observation system owned by the environment.

The run config declares the observation sources once::

    observations:
      robot: [go2]
      contact: [feet]
      object: [box]
      camera: [ego]
      site: [gripper]
      command: true
      prev_action: true

The environment then publishes exactly those sources through an
:class:`ObservationRegistry` (``env.obs``).  Adapters and tasks read typed
views (``env.obs.robot("go2").joint_pos``) and never string-index a cache.
Canonical artifact keys (``articulation/...``, ``contact/...``,
``object/...``, ``site/...``) exist only at the serializer/meta boundary.

There is no per-adapter request layer and no raw cache.  This is a
deliberate breaking change.

The module is torch-only so it is importable without a simulator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

__all__ = [
    "CameraObs",
    "CommandObs",
    "ContactObs",
    "ObjectObs",
    "ObsConfigError",
    "ObsContractError",
    "ObsSnapshot",
    "ObservationConfig",
    "ObservationRegistry",
    "ObservationSchema",
    "ObservationSpec",
    "PrevActionObs",
    "RobotObs",
    "SiteObs",
    "OBS_SCHEMA_VERSION",
]

#: Observation schema version (meta.json ``observations.version``).
OBS_SCHEMA_VERSION = 1

_DTYPE_TO_TORCH = {
    "f32": torch.float32,
    "u8": torch.uint8,
    "bool": torch.bool,
}
_TORCH_TO_DTYPE = {v: k for k, v in _DTYPE_TO_TORCH.items()}
_PASSTHROUGH_DTYPES = frozenset({"json"})


class ObsConfigError(ValueError):
    """Raised when an ``observations:`` declaration is malformed."""


class ObsContractError(AssertionError):
    """Raised when a published observation violates its schema."""


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
_SECTIONS = ("robot", "contact", "object", "camera", "site")


def _names(value: object, section: str) -> tuple[str, ...]:
    """Normalize one observation name list (str or list of strings)."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raw: Sequence[object] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        raw = tuple(value)
    else:
        raise ObsConfigError(
            f"observations.{section} must be a name or a list of names, got {value!r}"
        )
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ObsConfigError(
                f"observations.{section} entries must be non-empty strings, got {item!r}"
            )
        names.append(item)
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ObsConfigError(f"observations.{section} declares duplicate names: {dupes}")
    return tuple(names)


def _flag(value: object, key: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ObsConfigError(f"observations.{key} must be a bool, got {value!r}")
    return value


@dataclass(frozen=True)
class ObservationConfig:
    """The parsed ``observations:`` declaration of one run."""

    robot: tuple[str, ...]
    contact: tuple[str, ...] = ()
    object: tuple[str, ...] = ()
    camera: tuple[str, ...] = ()
    site: tuple[str, ...] = ()
    command: bool = False
    prev_action: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> ObservationConfig:
        """Parse one ``observations:`` mapping (required; robot non-empty)."""
        if value is None:
            raise ObsConfigError(
                "no `observations:` section declared; declare at least "
                "`observations: {robot: [<articulation>]}`"
            )
        if isinstance(value, ObservationConfig):
            return value
        if not isinstance(value, Mapping):
            raise ObsConfigError(f"observations must be a mapping, got {value!r}")
        unknown = sorted(set(value) - set(_SECTIONS) - {"command", "prev_action"})
        if unknown:
            raise ObsConfigError(
                f"observations has unknown key(s) {unknown}; allowed: "
                f"{list(_SECTIONS) + ['command', 'prev_action']}"
            )
        robot = _names(value.get("robot"), "robot")
        if not robot:
            raise ObsConfigError(
                "observations.robot must name at least one articulation "
                "(there is no implicit default)"
            )
        return cls(
            robot=robot,
            contact=_names(value.get("contact"), "contact"),
            object=_names(value.get("object"), "object"),
            camera=_names(value.get("camera"), "camera"),
            site=_names(value.get("site"), "site"),
            command=_flag(value.get("command"), "command"),
            prev_action=_flag(value.get("prev_action"), "prev_action"),
        )

    def to_mapping(self) -> dict:
        """JSON-safe round-trip form (the ``observations:`` config section)."""
        return {
            "robot": list(self.robot),
            "contact": list(self.contact),
            "object": list(self.object),
            "camera": list(self.camera),
            "site": list(self.site),
            "command": self.command,
            "prev_action": self.prev_action,
        }

    def validate(
        self,
        *,
        articulations: Iterable[str],
        contacts: Iterable[str],
        objects: Iterable[str],
        cameras: Iterable[str],
        sites: Iterable[str] | None = None,
    ) -> None:
        """Validate declared names against the run's declared sources.

        ``sites`` is the backend-resolved site inventory.  ``None`` defers
        site validation to the backend's model/scene resolution (the run
        builder does not know the compiled site table).
        """
        declared = {
            "robot": (set(articulations), "articulations"),
            "contact": (set(contacts), "contacts"),
            "object": (set(objects), "props"),
            "camera": (set(cameras), "cameras"),
        }
        if sites is not None:
            declared["site"] = (set(sites), "sites")
        for section, (available, what) in declared.items():
            if section == "site" and sites is None:
                continue
            names = getattr(self, section)
            unknown = [n for n in names if n not in available]
            if unknown:
                raise ObsConfigError(
                    f"observations.{section} names unknown {what}: {unknown}; "
                    f"declared: {sorted(available)}"
                )


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservationSpec:
    """Metadata/shape contract of one observation artifact key."""

    key: str
    shape: tuple[int, ...] | None
    dtype: str
    joint_names: tuple[str, ...] | None = None
    unit: str | None = None
    frame: str | None = None
    source: str = "robot"
    record: bool = True


class ObservationSchema:
    """The run's observation schema (keys, shapes, artifact contract)."""

    def __init__(self, entries: Mapping[str, ObservationSpec]):
        self.entries: dict[str, ObservationSpec] = dict(entries)
        self.version = OBS_SCHEMA_VERSION

    # ---- container surface ---------------------------------------------- #
    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> ObservationSpec:
        return self.entries[key]

    def __iter__(self):
        return iter(sorted(self.entries))

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> list[str]:
        """Every schema key, sorted."""
        return sorted(self.entries)

    def record_keys(self) -> list[str]:
        """Canonical artifact keys emitted by ``ObsSnapshot.record()``."""
        return sorted(k for k, spec in self.entries.items() if spec.record)

    def shape(self, key: str) -> tuple[int, ...] | None:
        return self.entries[key].shape

    def joint_names(self, key: str) -> tuple[str, ...] | None:
        return self.entries[key].joint_names

    def to_meta(self) -> dict:
        """JSON-safe meta.json snapshot (``meta["observations"]``)."""
        return {
            "version": self.version,
            "keys": {
                spec.key: {
                    "shape": None if spec.shape is None else [int(d) for d in spec.shape],
                    "dtype": spec.dtype,
                    "joint_names": None if spec.joint_names is None else list(spec.joint_names),
                    "unit": spec.unit,
                    "frame": spec.frame,
                    "source": spec.source,
                    "record": spec.record,
                }
                for spec in (self.entries[k] for k in sorted(self.entries))
            },
        }

    # ---- validation ------------------------------------------------------ #
    def validate_tensor(self, key: str, value: torch.Tensor, num_envs: int) -> None:
        """Validate one published tensor against its spec (batch axis included)."""
        spec = self.entries.get(key)
        if spec is None:
            raise ObsContractError(f"observation key {key!r} is not in the schema")
        if not isinstance(value, torch.Tensor):
            raise ObsContractError(
                f"observation {key!r} must be a torch.Tensor, got {type(value).__name__}"
            )
        expected = None if spec.shape is None else (num_envs, *spec.shape)
        if expected is not None and tuple(value.shape) != expected:
            raise ObsContractError(f"observation {key!r} shape {tuple(value.shape)} != {expected}")
        if value.ndim == 0 or value.shape[0] != num_envs:
            raise ObsContractError(
                f"observation {key!r} must have {num_envs} env rows, "
                f"got shape {tuple(value.shape)}"
            )
        want = _DTYPE_TO_TORCH.get(spec.dtype)
        if want is not None and value.dtype != want:
            raise ObsContractError(f"observation {key!r} dtype {value.dtype} != {want}")


# --------------------------------------------------------------------------- #
# typed views
# --------------------------------------------------------------------------- #
class _View:
    """Base for one typed view over an :class:`ObsSnapshot`."""

    def __init__(self, snapshot: ObsSnapshot, name: str | None = None):
        self._snapshot = snapshot
        self.name = name

    def _value(self, key: str) -> torch.Tensor:
        return self._snapshot.value(key)

    def _optional(self, key: str) -> torch.Tensor | None:
        return self._snapshot.get(key)

    @property
    def key(self) -> str:
        """Canonical artifact key prefix of this view."""
        raise NotImplementedError


class RobotObs(_View):
    """Typed robot/articulation state for one declared articulation.

    ``joint_names`` is the full articulation order (``joint_pos_rel``'s
    order); ``policy_joint_names`` is the policy subset order of the recorded
    ``joint_pos`` / ``joint_vel`` / ``applied_torque`` leaves.
    """

    def __init__(
        self,
        snapshot: ObsSnapshot,
        name: str,
        joint_names: tuple[str, ...],
        policy_joint_names: tuple[str, ...] | None = None,
    ):
        super().__init__(snapshot, name)
        self.joint_names = joint_names
        self.policy_joint_names = joint_names if policy_joint_names is None else policy_joint_names

    @property
    def key(self) -> str:
        return f"articulation/{self.name}"

    def _robot(self, leaf: str) -> torch.Tensor:
        return self._value(f"{self.key}/{leaf}")

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._robot("joint_pos")

    @property
    def joint_pos_rel(self) -> torch.Tensor:
        return self._robot("joint_pos_rel")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._robot("joint_vel")

    @property
    def joint_vel_full(self) -> torch.Tensor:
        """Full-articulation joint velocity (all joints, asset order)."""
        return self._robot("joint_vel_full")

    @property
    def applied_torque(self) -> torch.Tensor:
        return self._robot("applied_torque")

    @property
    def gravity_b(self) -> torch.Tensor:
        return self._robot("gravity_b")

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self._robot("base_pos_w")

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self._robot("base_quat_w")

    @property
    def base_lin_vel_b(self) -> torch.Tensor:
        return self._robot("base_lin_vel_b")

    @property
    def base_ang_vel_b(self) -> torch.Tensor:
        return self._robot("base_ang_vel_b")

    @property
    def base_lin_vel_w(self) -> torch.Tensor:
        return self._robot("base_lin_vel_w")

    @property
    def base_ang_vel_w(self) -> torch.Tensor:
        return self._robot("base_ang_vel_w")

    def joints(self, names: Sequence[str], leaf: str = "joint_pos_rel") -> torch.Tensor:
        """Gather a named joint subset from one joint-vector leaf.

        The requested order is the served order; names must be unique and
        published by this articulation.
        """
        if isinstance(names, (str, bytes)):
            raise TypeError("RobotObs.joints expects a sequence of names, not a string")
        wanted = tuple(names)
        if not wanted:
            raise ObsContractError("RobotObs.joints needs at least one joint name")
        dupes = sorted({n for n in wanted if wanted.count(n) > 1})
        if dupes:
            raise ObsContractError(f"RobotObs.joints has duplicate names {dupes}")
        order = (
            self.joint_names
            if leaf in ("joint_pos_rel", "joint_vel_full")
            else self.policy_joint_names
        )
        index = {n: i for i, n in enumerate(order)}
        missing = [n for n in wanted if n not in index]
        if missing:
            raise ObsContractError(
                f"articulation {self.name!r} has no joints {missing} for "
                f"{leaf!r}; published: {list(order)}"
            )
        value = getattr(self, leaf)
        gather = torch.tensor([index[n] for n in wanted], dtype=torch.int64, device=value.device)
        return value.index_select(-1, gather)


class ContactObs(_View):
    """Typed contact-group reading for one declared contact group."""

    @property
    def key(self) -> str:
        return f"contact/{self.name}"

    @property
    def value(self) -> torch.Tensor:
        return self._value(self.key)

    def any(self) -> torch.Tensor:
        """Per-env any-contact mask (``(N,)`` bool)."""
        value = self.value
        if value.dtype == torch.bool:
            return value.reshape(value.shape[0], -1).any(dim=-1)
        return (value.reshape(value.shape[0], -1) > 0).any(dim=-1)

    def detail(self) -> list[dict] | None:
        """Per-contact records on explicit access (lazy; absent -> ``None``).

        The observation refresh never reads detail: a backend without a
        per-contact face answers ``None`` here, and an explicit capability
        failure from a declared detail provider propagates only to this call.
        """
        return self._snapshot.contact_detail(self.name)


class ObjectObs(_View):
    """Typed prop/object state for one declared object."""

    @property
    def key(self) -> str:
        return f"object/{self.name}"

    @property
    def pos_w(self) -> torch.Tensor:
        return self._value(f"{self.key}/pos")

    @property
    def quat_w(self) -> torch.Tensor:
        return self._value(f"{self.key}/quat")

    @property
    def joints(self) -> Mapping[str, torch.Tensor]:
        prefix = f"{self.key}/joints/"
        return MappingProxyType(
            {
                key[len(prefix) :]: value
                for key, value in self._snapshot.values.items()
                if key.startswith(prefix)
            }
        )

    def in_robot_frame(self, robot: str | RobotObs) -> tuple[torch.Tensor, torch.Tensor]:
        """Object root pose expressed in one robot's base frame."""
        from ..core import quat as _quat

        view = robot if isinstance(robot, RobotObs) else self._snapshot.robot(robot)
        base_pos = view.base_pos_w
        base_quat = view.base_quat_w
        pos = self.pos_w
        rel = pos - base_pos
        # rotate world-offset into the base frame: R_b^T @ rel
        rot_t = _quat.to_rotmat(base_quat).transpose(-1, -2)
        pos_b = (rot_t @ rel.unsqueeze(-1)).squeeze(-1)
        quat_b = _quat.mul(_quat.conj(base_quat), self.quat_w)
        return pos_b, quat_b


@dataclass(frozen=True)
class _CameraFrame:
    value: torch.Tensor | None


class CameraObs(_View):
    """Typed camera frames for one declared camera rig."""

    @property
    def key(self) -> str:
        return f"cam/{self.name}"

    def frame(self, kind: str) -> torch.Tensor | None:
        if kind not in ("rgb", "depth"):
            raise ObsContractError(f"camera frame kind must be 'rgb' or 'depth', got {kind!r}")
        return self._optional(f"{self.key}/{kind}")

    @property
    def rgb(self) -> torch.Tensor | None:
        return self.frame("rgb")

    @property
    def depth(self) -> torch.Tensor | None:
        return self.frame("depth")

    @property
    def t(self) -> int:
        return self._snapshot.t


class SiteObs(_View):
    """Typed site position for one declared record site."""

    @property
    def key(self) -> str:
        return f"site/{self.name}"

    @property
    def pos_w(self) -> torch.Tensor:
        return self._value(f"{self.key}/pos")


class _SlotObs(_View):
    """Typed runtime slot (command / previous action)."""

    SLOT = ""

    @property
    def key(self) -> str:
        return self.SLOT

    @property
    def value(self) -> torch.Tensor | None:
        return self._optional(self.SLOT)

    def __getitem__(self, index: object) -> torch.Tensor:
        return self.value[index]

    def __bool__(self) -> bool:
        return self.value is not None

    @property
    def shape(self) -> tuple[int, ...] | None:
        value = self.value
        return None if value is None else tuple(value.shape)

    @property
    def device(self):
        value = self.value
        return None if value is None else value.device


class CommandObs(_SlotObs):
    """The run's post-sampler command slot."""

    SLOT = "command"


class PrevActionObs(_SlotObs):
    """The adapter's previous raw action slot."""

    SLOT = "prev_action"


def _robot_entries(
    name: str,
    full_joint_names: Sequence[str],
    policy_joint_names: Sequence[str],
) -> dict[str, ObservationSpec]:
    full = tuple(str(n) for n in full_joint_names)
    policy = tuple(str(n) for n in policy_joint_names) or full
    prefix = f"articulation/{name}"
    p = len(policy)
    return {
        f"{prefix}/joint_pos": ObservationSpec(
            f"{prefix}/joint_pos",
            (p,),
            "f32",
            joint_names=policy,
            unit="rad",
            source="robot",
        ),
        f"{prefix}/joint_pos_rel": ObservationSpec(
            f"{prefix}/joint_pos_rel",
            (len(full),),
            "f32",
            joint_names=full,
            unit="rad",
            source="robot",
            record=False,
        ),
        f"{prefix}/joint_vel": ObservationSpec(
            f"{prefix}/joint_vel", (p,), "f32", joint_names=policy, unit="rad/s", source="robot"
        ),
        f"{prefix}/joint_vel_full": ObservationSpec(
            f"{prefix}/joint_vel_full",
            (len(full),),
            "f32",
            joint_names=full,
            unit="rad/s",
            source="robot",
            record=False,
        ),
        f"{prefix}/applied_torque": ObservationSpec(
            f"{prefix}/applied_torque",
            (p,),
            "f32",
            joint_names=policy,
            unit="N·m",
            source="robot",
        ),
        f"{prefix}/gravity_b": ObservationSpec(
            f"{prefix}/gravity_b", (3,), "f32", unit="g", frame="base", source="computed"
        ),
        f"{prefix}/base_pos_w": ObservationSpec(
            f"{prefix}/base_pos_w", (3,), "f32", unit="m", frame="world", source="robot"
        ),
        f"{prefix}/base_quat_w": ObservationSpec(
            f"{prefix}/base_quat_w", (4,), "f32", frame="world", source="robot"
        ),
        f"{prefix}/base_lin_vel_w": ObservationSpec(
            f"{prefix}/base_lin_vel_w", (3,), "f32", unit="m/s", frame="world", source="robot"
        ),
        f"{prefix}/base_ang_vel_w": ObservationSpec(
            f"{prefix}/base_ang_vel_w", (3,), "f32", unit="rad/s", frame="world", source="robot"
        ),
        f"{prefix}/base_lin_vel_b": ObservationSpec(
            f"{prefix}/base_lin_vel_b", (3,), "f32", unit="m/s", frame="base", source="robot"
        ),
        f"{prefix}/base_ang_vel_b": ObservationSpec(
            f"{prefix}/base_ang_vel_b", (3,), "f32", unit="rad/s", frame="base", source="robot"
        ),
        f"{prefix}/base_yaw": ObservationSpec(
            f"{prefix}/base_yaw",
            (1,),
            "f32",
            unit="rad",
            frame="world",
            source="robot",
            record=False,
        ),
    }


def build_observation_schema(
    config: ObservationConfig,
    *,
    full_joint_names: Mapping[str, Sequence[str]],
    policy_joint_names: Mapping[str, Sequence[str]] | None = None,
    contacts: Mapping[str, tuple[tuple[int, ...] | None, str]] | None = None,
    objects: Mapping[str, Sequence[str]] | None = None,
    cameras: Mapping[str, Mapping[str, tuple[tuple[int, ...], str]]] | None = None,
    sites: Iterable[str] = (),
) -> ObservationSchema:
    """Assemble the run's observation schema from the declared sources.

    ``contacts`` maps name -> ``(shape, dtype)``; ``objects`` maps name ->
    declared joint names; ``cameras`` maps camera name -> ``{key: (shape,
    dtype)}`` (``cam/<name>/rgb`` / ``cam/<name>/depth``).
    """
    entries: dict[str, ObservationSpec] = {}
    policy_names = policy_joint_names or {}
    for name in config.robot:
        if name not in full_joint_names:
            raise ObsConfigError(f"observations.robot names unknown articulation {name!r}")
        entries.update(
            _robot_entries(
                name,
                full_joint_names[name],
                policy_names.get(name, full_joint_names[name]),
            )
        )
    contact_table = contacts or {}
    for name in config.contact:
        if name not in contact_table:
            raise ObsConfigError(f"observations.contact names unknown group {name!r}")
        shape, dtype = contact_table[name]
        entries[f"contact/{name}"] = ObservationSpec(
            f"contact/{name}", shape, dtype, frame="world", source="contact"
        )
    object_table = objects or {}
    for name in config.object:
        if name not in object_table:
            raise ObsConfigError(f"observations.object names unknown prop {name!r}")
        entries[f"object/{name}/pos"] = ObservationSpec(
            f"object/{name}/pos", (3,), "f32", unit="m", frame="world", source="object"
        )
        entries[f"object/{name}/quat"] = ObservationSpec(
            f"object/{name}/quat", (4,), "f32", frame="world", source="object"
        )
        for joint in object_table[name]:
            entries[f"object/{name}/joints/{joint}"] = ObservationSpec(
                f"object/{name}/joints/{joint}",
                (1,),
                "f32",
                unit="rad",
                frame="world",
                source="object",
            )
    camera_table = cameras or {}
    for name in config.camera:
        frames = camera_table.get(name)
        if not frames:
            raise ObsConfigError(f"observations.camera names unknown camera {name!r}")
        for key, (shape, dtype) in frames.items():
            entries[key] = ObservationSpec(
                key, tuple(shape), dtype, frame="sensor", source="camera", record=False
            )
    for name in config.site:
        entries[f"site/{name}/pos"] = ObservationSpec(
            f"site/{name}/pos", (3,), "f32", unit="m", frame="world", source="site"
        )
    if config.command:
        entries["command"] = ObservationSpec("command", None, "f32", source="sampler", record=False)
    if config.prev_action:
        entries["prev_action"] = ObservationSpec(
            "prev_action", None, "f32", source="adapter", record=False
        )
    return ObservationSchema(entries)


# --------------------------------------------------------------------------- #
# snapshot
# --------------------------------------------------------------------------- #
class ObsSnapshot:
    """Immutable view of one tick's published observations."""

    def __init__(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        schema: ObservationSchema,
        num_envs: int,
        device: str | torch.device,
        t: int = 0,
        details: Mapping[str, list[dict]] | None = None,
        detail_providers: Mapping[str, object] | None = None,
    ):
        self._values = MappingProxyType(dict(values))
        self.schema = schema
        self.num_envs = int(num_envs)
        self.device = device
        self.t = int(t)
        self._details = MappingProxyType({k: list(v) for k, v in (details or {}).items()})
        self._detail_providers = MappingProxyType(dict(detail_providers or {}))

    @property
    def values(self) -> Mapping[str, torch.Tensor]:
        return self._values

    def get(self, key: str) -> torch.Tensor | None:
        return self._values.get(key)

    def value(self, key: str) -> torch.Tensor:
        value = self._values.get(key)
        if value is None:
            raise ObsContractError(
                f"observation {key!r} was not published this step; "
                f"published: {sorted(self._values)}"
            )
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.value(key)

    def keys(self) -> list[str]:
        return sorted(self._values)

    # ---- typed accessors ------------------------------------------------- #
    def robot(self, name: str) -> RobotObs:
        prefix = f"articulation/{name}"
        if prefix + "/joint_pos" not in self._values:
            raise ObsContractError(
                f"robot observation {name!r} is not declared/published; "
                f"declared robots: "
                f"{sorted(k for k in self._values if k.startswith('articulation/'))}"
            )
        joint_names = self.schema.joint_names(f"{prefix}/joint_pos_rel") or ()
        policy_names = self.schema.joint_names(f"{prefix}/joint_pos") or ()
        return RobotObs(self, name, tuple(joint_names), tuple(policy_names))

    def contact(self, name: str) -> ContactObs:
        if f"contact/{name}" not in self._values:
            raise ObsContractError(
                f"contact observation {name!r} is not declared/published; "
                f"declared contacts: {sorted(k for k in self._values if k.startswith('contact/'))}"
            )
        return ContactObs(self, name)

    def object(self, name: str) -> ObjectObs:
        if f"object/{name}/pos" not in self._values:
            raise ObsContractError(
                f"object observation {name!r} is not declared/published; "
                f"declared objects: "
                f"{sorted({k.split('/')[1] for k in self._values if k.startswith('object/')})}"
            )
        return ObjectObs(self, name)

    def camera(self, name: str) -> CameraObs:
        prefix = f"cam/{name}"
        if prefix + "/rgb" not in self._values and prefix + "/depth" not in self._values:
            raise ObsContractError(
                f"camera observation {name!r} is not declared/published; "
                f"declared cameras: "
                f"{sorted({k.split('/')[1] for k in self._values if k.startswith('cam/')})}"
            )
        return CameraObs(self, name)

    def site(self, name: str) -> SiteObs:
        if f"site/{name}/pos" not in self._values:
            raise ObsContractError(
                f"site observation {name!r} is not declared/published; "
                f"declared sites: "
                f"{sorted({k.split('/')[1] for k in self._values if k.startswith('site/')})}"
            )
        return SiteObs(self, name)

    @property
    def command(self) -> CommandObs:
        return CommandObs(self)

    @property
    def prev_action(self) -> PrevActionObs:
        return PrevActionObs(self)

    # ---- artifact boundary ---------------------------------------------- #
    def record(self) -> dict[str, torch.Tensor]:
        """Canonical artifact keys only (no cameras, slots or bookkeeping)."""
        return {
            key: value for key, value in self._values.items() if self.schema.entries[key].record
        }

    def contact_detail(self, name: str) -> list[dict] | None:
        """Materialize one group's detail lazily (explicit access only)."""
        provider = self._detail_providers.get(name)
        if provider is not None:
            records = provider()
            return None if records is None else [dict(item) for item in records]
        detail = self._details.get(name)
        return None if detail is None else [dict(item) for item in detail]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
class ObservationRegistry:
    """The environment-owned observation registry (``env.obs``)."""

    def __init__(
        self,
        config: ObservationConfig,
        schema: ObservationSchema,
        *,
        num_envs: int,
        device: str | torch.device,
    ):
        self.config = config
        self.schema = schema
        self.num_envs = int(num_envs)
        self.device = device
        self.t = 0
        self._values: dict[str, torch.Tensor] = {}
        self._details: dict[str, list[dict]] = {}
        self._detail_providers: dict[str, object] = {}

    # ---- publication ----------------------------------------------------- #
    def set(self, key: str, value: torch.Tensor) -> None:
        """Publish one tensor (validated against the schema)."""
        self.schema.validate_tensor(key, value, self.num_envs)
        self._values[key] = value

    def set_many(self, values: Mapping[str, torch.Tensor]) -> None:
        for key, value in values.items():
            self.set(key, value)

    def set_robot(self, name: str, leaf: str, value: torch.Tensor) -> None:
        self.set(f"articulation/{name}/{leaf}", value)

    def set_contact(self, name: str, value: torch.Tensor) -> None:
        self.set(f"contact/{name}", value)

    def set_contact_detail(self, name: str, records: Sequence[Mapping]) -> None:
        self._details[name] = [dict(item) for item in records]

    def set_contact_detail_provider(self, name: str, provider) -> None:
        """Register a lazy per-contact detail provider (explicit access only)."""
        self._detail_providers[name] = provider

    def set_object(self, name: str, leaf: str, value: torch.Tensor) -> None:
        self.set(f"object/{name}/{leaf}", value)

    def set_camera(self, key: str, value: torch.Tensor) -> None:
        self.set(key, value)

    def set_site(self, name: str, value: torch.Tensor) -> None:
        self.set(f"site/{name}/pos", value)

    def set_command(self, value: torch.Tensor | None) -> None:
        if value is not None:
            self.set("command", value)

    def set_prev_action(self, value: torch.Tensor | None) -> None:
        if value is not None:
            self.set("prev_action", value)

    # ---- access ----------------------------------------------------------- #
    def _require_declared(self, prefix: str, name: str, what: str) -> None:
        if not any(key.startswith(prefix) for key in self.schema.entries):
            raise ObsContractError(f"observation {what} {name!r} is not declared in this run")

    def robot(self, name: str) -> RobotObs:
        self._require_declared("articulation/", name, "robot")
        return self.snapshot().robot(name)

    def contact(self, name: str) -> ContactObs:
        self._require_declared("contact/", name, "contact")
        return self.snapshot().contact(name)

    def object(self, name: str) -> ObjectObs:
        self._require_declared("object/", name, "object")
        return self.snapshot().object(name)

    def camera(self, name: str) -> CameraObs:
        self._require_declared("cam/", name, "camera")
        return self.snapshot().camera(name)

    def site(self, name: str) -> SiteObs:
        self._require_declared("site/", name, "site")
        return self.snapshot().site(name)

    @property
    def command(self) -> CommandObs:
        return self.snapshot().command

    @property
    def prev_action(self) -> PrevActionObs:
        return self.snapshot().prev_action

    def snapshot(self) -> ObsSnapshot:
        """An immutable copy of this tick's observation values."""
        return ObsSnapshot(
            self._values,
            schema=self.schema,
            num_envs=self.num_envs,
            device=self.device,
            t=self.t,
            details=self._details,
            detail_providers=self._detail_providers,
        )

    def record(self) -> dict[str, torch.Tensor]:
        """Current canonical artifact keys (schema record face)."""
        return {
            key: value for key, value in self._values.items() if self.schema.entries[key].record
        }

    # ---- rewind ---------------------------------------------------------- #
    def rewind_state(self) -> dict[str, object]:
        """Snapshot-safe registry payload (values, slots, detail, t)."""
        return {
            "t": int(self.t),
            "values": dict(self._values),
            "details": {k: [dict(item) for item in v] for k, v in self._details.items()},
        }

    def restore_rewind_state(self, state: Mapping[str, object]) -> None:
        """Restore a :meth:`rewind_state` payload verbatim."""
        self.t = int(state.get("t", 0))
        self._values.clear()
        self._values.update(dict(state.get("values") or {}))
        self._details.clear()
        for name, records in dict(state.get("details") or {}).items():
            self._details[str(name)] = [dict(item) for item in records]
        # providers are construction-time wiring and survive rewind
