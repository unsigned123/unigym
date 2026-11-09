"""The neutral model IR: one static description for every backend.

The dataclasses here are pure data (``frozen=True, slots=True``): declared
fields are populated before compilation, compiled fields stay ``None``
until a backend's ``describe`` fills them.  A ``ModelIR`` never invents a
value -- a ``None`` means "this backend has no such concept" or "not
compiled yet".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, TypeVar

import numpy as np

from ._serialize import (
    JsonValue,
    _is_mapping,
    decode,
    encode,
    from_row,
    row_dict,
)
from .errors import (
    ModelIRError,
    StaleModelError,
    UnknownExtensionError,
)
from .table import Table

if TYPE_CHECKING:
    from unigym.backends.registry import Trait
    from unigym.core.quat_types import QuatWXYZ

    from .diff import ModelDiff

__all__ = [
    "Actuator",
    "Body",
    "Camera",
    "EqSlot",
    "Geom",
    "GeomContact",
    "GeomGroup",
    "Joint",
    "Material",
    "MjExt",
    "ModelExt",
    "ModelExtension",
    "ModelIR",
    "ModelIRError",
    "Site",
    "StaleModelError",
    "Texture",
    "UnknownExtensionError",
    "UnparsedExt",
    "UsdExt",
]


class _NamedRow(Protocol):
    """A table row carrying its compiled global name."""

    @property
    def name(self) -> str:
        """The row's compiled global name."""
        ...


_RowT = TypeVar("_RowT")


class ModelExtension(Protocol):
    """One backend namespace payload that ``ModelIR.ext`` can carry."""

    def validate(self) -> ModelExtension:
        """Validate this extension and return it."""
        ...

    def to_dict(self) -> Mapping[str, JsonValue]:
        """Return the extension's canonical dictionary form."""
        ...


def _empty_json_map() -> dict[str, JsonValue]:
    """An empty JSON mapping with concrete key/value types."""
    return {}


def _empty_str_map() -> dict[str, str]:
    """An empty string mapping with concrete key/value types."""
    return {}


def _empty_object_map() -> dict[str, object]:
    """An empty map of opaque metadata values."""
    return {}


def _empty_ext_map() -> dict[str, ModelExtension]:
    """An empty extension-namespace mapping."""
    return {}


class _PrettyPrinter(Protocol):
    """The slice of the IPython pretty-printer protocol this module uses."""

    def text(self, text: str) -> None:
        """Append one plain-text chunk."""
        ...


def _quat_wxyz(values: Iterable[float]) -> QuatWXYZ:
    """Return a read-only ``QuatWXYZ`` over a fresh array of *values*.

    The stored ``quat`` tuple stays the wire form; this typed view is
    built on demand (lazy import keeps ``unigym.model`` engine-free).
    """
    from unigym.core.quat_types import QuatWXYZ

    data = np.array(values, dtype=float)
    data.flags.writeable = False
    return QuatWXYZ(data)


@dataclass(frozen=True, slots=True)
class Body:
    """One rigid body (world included) with declared and compiled fields."""

    name: str
    parent: str | None = None
    path: tuple[str, ...] = ()
    qualname: str = ""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mocap: bool = False
    explicitinertial: bool = False
    body_id: int | None = None
    mass: float | None = None
    inertia: tuple[float, float, float] | None = None
    ipos: tuple[float, float, float] | None = None
    iquat: tuple[float, float, float, float] | None = None

    @property
    def quat_wxyz(self) -> QuatWXYZ:
        """The body's ``(w, x, y, z)`` orientation as a typed view."""
        return _quat_wxyz(self.quat)


@dataclass(frozen=True, slots=True)
class Joint:
    """One joint; addresses and actuator ownership are compiled-only."""

    name: str
    type: str
    body: str
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    range: tuple[float, float] | None = None
    limited: bool = False
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    damping: float | None = None
    armature: float | None = None
    actuator: str | None = None
    qpos_adr: int | None = None
    dof_adr: int | None = None
    path: tuple[str, ...] = ()
    qualname: str = ""


@dataclass(frozen=True, slots=True)
class Geom:
    """One collision/visual geom attached to a body."""

    name: str
    type: str
    body: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str | None = None
    group: int = 0
    contype: int = 1
    conaffinity: int = 1
    density: float | None = None
    mass: float | None = None
    geom_id: int | None = None
    path: tuple[str, ...] = ()
    qualname: str = ""


@dataclass(frozen=True, slots=True)
class Site:
    """One site marker attached to a body."""

    name: str
    type: str
    body: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str | None = None
    group: int = 0
    site_id: int | None = None
    path: tuple[str, ...] = ()
    qualname: str = ""

    @property
    def quat_wxyz(self) -> QuatWXYZ:
        """The site's ``(w, x, y, z)`` orientation as a typed view."""
        return _quat_wxyz(self.quat)


@dataclass(frozen=True, slots=True)
class Camera:
    """One camera attached to a body."""

    name: str
    body: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fovy: float | None = None
    mode: str = "fixed"
    targetbody: str | None = None
    camera_id: int | None = None
    path: tuple[str, ...] = ()
    qualname: str = ""

    @property
    def quat_wxyz(self) -> QuatWXYZ:
        """The camera's ``(w, x, y, z)`` orientation as a typed view."""
        return _quat_wxyz(self.quat)


@dataclass(frozen=True, slots=True)
class Actuator:
    """One actuator, labelled by the kind of target it drives."""

    name: str
    target: str = ""
    trntype: str = ""
    kind: str = "other"
    owner: str | None = None
    ctrlrange: tuple[float, float] | None = None
    gear: tuple[float, ...] = ()
    gainprm: tuple[float, ...] = ()
    biasprm: tuple[float, ...] = ()
    actuator_id: int | None = None
    path: tuple[str, ...] = ()
    qualname: str = ""


@dataclass(frozen=True, slots=True)
class Material:
    """One material declaration with its texture bindings."""

    name: str
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    textures: tuple[str, ...] = ()
    reflectance: float | None = None
    roughness: float | None = None
    metallic: float | None = None
    emission: float | None = None
    specular: float | None = None
    shininess: float | None = None
    texrepeat: tuple[float, float] = (1.0, 1.0)
    texuniform: bool = False


@dataclass(frozen=True, slots=True)
class Texture:
    """One MuJoCo texture entity (a ``mj`` extension row, never base IR).

    ``file`` is always the basename: the digest is host-independent.  A
    ``None`` field means the source did not declare that value.
    """

    name: str
    type: str = "2d"
    builtin: str | None = None
    file: str | None = None
    width: int | None = None
    height: int | None = None
    nchannel: int | None = None


@dataclass(frozen=True, slots=True)
class GeomContact:
    """One geom's MuJoCo contact parameters (a ``mj`` extension row).

    The row is keyed by the geom's canonical IR name in ``MjExt``.  A
    ``None`` field means the source did not declare that parameter.
    """

    name: str
    friction: tuple[float, ...] | None = None
    solref: tuple[float, ...] | None = None
    solimp: tuple[float, ...] | None = None
    margin: float | None = None
    gap: float | None = None
    priority: int | None = None
    condim: int | None = None


@dataclass(frozen=True, slots=True)
class GeomGroup:
    """One MuJoCo render group and the geoms it carries."""

    name: str
    group: int
    geoms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EqSlot:
    """One MuJoCo equality constraint slot."""

    name: str
    type: str
    obj1: str | None = None
    obj2: str | None = None
    active: bool = True


@dataclass(frozen=True, slots=True)
class ModelExt:
    """Base class for one backend namespace extension."""

    namespace: ClassVar[str] = ""

    def validate(self) -> Self:
        """Validate this extension (the base namespace has no references)."""
        return self

    def to_dict(self) -> Mapping[str, JsonValue]:
        """Return the extension's canonical dictionary form."""
        return {}


@dataclass(frozen=True, slots=True)
class MjExt(ModelExt):
    """MuJoCo-specific model facts (namespace ``mj``)."""

    namespace: ClassVar[str] = "mj"
    opt: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)
    geom_groups: Table[GeomGroup] = field(default_factory=lambda: Table[GeomGroup]())
    eq_slots: Table[EqSlot] = field(default_factory=lambda: Table[EqSlot]())
    mocap_bodies: Table[Body] = field(default_factory=lambda: Table[Body]())
    textures: Table[Texture] = field(default_factory=lambda: Table[Texture]())
    geom_contact: Table[GeomContact] = field(default_factory=lambda: Table[GeomContact]())

    @classmethod
    def from_dict(cls, data: Mapping[str, JsonValue]) -> Self:
        """Rebuild an ``MjExt`` from a canonical mapping."""
        decoded = decode(dict(data))
        if not _is_mapping(decoded):
            raise ModelIRError("mj extension: descriptor must be a mapping")
        return cls(
            opt=dict(_mapping_value(decoded.get("opt"))),
            geom_groups=_table_from_dict(
                GeomGroup, _mapping_value(decoded.get("geom_groups")), _TUPLE_FIELDS
            ),
            eq_slots=_table_from_dict(
                EqSlot, _mapping_value(decoded.get("eq_slots")), _TUPLE_FIELDS
            ),
            mocap_bodies=_table_from_dict(
                Body, _mapping_value(decoded.get("mocap_bodies")), _TUPLE_FIELDS
            ),
            textures=_table_from_dict(
                Texture, _mapping_value(decoded.get("textures")), _TUPLE_FIELDS
            ),
            geom_contact=_table_from_dict(
                GeomContact, _mapping_value(decoded.get("geom_contact")), _TUPLE_FIELDS
            ),
        )

    def to_dict(self) -> Mapping[str, JsonValue]:
        """Return the canonical ``mj`` extension dictionary."""
        return {
            "opt": dict(self.opt),
            "geom_groups": _table_payload(self.geom_groups),
            "eq_slots": _table_payload(self.eq_slots),
            "mocap_bodies": _table_payload(self.mocap_bodies),
            "textures": _table_payload(self.textures),
            "geom_contact": _table_payload(self.geom_contact),
        }


@dataclass(frozen=True, slots=True)
class UsdExt(ModelExt):
    """IsaacLab/USD-specific model facts (namespace ``usd``).

    ``prim_paths`` maps an IR entity name (body, joint, camera, ...) to its
    USD prim path, ``articulation`` names the root prim of the articulation
    (``None`` for a scene without one), and ``attrs`` carries small
    declared extras that have no base-IR concept (actuator expressions,
    per-joint velocity limits, camera resolutions).  Base-IR concepts stay
    in the base tables; this extension never overrides them.
    """

    namespace: ClassVar[str] = "usd"
    prim_paths: Mapping[str, str] = field(default_factory=_empty_str_map)
    articulation: str | None = None
    attrs: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)

    @classmethod
    def from_dict(cls, data: Mapping[str, JsonValue]) -> Self:
        """Rebuild a ``UsdExt`` from a canonical mapping."""
        decoded = decode(dict(data))
        if not _is_mapping(decoded):
            raise ModelIRError("usd extension: descriptor must be a mapping")
        prim_paths = {
            name: value
            for name, value in _mapping_value(decoded.get("prim_paths")).items()
            if isinstance(value, str)
        }
        articulation = decoded.get("articulation")
        return cls(
            prim_paths=prim_paths,
            articulation=(articulation if isinstance(articulation, str) else None),
            attrs=dict(_mapping_value(decoded.get("attrs"))),
        )

    def to_dict(self) -> Mapping[str, JsonValue]:
        """Return the canonical ``usd`` extension dictionary."""
        return {
            "prim_paths": dict(self.prim_paths),
            "articulation": self.articulation,
            "attrs": dict(self.attrs),
        }


@dataclass(frozen=True, slots=True)
class UnparsedExt:
    """An unknown extension preserved verbatim for forward compatibility."""

    namespace: str
    data: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)

    def validate(self) -> Self:
        """Unknown payloads are opaque: nothing can be validated."""
        return self

    def to_dict(self) -> Mapping[str, JsonValue]:
        """Return the preserved payload unchanged."""
        return dict(self.data)


_TUPLE_FIELDS: Mapping[object, frozenset[str]] = {
    Body: frozenset({"path", "pos", "quat", "inertia", "ipos", "iquat"}),
    Joint: frozenset({"axis", "range", "pos", "path"}),
    Geom: frozenset({"pos", "quat", "size", "path"}),
    Site: frozenset({"pos", "quat", "size", "path"}),
    Camera: frozenset({"pos", "quat", "path"}),
    Actuator: frozenset({"ctrlrange", "gear", "gainprm", "biasprm", "path"}),
    Material: frozenset({"rgba", "textures", "texrepeat"}),
    GeomGroup: frozenset({"geoms"}),
    EqSlot: frozenset(),
    Texture: frozenset(),
    GeomContact: frozenset({"friction", "solref", "solimp"}),
}

_TABLES: tuple[str, ...] = (
    "bodies",
    "joints",
    "geoms",
    "sites",
    "cameras",
    "actuators",
    "materials",
)


def _mapping_value(value: JsonValue) -> dict[str, JsonValue]:
    """Return *value* when it is a string-keyed mapping, else an empty one."""
    return dict(value) if _is_mapping(value) else {}


def _table_payload(
    table: Iterable[_NamedRow],
) -> dict[str, dict[str, JsonValue]]:
    """Return ``{name: row dict}`` for one table."""
    return {row.name: row_dict(row) for row in table}


def _table_from_dict(
    cls: Callable[..., _RowT],
    mapping: Mapping[str, JsonValue],
    tuple_fields: Mapping[object, frozenset[str]],
) -> Table[_RowT]:
    """Build a table from a name-keyed mapping, sorted by name."""
    rows = [
        from_row(cls, _mapping_value(mapping[name]), tuple_fields.get(cls, frozenset()))
        for name in sorted(mapping)
    ]
    return Table(rows)


def _keep_rows(
    table: Table[_RowT],
    selected: Iterable[str] | None,
) -> Table[_RowT]:
    """Return *table* unchanged for ``None``, else the named rows."""
    if selected is None:
        return table
    keep = set(selected)
    return Table(row for row in table if getattr(row, "name", None) in keep)


def _check_unique(kind: str, table: Iterable[_NamedRow]) -> None:
    """Reject duplicate names even if a table was tampered with."""
    seen: set[str] = set()
    for row in table:
        if row.name in seen:
            raise ModelIRError(f"model IR: duplicate {kind} name {row.name!r}")
        seen.add(row.name)


def _string_value(value: JsonValue, default: str = "") -> str:
    """Return *value* when it is a string, else *default*."""
    return value if isinstance(value, str) else default


def _optional_string(value: JsonValue) -> str | None:
    """Return *value* when it is a string, else ``None``."""
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class ModelIR:
    """The backend-neutral static model description."""

    name: str
    backend: str | None
    compiled: bool
    bodies: Table[Body] = field(default_factory=lambda: Table[Body]())
    joints: Table[Joint] = field(default_factory=lambda: Table[Joint]())
    geoms: Table[Geom] = field(default_factory=lambda: Table[Geom]())
    sites: Table[Site] = field(default_factory=lambda: Table[Site]())
    cameras: Table[Camera] = field(default_factory=lambda: Table[Camera]())
    actuators: Table[Actuator] = field(default_factory=lambda: Table[Actuator]())
    materials: Table[Material] = field(default_factory=lambda: Table[Material]())
    # ``meta`` values are opaque: ``builder`` is the backend's native MjSpec
    # (never serialized), the rest are plain scalar descriptors.
    meta: Mapping[str, object] = field(default_factory=_empty_object_map)
    ext: Mapping[str, ModelExtension] = field(default_factory=_empty_ext_map)

    def __post_init__(self) -> None:
        """Freeze the two mutable mappings into plain dictionaries."""
        object.__setattr__(self, "meta", dict(self.meta))
        object.__setattr__(self, "ext", dict(self.ext))

    @property
    def mj(self) -> MjExt | None:
        """Return the ``mj`` extension when present, else ``None``."""
        ext = self.ext.get("mj")
        return ext if isinstance(ext, MjExt) else None

    @property
    def usd(self) -> UsdExt | None:
        """Return the ``usd`` extension when present, else ``None``."""
        ext = self.ext.get("usd")
        return ext if isinstance(ext, UsdExt) else None

    def pose(
        self,
        name: str,
        *,
        camera: bool = False,
    ) -> tuple[tuple[float, ...], QuatWXYZ]:
        """Return ``(pos, quat_wxyz)`` for one body or camera.

        Args:
            name: A body name/qualname (or a camera's with ``camera=True``).
            camera: Read the camera face instead of the body face.

        Returns:
            A ``(position, quaternion)`` pair whose quaternion is a
            read-only :class:`~unigym.core.quat_types.QuatWXYZ`.

        Raises:
            KeyError: *name* is absent from the selected face.
        """
        face = "camera" if camera else "body"
        table = self.cameras if camera else self.bodies
        try:
            row = table[name]
        except KeyError:
            raise KeyError(f"model IR: {face} {name!r} is not in this model") from None
        return row.pos, row.quat_wxyz

    def to_dict(
        self,
        include_ext: bool = False,
        include_meta: bool = False,
    ) -> dict[str, Any]:
        """Return the canonical dictionary (stable keys, hex floats).

        The canonical document is a heterogeneous, wire-stable JSON tree
        whose optional sections are read dynamically by external consumers
        (``describe --json``, scene digests), so its exact value type is
        intentionally ``Any`` at this boundary.

        Args:
            include_ext: Also export the backend namespace extensions.
            include_meta: Also export ``meta`` (the opaque native builder
                is never serialized).
        """
        data: dict[str, Any] = {
            "name": self.name,
            "backend": self.backend,
            "compiled": self.compiled,
        }
        data["bodies"] = _table_payload(self.bodies)
        data["joints"] = _table_payload(self.joints)
        data["geoms"] = _table_payload(self.geoms)
        data["sites"] = _table_payload(self.sites)
        data["cameras"] = _table_payload(self.cameras)
        data["actuators"] = _table_payload(self.actuators)
        data["materials"] = _table_payload(self.materials)
        if include_meta:
            data["meta"] = {key: value for key, value in self.meta.items() if key != "builder"}
        if include_ext:
            data["ext"] = {
                namespace: extension.to_dict() for namespace, extension in self.ext.items()
            }
        encoded = encode(data)
        if not isinstance(encoded, dict):
            raise ModelIRError("model IR: canonical document is not a mapping")
        return encoded

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JsonValue],
        strict: bool = True,
    ) -> Self:
        """Rebuild a model IR from :meth:`to_dict` output.

        Args:
            data: The canonical mapping.
            strict: Reject unknown extension namespaces instead of
                preserving them as :class:`UnparsedExt`.
        """
        decoded = decode(dict(data))
        if not _is_mapping(decoded):
            raise ModelIRError("model IR: descriptor must be a mapping")
        ext: dict[str, ModelExtension] = {}
        ext_value = decoded.get("ext")
        if _is_mapping(ext_value):
            for namespace, payload in ext_value.items():
                decoded_payload = _mapping_value(payload)
                if namespace == "mj":
                    ext[namespace] = MjExt.from_dict(decoded_payload)
                elif namespace == "usd":
                    ext[namespace] = UsdExt.from_dict(decoded_payload)
                elif strict:
                    raise UnknownExtensionError(
                        f"model IR: unknown extension namespace {namespace!r} "
                        f"(pass strict=False to preserve it as UnparsedExt)"
                    )
                else:
                    ext[namespace] = UnparsedExt(namespace, dict(decoded_payload))
        return cls(
            name=_string_value(decoded.get("name", "")),
            backend=_optional_string(decoded.get("backend")),
            compiled=bool(decoded.get("compiled", False)),
            bodies=_table_from_dict(Body, _mapping_value(decoded.get("bodies")), _TUPLE_FIELDS),
            joints=_table_from_dict(Joint, _mapping_value(decoded.get("joints")), _TUPLE_FIELDS),
            geoms=_table_from_dict(Geom, _mapping_value(decoded.get("geoms")), _TUPLE_FIELDS),
            sites=_table_from_dict(Site, _mapping_value(decoded.get("sites")), _TUPLE_FIELDS),
            cameras=_table_from_dict(Camera, _mapping_value(decoded.get("cameras")), _TUPLE_FIELDS),
            actuators=_table_from_dict(
                Actuator, _mapping_value(decoded.get("actuators")), _TUPLE_FIELDS
            ),
            materials=_table_from_dict(
                Material, _mapping_value(decoded.get("materials")), _TUPLE_FIELDS
            ),
            meta=dict(_mapping_value(decoded.get("meta"))),
            ext=ext,
        )

    def validate(self) -> Self:
        """Check names and cross-table references; return ``self``.

        Raises:
            ModelIRError: A duplicate name, dangling body reference or
                reversed joint range.
        """
        _check_unique("bodies", self.bodies)
        _check_unique("joints", self.joints)
        _check_unique("geoms", self.geoms)
        _check_unique("sites", self.sites)
        _check_unique("cameras", self.cameras)
        _check_unique("actuators", self.actuators)
        _check_unique("materials", self.materials)
        bodies = set(self.bodies.names())
        for body in self.bodies:
            if body.parent is not None and body.parent not in bodies:
                raise ModelIRError(
                    f"model IR: body {body.name!r} names missing parent " f"{body.parent!r}"
                )
        for joint in self.joints:
            if joint.body not in bodies:
                raise ModelIRError(
                    f"model IR: joint {joint.name!r} references missing " f"body {joint.body!r}"
                )
            if joint.range is not None and joint.range[0] > joint.range[1]:
                raise ModelIRError(
                    f"model IR: joint {joint.name!r} has reversed range "
                    f"{joint.range!r} (lo must be <= hi)"
                )
        for camera in self.cameras:
            if camera.body not in bodies:
                raise ModelIRError(
                    f"model IR: camera {camera.name!r} references missing " f"body {camera.body!r}"
                )
        self._validate_actuators(bodies)
        for extension in self.ext.values():
            extension.validate()
        return self

    def _validate_actuators(self, bodies: set[str]) -> None:
        """Check that every actuator target names a declared entity."""
        joints = set(self.joints.names())
        sites = set(self.sites.names())
        known = {"joint": joints, "jointinparent": joints, "site": sites, "body": bodies}
        for actuator in self.actuators:
            targets = known.get(actuator.trntype)
            if targets is not None and actuator.target not in targets:
                raise ModelIRError(
                    f"model IR: actuator {actuator.name!r} targets missing "
                    f"{actuator.trntype} {actuator.target!r}"
                )

    def subset(
        self,
        *,
        bodies: Iterable[str] | None = None,
        joints: Iterable[str] | None = None,
        geoms: Iterable[str] | None = None,
        sites: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
        actuators: Iterable[str] | None = None,
        materials: Iterable[str] | None = None,
        name: str | None = None,
    ) -> ModelIR:
        """Return a filtered view of this IR.

        Every argument is an optional iterable of row names; ``None`` keeps
        that whole table.  Rows are kept by identity (their compiled data is
        never rewritten), in this model's order, so a per-articulation view
        preserves the compiled addresses/ids.  ``meta`` and ``ext`` are
        carried over unchanged.  The result is NOT re-validated: a subset
        may intentionally drop a parent body (it is a view, not a new
        scene).

        Args:
            bodies: Optional body-name iterable (``None`` keeps all bodies).
            joints: Optional joint-name iterable (``None`` keeps all joints).
            geoms: Optional geom-name iterable (``None`` keeps all geoms).
            sites: Optional site-name iterable (``None`` keeps all sites).
            cameras: Optional camera-name iterable (``None`` keeps all
                cameras).
            actuators: Optional actuator-name iterable (``None`` keeps all
                actuators).
            materials: Optional material-name iterable (``None`` keeps all
                materials).
            name: Optional IR name override (default: this model's name).
        """
        return ModelIR(
            name=self.name if name is None else name,
            backend=self.backend,
            compiled=self.compiled,
            bodies=_keep_rows(self.bodies, bodies),
            joints=_keep_rows(self.joints, joints),
            geoms=_keep_rows(self.geoms, geoms),
            sites=_keep_rows(self.sites, sites),
            cameras=_keep_rows(self.cameras, cameras),
            actuators=_keep_rows(self.actuators, actuators),
            materials=_keep_rows(self.materials, materials),
            meta=dict(self.meta),
            ext=dict(self.ext),
        )

    def subtree(self, root: str) -> Table[Body]:
        """Return the bodies under *root* (the root itself included).

        Args:
            root: A body name (or qualname) present in :attr:`bodies`.

        Returns:
            A :class:`~unigym.model.table.Table` of the root body and all
            bodies whose ``path`` extends the root's path.

        Raises:
            ModelIRError: *root* names no body in this model.
        """
        if root not in self.bodies:
            raise ModelIRError(f"model IR: subtree root {root!r} is not a body")
        root_path = tuple(self.bodies[root].path)
        rows: list[Body] = []
        for body in self.bodies:
            path = tuple(body.path)
            if body.name == root:
                rows.append(body)
            elif root_path and path[: len(root_path)] == root_path:
                rows.append(body)
        return Table(rows)

    def supports(self, *traits: Trait) -> bool:
        """Return True when *self.backend* declares every requested trait."""
        if self.backend is None:
            return False
        from unigym.backends.registry import backend_info

        return backend_info(self.backend).has(*traits)

    def explain_missing(self, *traits: Trait) -> str:
        """Return a one-line explanation of missing traits (empty if none)."""
        if self.backend is None:
            return "model IR has no backend: capabilities cannot be resolved"
        from unigym.backends.registry import backend_info

        missing = backend_info(self.backend).missing(*traits)
        if not missing:
            return ""
        names = ", ".join(trait.name for trait in missing)
        return f"backend {self.backend!r} lacks {names}"

    def diff(
        self,
        other: ModelIR,
        *,
        include_ext: bool = False,
        ignore_compiled: bool = False,
    ) -> ModelDiff:
        """Return the face-by-face difference to *other*.

        Comparing a declared IR with a compiled one is allowed but noisy:
        every compiled-only field shows as a ``None`` -> value change.
        Pass ``ignore_compiled=True`` to skip ``*_id``, ``*_adr``,
        ``mass``, ``inertia``, ``ipos`` and ``iquat``.

        Args:
            other: The ``ModelIR`` to compare against.
            include_ext: Also compare ``ext.<namespace>`` entity maps.
            ignore_compiled: Skip compiled-only fields.

        Returns:
            A falsy :class:`ModelDiff` when the models are equal.
        """
        from .diff import diff_models

        return diff_models(self, other, include_ext=include_ext, ignore_compiled=ignore_compiled)

    def _repr_html_(self) -> str:
        """Return an HTML summary with one small table per non-empty face."""
        from html import escape

        counts = ", ".join(f"{name}={len(getattr(self, name))}" for name in _TABLES)
        links = " | ".join(
            f"<a href='#{name}'>{name}</a> ({len(getattr(self, name))})"
            for name in _TABLES
            if len(getattr(self, name))
        )
        parts = [
            "<div class='unigym-model-ir'>",
            f"<h3>ModelIR <code>{escape(self.name)}</code></h3>",
            f"<p>backend={escape(str(self.backend))}, " f"compiled={self.compiled}</p>",
            f"<p>{escape(counts)}</p>",
            f"<p>{links}</p>",
        ]
        for name in _TABLES:
            table = getattr(self, name)
            if not len(table):
                continue
            parts.append(f"<h4 id='{name}'>{name} ({len(table)})</h4>")
            parts.append(table._repr_html_())
        parts.append("</div>")
        return "\n".join(parts)

    def _repr_pretty_(self, p: _PrettyPrinter, cycle: bool) -> None:
        """Render through the IPython pretty protocol (no IPython import)."""
        if cycle:
            p.text("ModelIR(...)")
            return
        p.text(repr(self))

    def __repr__(self) -> str:
        """Return a compact count summary of the model tables."""
        return (
            f"ModelIR(name={self.name!r}, backend={self.backend!r}, "
            f"compiled={self.compiled}, bodies={len(self.bodies)}, "
            f"joints={len(self.joints)}, geoms={len(self.geoms)}, "
            f"cameras={len(self.cameras)}, actuators={len(self.actuators)})"
        )
