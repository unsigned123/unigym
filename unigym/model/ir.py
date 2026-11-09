"""The neutral model IR: one static description for every backend.

The dataclasses here are pure data (``frozen=True, slots=True``): declared
fields are populated before compilation, compiled fields stay ``None``
until a backend's ``describe`` fills them.  A ``ModelIR`` never invents a
value -- a ``None`` means "this backend has no such concept" or "not
compiled yet", exactly as design v098 section 9.4 requires.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ._serialize import decode, encode, from_row, row_dict
from .errors import (
    ModelIRError,
    StaleModelError,
    UnknownExtensionError,
)
from .table import Table

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
    "ModelIR",
    "ModelIRError",
    "Site",
    "StaleModelError",
    "Texture",
    "UnknownExtensionError",
    "UnparsedExt",
    "UsdExt",
]


def _quat_wxyz(values):
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
    def quat_wxyz(self):
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
    def quat_wxyz(self):
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
    def quat_wxyz(self):
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

    def validate(self):
        """Validate this extension (the base namespace has no references)."""
        return self

    def to_dict(self):
        """Return the extension's canonical dictionary form."""
        return {}


@dataclass(frozen=True, slots=True)
class MjExt(ModelExt):
    """MuJoCo-specific model facts (namespace ``mj``)."""

    namespace: ClassVar[str] = "mj"
    opt: Mapping = field(default_factory=dict)
    geom_groups: Table = field(default_factory=Table)
    eq_slots: Table = field(default_factory=Table)
    mocap_bodies: Table = field(default_factory=Table)
    textures: Table = field(default_factory=Table)
    geom_contact: Table = field(default_factory=Table)

    @classmethod
    def from_dict(cls, data):
        """Rebuild an ``MjExt`` from a canonical mapping."""
        decoded = decode(dict(data))
        return cls(
            opt=dict(decoded.get("opt") or {}),
            geom_groups=_table_from_dict(
                GeomGroup, decoded.get("geom_groups") or {}, _TUPLE_FIELDS),
            eq_slots=_table_from_dict(
                EqSlot, decoded.get("eq_slots") or {}, _TUPLE_FIELDS),
            mocap_bodies=_table_from_dict(
                Body, decoded.get("mocap_bodies") or {}, _TUPLE_FIELDS),
            textures=_table_from_dict(
                Texture, decoded.get("textures") or {}, _TUPLE_FIELDS),
            geom_contact=_table_from_dict(
                GeomContact, decoded.get("geom_contact") or {},
                _TUPLE_FIELDS),
        )

    def to_dict(self):
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
    prim_paths: Mapping = field(default_factory=dict)
    articulation: str | None = None
    attrs: Mapping = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        """Rebuild a ``UsdExt`` from a canonical mapping."""
        decoded = decode(dict(data))
        return cls(
            prim_paths=dict(decoded.get("prim_paths") or {}),
            articulation=decoded.get("articulation"),
            attrs=dict(decoded.get("attrs") or {}),
        )

    def to_dict(self):
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
    data: Mapping = field(default_factory=dict)

    def validate(self):
        """Unknown payloads are opaque: nothing can be validated."""
        return self

    def to_dict(self):
        """Return the preserved payload unchanged."""
        return dict(self.data)


_TUPLE_FIELDS = {
    Body: frozenset({"path", "pos", "quat", "inertia", "ipos", "iquat"}),
    Joint: frozenset({"axis", "range", "pos", "path"}),
    Geom: frozenset({"pos", "quat", "size", "path"}),
    Site: frozenset({"pos", "quat", "size", "path"}),
    Camera: frozenset({"pos", "quat", "path"}),
    Actuator: frozenset(
        {"ctrlrange", "gear", "gainprm", "biasprm", "path"}),
    Material: frozenset({"rgba", "textures", "texrepeat"}),
    GeomGroup: frozenset({"geoms"}),
    EqSlot: frozenset(),
    Texture: frozenset(),
    GeomContact: frozenset({"friction", "solref", "solimp"}),
}

_TABLES = (
    ("bodies", Body),
    ("joints", Joint),
    ("geoms", Geom),
    ("sites", Site),
    ("cameras", Camera),
    ("actuators", Actuator),
    ("materials", Material),
)


def _table_payload(table):
    """Return ``{name: row dict}`` for one table."""
    return {row.name: row_dict(row) for row in table}


def _table_from_dict(cls, mapping, tuple_fields):
    """Build a table from a name-keyed mapping, sorted by name."""
    rows = [
        from_row(cls, mapping[name], tuple_fields.get(cls, ()))
        for name in sorted(mapping)
    ]
    return Table(rows)


def _check_unique(kind, table):
    """Reject duplicate names even if a table was tampered with."""
    seen = set()
    for row in table:
        if row.name in seen:
            raise ModelIRError(
                f"model IR: duplicate {kind} name {row.name!r}")
        seen.add(row.name)


@dataclass(frozen=True, slots=True)
class ModelIR:
    """The backend-neutral static model description."""

    name: str
    backend: str | None
    compiled: bool
    bodies: Table = field(default_factory=Table)
    joints: Table[Joint] = field(default_factory=lambda: Table[Joint]())
    geoms: Table = field(default_factory=Table)
    sites: Table = field(default_factory=Table)
    cameras: Table = field(default_factory=Table)
    actuators: Table[Actuator] = field(default_factory=lambda: Table[Actuator]())
    materials: Table = field(default_factory=Table)
    meta: Mapping = field(default_factory=dict)
    ext: Mapping = field(default_factory=dict)

    def __post_init__(self):
        """Freeze the two mutable mappings into plain dictionaries."""
        object.__setattr__(self, "meta", dict(self.meta))
        object.__setattr__(self, "ext", dict(self.ext))

    @property
    def mj(self):
        """Return the ``mj`` extension when present, else ``None``."""
        ext = self.ext.get("mj")
        return ext if isinstance(ext, MjExt) else None

    @property
    def usd(self):
        """Return the ``usd`` extension when present, else ``None``."""
        ext = self.ext.get("usd")
        return ext if isinstance(ext, UsdExt) else None

    def pose(self, name, *, camera=False):
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
            raise KeyError(
                f"model IR: {face} {name!r} is not in this model") from None
        return row.pos, row.quat_wxyz

    def to_dict(self, include_ext=False, include_meta=False):
        """Return the canonical dictionary (stable keys, hex floats).

        Args:
            include_ext: Also export the backend namespace extensions.
            include_meta: Also export ``meta`` (the opaque native builder
                is never serialized).
        """
        data = {
            "name": self.name,
            "backend": self.backend,
            "compiled": self.compiled,
        }
        for key, _ in _TABLES:
            data[key] = _table_payload(getattr(self, key))
        if include_meta:
            data["meta"] = {
                key: value for key, value in self.meta.items()
                if key != "builder"
            }
        if include_ext:
            data["ext"] = {
                namespace: extension.to_dict()
                for namespace, extension in self.ext.items()
            }
        return encode(data)

    @classmethod
    def from_dict(cls, data, strict=True):
        """Rebuild a model IR from :meth:`to_dict` output.

        Args:
            data: The canonical mapping.
            strict: Reject unknown extension namespaces instead of
                preserving them as :class:`UnparsedExt`.
        """
        decoded = decode(dict(data))
        kwargs = {key: _table_from_dict(
            entity, decoded.get(key) or {}, _TUPLE_FIELDS)
            for key, entity in _TABLES}
        meta = dict(decoded.get("meta") or {})
        ext = {}
        for namespace, payload in (decoded.get("ext") or {}).items():
            if namespace == "mj":
                ext[namespace] = MjExt.from_dict(payload)
            elif namespace == "usd":
                ext[namespace] = UsdExt.from_dict(payload)
            elif strict:
                raise UnknownExtensionError(
                    f"model IR: unknown extension namespace {namespace!r} "
                    f"(pass strict=False to preserve it as UnparsedExt)")
            else:
                ext[namespace] = UnparsedExt(namespace, dict(payload))
        return cls(
            name=decoded.get("name", ""),
            backend=decoded.get("backend"),
            compiled=bool(decoded.get("compiled", False)),
            meta=meta,
            ext=ext,
            **kwargs,
        )

    def validate(self):
        """Check names and cross-table references; return ``self``.

        Raises:
            ModelIRError: A duplicate name, dangling body reference or
                reversed joint range.
        """
        for kind, _ in _TABLES:
            _check_unique(kind, getattr(self, kind))
        bodies = set(self.bodies.names())
        for body in self.bodies:
            if body.parent is not None and body.parent not in bodies:
                raise ModelIRError(
                    f"model IR: body {body.name!r} names missing parent "
                    f"{body.parent!r}")
        for joint in self.joints:
            if joint.body not in bodies:
                raise ModelIRError(
                    f"model IR: joint {joint.name!r} references missing "
                    f"body {joint.body!r}")
            if joint.range is not None and joint.range[0] > joint.range[1]:
                raise ModelIRError(
                    f"model IR: joint {joint.name!r} has reversed range "
                    f"{joint.range!r} (lo must be <= hi)")
        for camera in self.cameras:
            if camera.body not in bodies:
                raise ModelIRError(
                    f"model IR: camera {camera.name!r} references missing "
                    f"body {camera.body!r}")
        self._validate_actuators(bodies)
        for extension in self.ext.values():
            extension.validate()
        return self

    def _validate_actuators(self, bodies):
        """Check that every actuator target names a declared entity."""
        joints = set(self.joints.names())
        sites = set(self.sites.names())
        known = {"joint": joints, "jointinparent": joints,
                 "site": sites, "body": bodies}
        for actuator in self.actuators:
            targets = known.get(actuator.trntype)
            if targets is not None and actuator.target not in targets:
                raise ModelIRError(
                    f"model IR: actuator {actuator.name!r} targets missing "
                    f"{actuator.trntype} {actuator.target!r}")

    def subset(self, *, bodies=None, joints=None, geoms=None, sites=None,
               cameras=None, actuators=None, materials=None, name=None):
        """Return a filtered view of this IR (v103.4).

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
        selections = {
            "bodies": bodies, "joints": joints, "geoms": geoms,
            "sites": sites, "cameras": cameras, "actuators": actuators,
            "materials": materials,
        }
        tables = {}
        for key, selected in selections.items():
            table = getattr(self, key)
            if selected is None:
                tables[key] = table
                continue
            keep = set(selected)
            tables[key] = Table(
                row for row in table if row.name in keep)
        return ModelIR(
            name=self.name if name is None else name,
            backend=self.backend,
            compiled=self.compiled,
            meta=dict(self.meta),
            ext=dict(self.ext),
            **tables,
        )

    def subtree(self, root):
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
            raise ModelIRError(
                f"model IR: subtree root {root!r} is not a body")
        root_path = tuple(self.bodies[root].path)
        rows = []
        for body in self.bodies:
            path = tuple(body.path)
            if body.name == root:
                rows.append(body)
            elif root_path and path[:len(root_path)] == root_path:
                rows.append(body)
        return Table(rows)

    def supports(self, *traits):
        """Return True when *self.backend* declares every requested trait."""
        if self.backend is None:
            return False
        from unigym.backends.registry import backend_info

        return backend_info(self.backend).has(*traits)

    def explain_missing(self, *traits):
        """Return a one-line explanation of missing traits (empty if none)."""
        if self.backend is None:
            return "model IR has no backend: capabilities cannot be resolved"
        from unigym.backends.registry import backend_info

        missing = backend_info(self.backend).missing(*traits)
        if not missing:
            return ""
        names = ", ".join(trait.name for trait in missing)
        return f"backend {self.backend!r} lacks {names}"

    def diff(self, other, *, include_ext=False, ignore_compiled=False):
        """Return the face-by-face difference to *other* (design v098 N5).

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

        return diff_models(
            self, other, include_ext=include_ext,
            ignore_compiled=ignore_compiled)

    def _repr_html_(self):
        """Return an HTML summary with one small table per non-empty face."""
        from html import escape

        counts = ", ".join(
            f"{name}={len(getattr(self, name))}" for name, _ in _TABLES)
        links = " | ".join(
            f"<a href='#{name}'>{name}</a> ({len(getattr(self, name))})"
            for name, _ in _TABLES if len(getattr(self, name)))
        parts = [
            "<div class='unigym-model-ir'>",
            f"<h3>ModelIR <code>{escape(self.name)}</code></h3>",
            f"<p>backend={escape(str(self.backend))}, "
            f"compiled={self.compiled}</p>",
            f"<p>{escape(counts)}</p>",
            f"<p>{links}</p>",
        ]
        for name, _ in _TABLES:
            table = getattr(self, name)
            if not len(table):
                continue
            parts.append(f"<h4 id='{name}'>{name} ({len(table)})</h4>")
            parts.append(table._repr_html_())
        parts.append("</div>")
        return "\n".join(parts)

    def _repr_pretty_(self, p, cycle):
        """Render through the IPython pretty protocol (no IPython import)."""
        if cycle:
            p.text("ModelIR(...)")
            return
        p.text(repr(self))

    def __repr__(self):
        """Return a compact count summary of the model tables."""
        return (
            f"ModelIR(name={self.name!r}, backend={self.backend!r}, "
            f"compiled={self.compiled}, bodies={len(self.bodies)}, "
            f"joints={len(self.joints)}, geoms={len(self.geoms)}, "
            f"cameras={len(self.cameras)}, actuators={len(self.actuators)})"
        )
