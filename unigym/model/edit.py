"""Engine-neutral model-edit batch transaction.

A :class:`ModelEditor` resolves body/geom names against the *published*
``ModelSet`` immediately, accumulates validated field writes, and applies
them exactly once through the backend's commit callable.  The transaction is
global: every edit applies to every env.  There is no ``env_ids`` parameter.

The module imports no engine (no mujoco, no warp, no isaaclab); backends own
the actual native writes behind the commit callback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, cast

from unigym.backends.registry import Trait, ModelEditField
from unigym.model.errors import ModelEditCapabilityError, ModelEditError

if TYPE_CHECKING:
    from unigym.model.ir import Body, ModelIR
    from unigym.model.set import ModelSet

__all__ = [
    "BodyEdit",
    "BodyEditHandle",
    "GeomEdit",
    "GeomEditHandle",
    "ModelEditField",
    "ModelEditResult",
    "ModelEditor",
]

ModelEditValue = float | tuple[float, float, float] | tuple[float, float, float, float]

_BODY_FIELDS: tuple[ModelEditField, ...] = (
    ModelEditField.BODY_MASS,
    ModelEditField.BODY_COM,
    ModelEditField.BODY_INERTIA,
    ModelEditField.BODY_INERTIA_QUAT,
)
_GEOM_FIELDS: tuple[ModelEditField, ...] = (
    ModelEditField.GEOM_FRICTION,
    ModelEditField.GEOM_RESTITUTION,
)

_MAX_RESTITUTION = 0.999


def _empty_values() -> dict[ModelEditField, ModelEditValue]:
    """Typed empty field map for dataclass defaults."""
    return {}


@dataclass(frozen=True)
class BodyEdit:
    """One resolved body edit (target identity plus validated values)."""

    articulation: str | None
    body: str
    body_id: int | None
    values: Mapping[ModelEditField, ModelEditValue] = field(default_factory=_empty_values)

    def __post_init__(self) -> None:
        """Freeze the value mapping into a plain dictionary."""
        frozen: dict[ModelEditField, ModelEditValue] = dict(self.values)
        object.__setattr__(self, "values", frozen)


@dataclass(frozen=True)
class GeomEdit:
    """One resolved geom edit (target identity plus validated values)."""

    articulation: str | None
    body: str
    geom: str
    geom_id: int | None
    values: Mapping[ModelEditField, ModelEditValue] = field(default_factory=_empty_values)

    def __post_init__(self) -> None:
        """Freeze the value mapping into a plain dictionary."""
        frozen: dict[ModelEditField, ModelEditValue] = dict(self.values)
        object.__setattr__(self, "values", frozen)


@dataclass(frozen=True)
class ModelEditResult:
    """The applied result of one committed transaction.

    Attributes:
        backend: The backend that applied the edit.
        revision: The backend model revision after the commit.
        bodies: The resolved body edits (empty for a no-op transaction).
        geoms: The resolved geom edits.
        warnings: Approximations or backend notes (e.g. restitution).
    """

    backend: str
    revision: int
    bodies: tuple[BodyEdit, ...] = ()
    geoms: tuple[GeomEdit, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """True when the transaction touched no target."""
        return not self.bodies and not self.geoms


@dataclass
class _PendingBody:
    articulation: str | None
    body: str
    body_id: int | None
    values: dict[ModelEditField, ModelEditValue] = field(default_factory=_empty_values)


@dataclass
class _PendingGeom:
    articulation: str | None
    body: str
    geom: str
    geom_id: int | None
    values: dict[ModelEditField, ModelEditValue] = field(default_factory=_empty_values)


def _field_name(field: ModelEditField) -> str:
    """Return the public dotted name of *field*."""
    return field.value


def _supported(fields: Iterable[ModelEditField]) -> str:
    """Return a stable, human-readable field list."""
    return ", ".join(sorted(_field_name(field) for field in fields)) or "<none>"


def _as_scalar(value: object, field: ModelEditField) -> float:
    """Validate one finite scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelEditError(
            f"model edit: {_field_name(field)} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ModelEditError(f"model edit: {_field_name(field)} must be finite, got {value!r}")
    return number


def _as_vector(value: object, field: ModelEditField, size: int) -> tuple[float, ...]:
    """Validate one finite numeric vector of exactly *size* elements."""
    if isinstance(value, (str, bytes)) or value is None:
        raise ModelEditError(
            f"model edit: {_field_name(field)} must be a {size}-tuple, "
            f"got {type(value).__name__}"
        )
    try:
        items: list[object] = list(value)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ModelEditError(
            f"model edit: {_field_name(field)} must be a {size}-tuple, "
            f"got {type(value).__name__}"
        ) from exc
    if len(items) != size:
        raise ModelEditError(
            f"model edit: {_field_name(field)} must have {size} elements, got {len(items)}"
        )
    out: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ModelEditError(
                f"model edit: {_field_name(field)} elements must be numbers, "
                f"got {type(item).__name__}"
            )
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise ModelEditError(
                f"model edit: {_field_name(field)} elements must be finite, got {item!r}"
            )
        out.append(number)
    return tuple(out)


def _normalized_quat(value: object) -> tuple[float, float, float, float]:
    """Validate and normalize a wxyz quaternion."""
    quat = _as_vector(value, ModelEditField.BODY_INERTIA_QUAT, 4)
    norm = sum(component * component for component in quat) ** 0.5
    if norm <= 1e-12:
        raise ModelEditError("model edit: body.inertia_quat must not be a zero-norm quaternion")
    return (quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm)


def _validate_body_fields(
    values: Mapping[ModelEditField, object],
) -> dict[ModelEditField, ModelEditValue]:
    """Validate the supplied body fields and return normalized values."""
    out: dict[ModelEditField, ModelEditValue] = {}
    for edit_field, raw in values.items():
        if raw is None:
            continue
        if edit_field is ModelEditField.BODY_MASS:
            mass = _as_scalar(raw, edit_field)
            if mass <= 0.0:
                raise ModelEditError(f"model edit: body.mass must be > 0, got {mass}")
            out[edit_field] = mass
        elif edit_field is ModelEditField.BODY_COM:
            out[edit_field] = cast(tuple[float, float, float], _as_vector(raw, edit_field, 3))
        elif edit_field is ModelEditField.BODY_INERTIA:
            inertia = _as_vector(raw, edit_field, 3)
            if any(component < 0.0 for component in inertia):
                raise ModelEditError(f"model edit: body.inertia must be >= 0, got {inertia}")
            out[edit_field] = cast(tuple[float, float, float], inertia)
        elif edit_field is ModelEditField.BODY_INERTIA_QUAT:
            out[edit_field] = _normalized_quat(raw)
        else:  # pragma: no cover - the caller filters body fields
            raise ModelEditError(f"model edit: {_field_name(edit_field)} is not a body field")
    return out


def _validate_geom_fields(
    values: Mapping[ModelEditField, object],
) -> dict[ModelEditField, ModelEditValue]:
    """Validate the supplied geom fields and return normalized values."""
    out: dict[ModelEditField, ModelEditValue] = {}
    for edit_field, raw in values.items():
        if raw is None:
            continue
        if edit_field is ModelEditField.GEOM_FRICTION:
            friction = _as_vector(raw, edit_field, 3)
            if any(component < 0.0 for component in friction):
                raise ModelEditError(f"model edit: geom.friction must be >= 0, got {friction}")
            out[edit_field] = cast(tuple[float, float, float], friction)
        elif edit_field is ModelEditField.GEOM_RESTITUTION:
            restitution = _as_scalar(raw, edit_field)
            if not 0.0 <= restitution <= _MAX_RESTITUTION:
                raise ModelEditError(
                    f"model edit: geom.restitution must be in [0, "
                    f"{_MAX_RESTITUTION}], got {restitution}"
                )
            out[edit_field] = restitution
        else:  # pragma: no cover - the caller filters geom fields
            raise ModelEditError(f"model edit: {_field_name(edit_field)} is not a geom field")
    return out


def _candidates(names: Iterable[str], limit: int = 12) -> str:
    """Return a bounded, stable candidate list."""
    ordered = sorted(names)
    shown = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f", ... ({len(ordered)} total)"
    return ", ".join(shown) + suffix


class BodyEditHandle:
    """A body target inside one transaction; ``.set`` accumulates fields."""

    __slots__ = ("_editor", "_pending")

    def __init__(self, editor: ModelEditor, pending: _PendingBody) -> None:
        self._editor = editor
        self._pending = pending

    def set(
        self,
        *,
        mass: float | None = None,
        com: Iterable[float] | None = None,
        inertia: Iterable[float] | None = None,
        inertia_quat: Iterable[float] | None = None,
    ) -> BodyEditHandle:
        """Stage body fields for this target (chainable).

        Args:
            mass: Body mass in kg, ``> 0``.
            com: Body-local centre of mass in metres.
            inertia: Principal diagonal inertia in ``kg m^2``.
            inertia_quat: Optional wxyz inertia frame rotation (normalized).

        Returns:
            This handle.

        Raises:
            ModelEditError: No field was supplied, a field is duplicated, a
                value is invalid, or the transaction is closed.
            ModelEditCapabilityError: The backend does not write the field.
        """
        self._editor._check_open()
        requested: dict[ModelEditField, object | None] = {
            ModelEditField.BODY_MASS: mass,
            ModelEditField.BODY_COM: com,
            ModelEditField.BODY_INERTIA: inertia,
            ModelEditField.BODY_INERTIA_QUAT: inertia_quat,
        }
        supplied: dict[ModelEditField, object] = {
            field: value for field, value in requested.items() if value is not None
        }
        if not supplied:
            raise ModelEditError(
                "model edit: BodyEditHandle.set() needs at least one field "
                "(mass, com, inertia, inertia_quat)"
            )
        self._editor._require_fields(supplied)
        duplicate = sorted(
            _field_name(field) for field in supplied if field in self._pending.values
        )
        if duplicate:
            raise ModelEditError(f"model edit: duplicate field(s) in one transaction: {duplicate}")
        self._pending.values.update(_validate_body_fields(supplied))
        return self


class GeomEditHandle:
    """A geom target inside one transaction; ``.set`` accumulates fields."""

    __slots__ = ("_editor", "_pending")

    def __init__(self, editor: ModelEditor, pending: _PendingGeom) -> None:
        self._editor = editor
        self._pending = pending

    def set(
        self,
        *,
        friction: Iterable[float] | None = None,
        restitution: float | None = None,
    ) -> GeomEditHandle:
        """Stage geom fields for this target (chainable).

        Args:
            friction: ``(slide, torsional, rolling)``, each finite and ``>= 0``.
            restitution: Normal restitution in ``[0, 0.999]``.

        Returns:
            This handle.

        Raises:
            ModelEditError: No field was supplied, a field is duplicated, a
                value is invalid, or the transaction is closed.
            ModelEditCapabilityError: The backend does not write the field.
        """
        self._editor._check_open()
        requested: dict[ModelEditField, object | None] = {
            ModelEditField.GEOM_FRICTION: friction,
            ModelEditField.GEOM_RESTITUTION: restitution,
        }
        supplied: dict[ModelEditField, object] = {
            field: value for field, value in requested.items() if value is not None
        }
        if not supplied:
            raise ModelEditError(
                "model edit: GeomEditHandle.set() needs at least one field "
                "(friction, restitution)"
            )
        self._editor._require_fields(supplied)
        duplicate = sorted(
            _field_name(field) for field in supplied if field in self._pending.values
        )
        if duplicate:
            raise ModelEditError(f"model edit: duplicate field(s) in one transaction: {duplicate}")
        self._pending.values.update(_validate_geom_fields(supplied))
        return self


class ModelEditor:
    """One global model-edit transaction (context-managed, single use).

    Args:
        backend: The live backend name.
        fields: The backend's writable field set.
        model_set: The published model snapshot names resolve against.
        revision: The model revision when the editor opened.
        commit: Backend callback ``(bodies, geoms) -> ModelEditResult``.
        is_open: Optional liveness probe (closed env / step guard).
        current_revision: Optional live revision probe for stale detection.
    """

    __slots__ = (
        "_backend",
        "_fields",
        "_model_set",
        "_revision",
        "_open_revision",
        "_commit",
        "_is_open",
        "_current_revision",
        "_bodies",
        "_geoms",
        "_closed",
        "_result",
    )

    def __init__(
        self,
        *,
        backend: str,
        fields: frozenset[ModelEditField],
        model_set: ModelSet,
        revision: int,
        commit: Callable[[tuple[BodyEdit, ...], tuple[GeomEdit, ...]], ModelEditResult],
        is_open: Callable[[], bool] | None = None,
        current_revision: Callable[[], int] | None = None,
    ) -> None:
        self._backend = backend
        self._fields = frozenset(fields)
        self._model_set = model_set
        self._revision = int(revision)
        self._open_revision = int(revision)
        self._commit = commit
        self._is_open = is_open
        self._current_revision = current_revision
        self._bodies: dict[tuple[str | None, str], _PendingBody] = {}
        self._geoms: dict[tuple[str | None, str], _PendingGeom] = {}
        self._closed = False
        self._result: ModelEditResult | None = None

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        """The live backend name."""
        return self._backend

    @property
    def fields(self) -> frozenset[ModelEditField]:
        """The backend's writable model fields."""
        return self._fields

    @property
    def revision(self) -> int:
        """The model revision (updated to the committed revision)."""
        return self._revision

    @property
    def result(self) -> ModelEditResult | None:
        """The committed result, or ``None`` before a successful commit."""
        return self._result

    @property
    def closed(self) -> bool:
        """True once the transaction committed, rolled back or failed."""
        return self._closed

    # ------------------------------------------------------------------ #
    # target handles
    # ------------------------------------------------------------------ #
    def body(self, articulation: str | None, body: str) -> BodyEditHandle:
        """Resolve a body target against the published model.

        Args:
            articulation: The owning articulation, ``None`` for a single
                articulation run (or the merged scene when it exists).
            body: A body name or qualname from the selected IR.

        Returns:
            A reusable :class:`BodyEditHandle` for this target.

        Raises:
            ModelEditError: The articulation/body is unknown, the body is the
                world, or the transaction is closed.
        """
        self._check_open()
        name, ir = self._select(articulation)
        row = self._resolve_body(ir, body)
        if row.name == "world" or row.body_id == 0:
            raise ModelEditError(
                "model edit: body parameter edits on the world body are "
                "rejected (body_id=0 carries the static world)"
            )
        key = (name, row.name)
        pending = self._bodies.get(key)
        if pending is None:
            pending = _PendingBody(name, row.name, row.body_id)
            self._bodies[key] = pending
        return BodyEditHandle(self, pending)

    def geom(
        self,
        articulation: str | None,
        *,
        body: str,
        geom: str,
    ) -> GeomEditHandle:
        """Resolve a geom target and require it to belong to *body*.

        Args:
            articulation: The owning articulation, ``None`` for a single
                articulation run (or the merged scene when it exists).
            body: The body the geom must be attached to.
            geom: A geom name or qualname from the selected IR.

        Returns:
            A reusable :class:`GeomEditHandle` for this target.

        Raises:
            ModelEditError: The body/geom is unknown, the geom belongs to a
                different body, or the transaction is closed.
        """
        self._check_open()
        name, ir = self._select(articulation)
        body_row = self._resolve_body(ir, body)
        try:
            row = ir.geoms[geom]
        except KeyError:
            raise ModelEditError(
                f"model edit: geom {geom!r} is not in "
                f"{self._scope_label(name)}; candidates: "
                f"{_candidates(ir.geoms.names())}"
            ) from None
        if row.body != body_row.name:
            raise ModelEditError(
                f"model edit: geom {row.name!r} belongs to body "
                f"{row.body!r}, not the named body {body_row.name!r}"
            )
        key = (name, row.name)
        pending = self._geoms.get(key)
        if pending is None:
            pending = _PendingGeom(name, body_row.name, row.name, row.geom_id)
            self._geoms[key] = pending
        return GeomEditHandle(self, pending)

    # ------------------------------------------------------------------ #
    # transaction lifecycle
    # ------------------------------------------------------------------ #
    def commit(self) -> ModelEditResult:
        """Apply every staged edit once and close the transaction.

        Returns:
            The :class:`ModelEditResult` (empty and revision-preserving for a
            no-op transaction).

        Raises:
            ModelEditError: The transaction is closed/stale, or the backend
                rejected the edit.
        """
        self._check_open()
        self._closed = True
        if self._current_revision is not None:
            current = int(self._current_revision())
            if current != self._open_revision:
                raise ModelEditError(
                    f"model edit: stale transaction (opened at revision "
                    f"{self._open_revision}, current revision {current}); "
                    f"model edits are not rewindable"
                )
        if not self._bodies and not self._geoms:
            result = ModelEditResult(backend=self._backend, revision=self._revision)
            self._result = result
            return result
        bodies = tuple(
            BodyEdit(
                articulation=pending.articulation,
                body=pending.body,
                body_id=pending.body_id,
                values=dict(pending.values),
            )
            for pending in self._bodies.values()
        )
        geoms = tuple(
            GeomEdit(
                articulation=pending.articulation,
                body=pending.body,
                geom=pending.geom,
                geom_id=pending.geom_id,
                values=dict(pending.values),
            )
            for pending in self._geoms.values()
        )
        result = self._commit(bodies, geoms)
        if not isinstance(result, ModelEditResult):
            raise TypeError(
                "model edit: the backend commit callback returned "
                f"{type(result).__name__}, expected ModelEditResult"
            )
        if result.backend != self._backend:
            raise ModelEditError(
                f"model edit: backend callback returned result for "
                f"{result.backend!r}, expected {self._backend!r}"
            )
        self._result = result
        self._revision = result.revision
        return result

    def rollback(self) -> None:
        """Discard every staged edit and close the transaction."""
        self._check_open()
        self._closed = True
        self._result = None

    def __enter__(self) -> ModelEditor:
        """Enter the transaction."""
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Commit on clean exit; roll back when an exception escaped."""
        del exc, tb
        if exc_type is None:
            if not self._closed:
                self.commit()
            return
        if not self._closed:
            self._closed = True
            self._result = None

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _check_open(self) -> None:
        """Reject use of a closed editor or a closed/stepping env."""
        if self._closed:
            raise ModelEditError(
                "model edit: this transaction is already closed; open a new "
                "run.model_edit() editor"
            )
        if self._is_open is not None and not self._is_open():
            raise ModelEditError(
                "model edit: the environment is closed or stepping; edits are "
                "only allowed between steps"
            )

    def _require_fields(self, values: Mapping[ModelEditField, object]) -> None:
        """Reject fields the backend does not declare writable."""
        unknown = sorted(_field_name(field) for field in values if field not in self._fields)
        if unknown:
            raise ModelEditCapabilityError(
                f"model edit: backend {self._backend!r} cannot edit {unknown}; "
                f"writable fields: {_supported(self._fields)}"
            )

    def _scope_label(self, articulation: str | None) -> str:
        """Return the human scope label for messages."""
        return "the merged scene" if articulation is None else f"articulation {articulation!r}"

    def _select(self, articulation: str | None) -> tuple[str | None, ModelIR]:
        """Resolve an articulation scope to ``(name-or-None, IR)``."""
        if articulation is None:
            if len(self._model_set) == 1:
                name = next(iter(self._model_set))
                return name, self._model_set[name]
            if self._model_set.merged is not None:
                return None, self._model_set.merged
            raise ModelEditError(
                "model edit: this run has several articulations and no "
                f"merged IR; pass an articulation name from "
                f"{self._model_set.names()}"
            )
        try:
            return articulation, self._model_set[articulation]
        except KeyError:
            raise ModelEditError(
                f"model edit: unknown articulation {articulation!r}; "
                f"candidates: {_candidates(self._model_set.names())}"
            ) from None

    def _resolve_body(self, ir: ModelIR, body: str) -> Body:
        """Resolve a body name against *ir* with a candidate list."""
        try:
            return ir.bodies[body]
        except KeyError:
            raise ModelEditError(
                f"model edit: body {body!r} is not in this model; "
                f"candidates: {_candidates(ir.bodies.names())}"
            ) from None


def trait_required(backend: str) -> None:
    """Raise when *backend* does not declare ``Trait.MODEL_EDIT``.

    Small shared helper for the run/env facades; keeps the error text in one
    place.
    """
    from unigym.backends.registry import backend_info

    if Trait.MODEL_EDIT not in backend_info(backend).traits:
        raise NotImplementedError(
            f"backend {backend!r} does not declare the MODEL_EDIT trait "
            f"(run.model_edit is optional and this backend declares it absent)"
        )
