"""IsaacLab 2 model-edit provider: body mass/COM/inertia across every env.

The PhysX view exposes per-env rows; v1 broadcasts one edit to all env rows
(there is no ``env_ids`` face).  Geom material edits have no published geom
IR in this backend, so they are a capability error rather than a silent
no-op.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from unigym.capabilities.traits import ISAACLAB2_MODEL_EDIT_FIELDS, ModelEditField
from unigym.model.edit import BodyEdit, GeomEdit, ModelEditResult
from unigym.model.errors import ModelEditCapabilityError, ModelEditError

__all__ = ["IsaacLab2ModelEditProvider"]


def _view(asset: object) -> object | None:
    """Return the articulation's root PhysX view when exposed."""
    view = getattr(asset, "root_physx_view", None)
    if view is None:
        view = getattr(asset, "_root_physx_view", None)
    return view


def _names(value: object) -> list[str]:
    """Return a name sequence as a plain list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _body_names(asset: object) -> list[str]:
    """Return the live body order used by the PhysX view getters/setters.

    ``data.body_names`` is the live ``root_physx_view`` link order, the same
    order the view getters/setters index.  It is the single live accessor:
    neither the asset-level authored list nor a second view property is a
    fallback.
    """
    data = getattr(asset, "data", None)
    return _names(getattr(data, "body_names", None))


def _rotation_matrix(quat_wxyz: Iterable[float]) -> tuple[tuple[float, ...], ...]:
    """Return the rotation matrix of a wxyz quaternion."""
    w, x, y, z = (float(value) for value in quat_wxyz)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _inertia_row(
    inertia: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[float, ...]:
    """Return the row-major ``R @ diag(I) @ R.T`` inertia matrix."""
    rotation = _rotation_matrix(quat)
    rows: list[float] = []
    for row in range(3):
        for column in range(3):
            value = 0.0
            for axis in range(3):
                value += rotation[row][axis] * inertia[axis] * rotation[column][axis]
            rows.append(value)
    return tuple(rows)


def _as_tensor(value: object) -> object:
    """Return *value* as a torch tensor (imported lazily)."""
    import torch

    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


class IsaacLab2ModelEditProvider:
    """Apply global body edits through ``root_physx_view`` setters."""

    fields: frozenset[ModelEditField] = ISAACLAB2_MODEL_EDIT_FIELDS

    def apply(
        self,
        env: object,
        bodies: Iterable[BodyEdit],
        geoms: Iterable[GeomEdit],
    ) -> ModelEditResult:
        """Broadcast body edits to every env row, then refresh the model set."""
        body_edits = tuple(bodies)
        geom_edits = tuple(geoms)
        if geom_edits:
            raise ModelEditCapabilityError(
                "isaaclab2 model edit: geom fields are not editable (the "
                "backend publishes no geom IR face); editable fields: "
                + ", ".join(sorted(field.value for field in self.fields))
            )
        robots = getattr(env, "robots", {})
        planned = self._plan(robots, body_edits)
        snapshots = self._snapshot(planned)
        try:
            for _, view, index, values in planned:
                self._apply_one(view, index, values)
            rebuild = getattr(env, "_rebuild_model_set")
            rebuild()
            revision = int(getattr(env, "_model_revision")) + 1
            setattr(env, "_model_revision", revision)
            return ModelEditResult(
                backend="isaaclab2",
                revision=revision,
                bodies=body_edits,
                geoms=(),
                warnings=(),
            )
        except Exception as exc:
            self._restore(snapshots)
            if isinstance(exc, ModelEditError):
                raise
            raise ModelEditError(f"isaaclab2 model edit failed: {exc}") from exc

    def _plan(
        self,
        robots: object,
        edits: tuple[BodyEdit, ...],
    ) -> list[tuple[str, object, int, Mapping[ModelEditField, object]]]:
        """Resolve and validate every target before the first PhysX write."""
        assets = robots if hasattr(robots, "get") else {}
        planned: list[tuple[str, object, int, Mapping[ModelEditField, object]]] = []
        for edit in edits:
            if edit.articulation is None:
                raise ModelEditError(
                    "isaaclab2 model edit: a merged-scene body edit has no " "articulation binding"
                )
            asset = assets.get(edit.articulation)
            if asset is None:
                raise ModelEditError(
                    f"isaaclab2 model edit: unknown articulation " f"{edit.articulation!r}"
                )
            view = _view(asset)
            if view is None:
                raise ModelEditCapabilityError(
                    f"isaaclab2 model edit: articulation "
                    f"{edit.articulation!r} exposes no root_physx_view"
                )
            body_names = _body_names(asset)
            if not body_names:
                raise ModelEditCapabilityError(
                    f"isaaclab2 model edit: articulation "
                    f"{edit.articulation!r} exposes no live data.body_names; "
                    f"cannot resolve the PhysX view body index"
                )
            try:
                index = body_names.index(edit.body)
            except ValueError:
                raise ModelEditError(
                    f"isaaclab2 model edit: body {edit.body!r} is not in "
                    f"articulation {edit.articulation!r}; candidates: "
                    f"{', '.join(body_names)}"
                ) from None
            unknown = sorted(field.value for field in edit.values if field not in self.fields)
            if unknown:
                raise ModelEditCapabilityError(
                    f"isaaclab2 model edit: backend 'isaaclab2' cannot edit "
                    f"{unknown}; writable fields: "
                    f"{', '.join(sorted(field.value for field in self.fields))}"
                )
            if (
                ModelEditField.BODY_INERTIA_QUAT in edit.values
                and ModelEditField.BODY_INERTIA not in edit.values
            ):
                raise ModelEditError(
                    "isaaclab2 model edit: body.inertia_quat requires "
                    "body.inertia in the same BodyEditHandle.set() call"
                )
            self._require_view_api(view, edit.articulation)
            planned.append((edit.articulation, view, index, edit.values))
        return planned

    @staticmethod
    def _require_view_api(view: object, articulation: str) -> None:
        """Require the getter/setter pairs the snapshot + restore needs."""
        for name in ("masses", "coms", "inertias"):
            getter = getattr(view, f"get_{name}", None)
            setter = getattr(view, f"set_{name}", None)
            if getter is None or setter is None:
                raise ModelEditCapabilityError(
                    f"isaaclab2 model edit: articulation {articulation!r} "
                    f"root_physx_view lacks get_{name}/set_{name}"
                )

    def _snapshot(
        self,
        planned: list[tuple[str, object, int, Mapping[ModelEditField, object]]],
    ) -> dict[str, tuple[object, object, object, object]]:
        """Snapshot full PhysX arrays for every touched articulation."""
        snapshots: dict[str, tuple[object, object, object, object]] = {}
        for articulation, view, _, _ in planned:
            if articulation in snapshots:
                continue
            masses = _as_tensor(getattr(view, "get_masses")())
            if getattr(masses, "ndim", 0) == 1:
                masses = masses.unsqueeze(0)
            coms = _as_tensor(getattr(view, "get_coms")())
            inertias = _as_tensor(getattr(view, "get_inertias")())
            snapshots[articulation] = (
                view,
                masses.clone(),
                coms.clone(),
                inertias.clone(),
            )
        return snapshots

    def _restore(self, snapshots: dict[str, tuple[object, object, object, object]]) -> None:
        """Restore every snapshot row; a restore failure never masks the cause."""
        for view, masses, coms, inertias in snapshots.values():
            try:
                getattr(view, "set_masses")(masses, self._indices(masses))
            except Exception:  # pragma: no cover - best effort rollback
                pass
            try:
                getattr(view, "set_coms")(coms, self._indices(coms))
            except Exception:  # pragma: no cover - best effort rollback
                pass
            try:
                getattr(view, "set_inertias")(inertias, self._indices(inertias))
            except Exception:  # pragma: no cover - best effort rollback
                pass

    def _apply_one(
        self,
        view: object,
        index: int,
        values: Mapping[ModelEditField, object],
    ) -> None:
        """Write one pre-validated body edit through the articulation view."""
        if ModelEditField.BODY_MASS in values:
            self._write_scalar(view, "masses", index, float(values[ModelEditField.BODY_MASS]))
        if ModelEditField.BODY_COM in values:
            com = tuple(float(item) for item in values[ModelEditField.BODY_COM])
            self._write_row(view, "coms", index, com)
        if ModelEditField.BODY_INERTIA in values:
            inertia = tuple(float(item) for item in values[ModelEditField.BODY_INERTIA])
            if ModelEditField.BODY_INERTIA_QUAT in values:
                quat = tuple(float(item) for item in values[ModelEditField.BODY_INERTIA_QUAT])
            else:
                quat = (1.0, 0.0, 0.0, 0.0)
            row = _inertia_row(inertia, quat)
            if not all(math.isfinite(item) for item in row):  # pragma: no cover
                raise ModelEditError("isaaclab2 model edit: non-finite inertia")
            self._write_row(view, "inertias", index, row)

    @staticmethod
    def _indices(rows: object) -> object:
        """All-row articulation indices for the broadcast write (no env_ids)."""
        import torch

        count = int(getattr(rows, "shape")[0])
        return torch.arange(count, device=getattr(rows, "device"), dtype=torch.int32)

    def _write_scalar(self, view: object, name: str, index: int, value: float) -> None:
        """Broadcast one scalar to every env row of *name*."""
        getter = getattr(view, f"get_{name}", None)
        setter = getattr(view, f"set_{name}", None)
        if getter is None or setter is None:
            raise ModelEditCapabilityError(
                f"isaaclab2 model edit: root_physx_view lacks get_{name}/set_{name}"
            )
        current = _as_tensor(getter())
        if getattr(current, "ndim", 0) == 1:
            current = current.unsqueeze(0)
        updated = current.clone()
        updated[:, index] = value
        setter(updated, self._indices(updated))

    def _write_row(
        self,
        view: object,
        name: str,
        index: int,
        value: tuple[float, ...],
    ) -> None:
        """Broadcast one COM/inertia row to every env row of *name*."""
        getter = getattr(view, f"get_{name}", None)
        setter = getattr(view, f"set_{name}", None)
        if getter is None or setter is None:
            raise ModelEditCapabilityError(
                f"isaaclab2 model edit: root_physx_view lacks get_{name}/set_{name}"
            )
        import torch

        current = _as_tensor(getter())
        shape = tuple(getattr(current, "shape", ()))
        if len(shape) == 2:
            rows = current.unsqueeze(0)
            single = True
        elif len(shape) == 3:
            rows = current
            single = False
        else:  # pragma: no cover - unexpected PhysX layout
            raise ModelEditError(f"isaaclab2 model edit: unsupported {name} shape {shape}")
        updated = rows.clone()
        row = torch.as_tensor(value, device=updated.device, dtype=updated.dtype)
        if name == "coms":
            # PhysX packs (x, y, z, qx, qy, qz, qw); keep the principal-axis
            # orientation and replace only the translation.
            updated[:, index, 0:3] = row
        elif name == "inertias":
            updated[:, index, :] = row
        else:  # pragma: no cover - defensive
            raise ModelEditError(f"isaaclab2 model edit: unsupported row field {name}")
        setter(updated[0] if single else updated, self._indices(updated))
