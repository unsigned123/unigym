"""Shared MuJoCo-family model-edit writer.

Both ``mujoco`` and ``mjwarp`` edit the same compiled ``MjModel`` on the
host.  This module owns the array writes, the ``body_simple`` safety valve,
``mj_setConst`` recomputation and the rollback snapshot; the backend wrappers
add their own device push / ``ModelSet`` refresh.

The writer never calls ``mj_forward`` and never touches qpos/qvel or the
solver warm-start.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace

import mujoco
import numpy as np

from unigym.model.edit import BodyEdit, GeomEdit
from unigym.model.errors import ModelEditError
from unigym.backends.registry import ModelEditField

__all__ = ["MujocoModelEditWriter", "refresh_published_model"]

_BODY_ARRAY_FIELDS: dict[ModelEditField, str] = {
    ModelEditField.BODY_MASS: "body_mass",
    ModelEditField.BODY_COM: "body_ipos",
    ModelEditField.BODY_INERTIA: "body_inertia",
    ModelEditField.BODY_INERTIA_QUAT: "body_iquat",
}
_GEOM_ARRAY_FIELDS: dict[ModelEditField, str] = {
    ModelEditField.GEOM_FRICTION: "geom_friction",
    ModelEditField.GEOM_RESTITUTION: "geom_solref",
}
#: Fields whose write can invalidate MuJoCo's simple-body shortcut.
_SIMPLE_SENSITIVE = (ModelEditField.BODY_COM, ModelEditField.BODY_INERTIA_QUAT)


def _restitution_zeta(restitution: float) -> float:
    """Return a non-dimensional damping ratio approximating *restitution*.

    MuJoCo's ``solref = (timeconst, dampratio)`` model is not a restitution
    coefficient; this is the standard underdamped mapping
    ``e = exp(-zeta*pi/sqrt(1-zeta^2))``.  ``e = 0`` maps to critical
    damping.
    """
    if restitution <= 0.0:
        return 1.0
    log_e = math.log(restitution)
    zeta = -log_e / math.sqrt(math.pi * math.pi + log_e * log_e)
    return float(min(max(zeta, 0.0), 1.0))


def refresh_published_model(env: object, provider: object) -> None:
    """Re-describe the compiled model and rebuild ``env.model_set``.

    The previous IR's ``meta`` annotations (the run's articulation labels) are
    carried across so a model edit never drops run-level metadata.
    """
    describe = getattr(provider, "describe")
    ir = describe(getattr(env, "mj_model"))
    previous = getattr(env, "_native_model_ir", None)
    meta = getattr(previous, "meta", None)
    if meta:
        ir = replace(ir, meta={**meta, **ir.meta})
    setattr(env, "_native_model_ir", ir)
    rebuild = getattr(env, "_rebuild_model_set")
    rebuild()


class _Rollback:
    """Snapshot the touched rows and restore them on failure."""

    def __init__(self, model: object) -> None:
        self._model = model
        self._rows: list[tuple[object, int, np.ndarray]] = []
        self._touched: set[tuple[str, int]] = set()

    def save(self, array_name: str, index: int) -> None:
        """Snapshot one 1-D/2-D row before its first write."""
        key = (array_name, index)
        if key in self._touched:
            return
        array = getattr(self._model, array_name, None)
        if array is None:
            return
        self._rows.append((array, index, np.array(array[index], copy=True)))
        self._touched.add(key)

    def restore(self) -> None:
        """Write every snapshot row back in reverse order."""
        for array, index, value in reversed(self._rows):
            array[index] = value
        self._rows.clear()
        self._touched.clear()


class MujocoModelEditWriter:
    """Apply neutral body/geom edits to one compiled ``MjModel``."""

    def __init__(self) -> None:
        self.fields = frozenset(_BODY_ARRAY_FIELDS) | frozenset(_GEOM_ARRAY_FIELDS)

    def apply(
        self,
        model: object,
        data: object,
        bodies: Iterable[BodyEdit],
        geoms: Iterable[GeomEdit],
    ) -> tuple[str, ...]:
        """Write the edits, run ``mj_setConst`` and return warnings.

        Args:
            model: The live ``mujoco.MjModel``.
            data: A scratch ``mujoco.MjData`` (index 0) for ``mj_setConst``.
            bodies: Resolved body edits.
            geoms: Resolved geom edits.

        Returns:
            Human-readable approximation warnings.

        Raises:
            ModelEditError: A target id is invalid, direct-mode ``solref`` is
                unsupported, or the model recomputation failed.  Every
                touched row is restored before the error escapes.
        """
        body_edits = tuple(bodies)
        geom_edits = tuple(geoms)
        warnings: list[str] = []
        rollback = _Rollback(model)
        simple = getattr(model, "body_simple", None)
        try:
            for edit in body_edits:
                index = self._body_index(model, edit)
                for field, value in edit.values.items():
                    array_name = _BODY_ARRAY_FIELDS.get(field)
                    if array_name is None:
                        raise ModelEditError(
                            f"mujoco model edit: unsupported body field {field.value}"
                        )
                    rollback.save(array_name, index)
                    getattr(model, array_name)[index] = value
                if simple is not None and any(field in edit.values for field in _SIMPLE_SENSITIVE):
                    rollback.save("body_simple", index)
                    simple[index] = 0
            for edit in geom_edits:
                index = self._geom_index(model, edit)
                for field, value in edit.values.items():
                    array_name = _GEOM_ARRAY_FIELDS.get(field)
                    if array_name is None:
                        raise ModelEditError(
                            f"mujoco model edit: unsupported geom field {field.value}"
                        )
                    rollback.save(array_name, index)
                    if field is ModelEditField.GEOM_RESTITUTION:
                        solref = getattr(model, "geom_solref")
                        if float(solref[index][0]) <= 0.0:
                            raise ModelEditError(
                                f"mujoco model edit: geom "
                                f"{edit.geom!r} uses direct-mode solref "
                                f"(solref[0]={float(solref[index][0])}); "
                                f"restitution approximation needs a "
                                f"positive time constant"
                            )
                        zeta = _restitution_zeta(float(value))
                        solref[index][1] = zeta
                        warnings.append(
                            f"geom {edit.geom!r}: restitution "
                            f"{float(value):.6g} is approximate; "
                            f"solref dampratio={zeta:.6g} "
                            f"(solref[0] retained)"
                        )
                    else:
                        getattr(model, array_name)[index] = value
            mujoco.mj_setConst(model, data)
        except Exception as exc:
            rollback.restore()
            if simple is not None:
                try:
                    mujoco.mj_setConst(model, data)
                except Exception:  # pragma: no cover - rollback best effort
                    pass
            if isinstance(exc, ModelEditError):
                raise
            raise ModelEditError(f"mujoco model edit failed: {exc}") from exc
        return tuple(warnings)

    @staticmethod
    def _body_index(model: object, edit: BodyEdit) -> int:
        """Return and range-check the edited body id."""
        index = edit.body_id
        if index is None:
            raise ModelEditError(f"mujoco model edit: body {edit.body!r} has no compiled body_id")
        count = int(getattr(model, "nbody"))
        if not 0 <= index < count:
            raise ModelEditError(
                f"mujoco model edit: body {edit.body!r} id {index} is outside "
                f"[0, nbody={count})"
            )
        return int(index)

    @staticmethod
    def _geom_index(model: object, edit: GeomEdit) -> int:
        """Return and range-check the edited geom id."""
        index = edit.geom_id
        if index is None:
            raise ModelEditError(f"mujoco model edit: geom {edit.geom!r} has no compiled geom_id")
        count = int(getattr(model, "ngeom"))
        if not 0 <= index < count:
            raise ModelEditError(
                f"mujoco model edit: geom {edit.geom!r} id {index} is outside "
                f"[0, ngeom={count})"
            )
        return int(index)
