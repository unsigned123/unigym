"""mjwarp model-edit provider (CPU writer + shared device-field push)."""

from __future__ import annotations

from collections.abc import Iterable

from unigym.backends.registry import MUJOCO_MODEL_EDIT_FIELDS, ModelEditField
from unigym.backends.mjwarp.model import MjwarpModelProvider
from unigym.backends.shared.mjfamily.model_edit import (
    MujocoModelEditWriter,
    published_state,
    refresh_published_model,
    restore_published_state,
)
from unigym.model.edit import BodyEdit, GeomEdit, ModelEditResult

__all__ = ["MjwarpModelEditProvider"]


class MjwarpModelEditProvider:
    """Apply global model edits on the CPU mirror and push the device model."""

    fields: frozenset[ModelEditField] = MUJOCO_MODEL_EDIT_FIELDS

    def __init__(self) -> None:
        self._writer = MujocoModelEditWriter()

    def apply(
        self,
        env: object,
        bodies: Iterable[BodyEdit],
        geoms: Iterable[GeomEdit],
    ) -> ModelEditResult:
        """Write CPU arrays, refresh the IR, then push the global device model."""
        body_edits = tuple(bodies)
        geom_edits = tuple(geoms)
        published = published_state(env)
        try:
            with self._writer.transaction(
                getattr(env, "mj_model"),
                getattr(env, "mj_datas")[0],
                body_edits,
                geom_edits,
            ) as warnings:
                refresh_published_model(env, MjwarpModelProvider())
                push = getattr(env, "_push_model_edit_fields")
                push()
                revision = int(getattr(env, "_model_revision")) + 1
                setattr(env, "_model_revision", revision)
                return ModelEditResult(
                    backend="mjwarp",
                    revision=revision,
                    bodies=body_edits,
                    geoms=geom_edits,
                    warnings=warnings,
                )
        except BaseException:
            restore_published_state(env, published)
            push = getattr(env, "_push_model_edit_fields", None)
            if callable(push):
                try:
                    push()
                except Exception:  # pragma: no cover - device resync best effort
                    pass
            raise
