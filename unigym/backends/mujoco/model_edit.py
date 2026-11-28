"""MuJoCo model-edit provider (shared CPU writer + published-set refresh)."""

from __future__ import annotations

from collections.abc import Iterable

from unigym.backends.shared.mjfamily.model_edit import (
    MujocoModelEditWriter,
    published_state,
    refresh_published_model,
    restore_published_state,
)
from unigym.backends.shared.mjfamily.model.provider import MujocoModelProvider
from unigym.capabilities.traits import MUJOCO_MODEL_EDIT_FIELDS, ModelEditField
from unigym.model.edit import BodyEdit, GeomEdit, ModelEditResult

__all__ = ["MujocoModelEditProvider"]


class MujocoModelEditProvider:
    """Apply global model edits to the live MuJoCo env."""

    fields: frozenset[ModelEditField] = MUJOCO_MODEL_EDIT_FIELDS

    def __init__(self) -> None:
        self._writer = MujocoModelEditWriter()

    def apply(
        self,
        env: object,
        bodies: Iterable[BodyEdit],
        geoms: Iterable[GeomEdit],
    ) -> ModelEditResult:
        """Write the edits, refresh the IR and bump the model revision."""
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
                refresh_published_model(env, MujocoModelProvider())
                revision = int(getattr(env, "_model_revision")) + 1
                setattr(env, "_model_revision", revision)
                return ModelEditResult(
                    backend="mujoco",
                    revision=revision,
                    bodies=body_edits,
                    geoms=geom_edits,
                    warnings=warnings,
                )
        except BaseException:
            restore_published_state(env, published)
            raise
