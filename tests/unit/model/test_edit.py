"""Neutral model-edit transaction: validation, lifecycle and atomicity."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from unigym.backends.registry import ModelEditField
from unigym.model import (
    Body,
    BodyEdit,
    Geom,
    GeomEdit,
    ModelEditResult,
    ModelEditor,
    ModelIR,
    ModelSet,
    ModelEditCapabilityError,
    ModelEditError,
    Table,
)

ALL_BODY_FIELDS = frozenset(
    {
        ModelEditField.BODY_MASS,
        ModelEditField.BODY_COM,
        ModelEditField.BODY_INERTIA,
        ModelEditField.BODY_INERTIA_QUAT,
    }
)
ALL_GEOM_FIELDS = frozenset({ModelEditField.GEOM_FRICTION, ModelEditField.GEOM_RESTITUTION})
ALL_FIELDS = ALL_BODY_FIELDS | ALL_GEOM_FIELDS


def _ir(name: str = "arm", *, offset: int = 0) -> ModelIR:
    bodies = Table(
        [
            Body(name="world", path=("world",), qualname="world", body_id=0 + offset),
            Body(
                name="link",
                parent="world",
                path=("world", "link"),
                qualname="world/link",
                body_id=1 + offset,
                mass=1.0,
                inertia=(0.1, 0.1, 0.1),
                ipos=(0.0, 0.0, 0.0),
                iquat=(1.0, 0.0, 0.0, 0.0),
            ),
            Body(
                name="tip",
                parent="link",
                path=("world", "link", "tip"),
                qualname="world/link/tip",
                body_id=2 + offset,
                mass=0.5,
            ),
        ]
    )
    geoms = Table(
        [
            Geom(name="link_geom", type="box", body="link", geom_id=0 + offset),
            Geom(name="tip_geom", type="box", body="tip", geom_id=1 + offset),
        ]
    )
    return ModelIR(
        name=name,
        backend="fake",
        compiled=True,
        bodies=bodies,
        geoms=geoms,
    )


def _set(*, single: bool = True, merged: bool = True) -> ModelSet:
    ir = _ir()
    if single:
        return ModelSet({"arm": ir}, merged=ir if merged else None)
    return ModelSet(
        {"left": _ir("left", offset=0), "right": _ir("right", offset=10)},
        merged=ir if merged else None,
    )


class _Recorder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[tuple[BodyEdit, ...], tuple[GeomEdit, ...]]] = []
        self.fail = fail
        self.revision = 4

    def __call__(
        self,
        bodies: tuple[BodyEdit, ...],
        geoms: tuple[GeomEdit, ...],
    ) -> ModelEditResult:
        self.calls.append((bodies, geoms))
        if self.fail:
            raise ModelEditError("fake backend refused the write")
        self.revision += 1
        return ModelEditResult(
            backend="fake",
            revision=self.revision,
            bodies=bodies,
            geoms=geoms,
            warnings=("fake warning",),
        )


def _editor(
    *,
    fields: frozenset[ModelEditField] = ALL_FIELDS,
    model_set: ModelSet | None = None,
    recorder: _Recorder | None = None,
    is_open=None,
    current_revision=None,
) -> ModelEditor:
    commit = recorder if recorder is not None else _Recorder()
    return ModelEditor(
        backend="fake",
        fields=fields,
        model_set=model_set if model_set is not None else _set(),
        revision=4,
        commit=commit,
        is_open=is_open,
        current_revision=current_revision,
    )


def test_body_and_geom_set_resolve_ids_and_normalize_values() -> None:
    """Names resolve to compiled ids and the quaternion is normalized."""
    recorder = _Recorder()
    with _editor(recorder=recorder) as edit:
        edit.body("arm", "link").set(
            mass=2,
            com=(0.01, 0.02, 0.03),
            inertia=(0.2, 0.3, 0.4),
            inertia_quat=(2.0, 0.0, 0.0, 0.0),
        )
        edit.geom("arm", body="link", geom="link_geom").set(
            friction=(0.9, 0.02, 0.001),
            restitution=0.2,
        )

    assert len(recorder.calls) == 1
    bodies, geoms = recorder.calls[0]
    (body,) = bodies
    (geom,) = geoms
    assert (body.articulation, body.body, body.body_id) == ("arm", "link", 1)
    assert body.values[ModelEditField.BODY_MASS] == 2.0
    assert body.values[ModelEditField.BODY_INERTIA_QUAT] == (1.0, 0.0, 0.0, 0.0)
    assert (geom.articulation, geom.body, geom.geom, geom.geom_id) == (
        "arm",
        "link",
        "link_geom",
        0,
    )
    assert geom.values[ModelEditField.GEOM_RESTITUTION] == 0.2


def test_result_carries_revision_and_warnings() -> None:
    """A successful commit exposes the backend result and new revision."""
    with _editor() as edit:
        edit.body("arm", "link").set(mass=2.0)

    assert edit.result is not None
    assert edit.result.revision == 5
    assert edit.result.warnings == ("fake warning",)
    assert edit.revision == 5
    assert edit.closed


def test_empty_transaction_is_a_revision_preserving_noop() -> None:
    """No staged edits means no backend call and no revision bump."""
    recorder = _Recorder()
    with _editor(recorder=recorder) as edit:
        pass

    assert recorder.calls == []
    assert edit.result is not None
    assert edit.result.empty
    assert edit.revision == 4


def test_duplicate_field_in_one_transaction_is_an_error() -> None:
    """The same field may not be staged twice, even with the same value."""
    edit = _editor()
    edit.body("arm", "link").set(mass=1.0)
    with pytest.raises(ModelEditError, match="duplicate"):
        edit.body("arm", "link").set(mass=1.0)


def test_set_with_no_field_is_an_error() -> None:
    """An empty ``.set()`` teaches the accepted fields."""
    with pytest.raises(ModelEditError, match="at least one field"):
        _editor().body("arm", "link").set()
    with pytest.raises(ModelEditError, match="at least one field"):
        _editor().geom("arm", body="link", geom="link_geom").set()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (ModelEditField.BODY_MASS, 0.0, "mass must be > 0"),
        (ModelEditField.BODY_MASS, -1.0, "mass must be > 0"),
        (ModelEditField.BODY_MASS, float("inf"), "must be finite"),
        (ModelEditField.BODY_COM, (0.0, 0.0), "must have 3 elements"),
        (ModelEditField.BODY_INERTIA, (0.1, -0.1, 0.1), "inertia must be >= 0"),
        (ModelEditField.BODY_INERTIA_QUAT, (0.0, 0.0, 0.0, 0.0), "zero-norm"),
        (ModelEditField.GEOM_FRICTION, (0.0, -1.0, 0.0), "friction must be >= 0"),
        (ModelEditField.GEOM_RESTITUTION, 1.0, "restitution must be in"),
    ],
)
def test_invalid_values_fail_before_commit(field, value, message) -> None:
    """Field validation is fail-fast and names the field."""
    edit = _editor()
    with pytest.raises(ModelEditError, match=message):
        if field in ALL_BODY_FIELDS:
            edit.body("arm", "link").set(**{field.name.lower().replace("body_", ""): value})
        else:
            edit.geom("arm", body="link", geom="link_geom").set(
                **{field.name.lower().replace("geom_", ""): value}
            )


def test_unknown_names_name_the_candidates() -> None:
    """Unknown articulation/body/geom errors list valid names."""
    edit = _editor()
    with pytest.raises(ModelEditError, match=r"unknown articulation 'nope'.*arm"):
        edit.body("nope", "link")
    with pytest.raises(ModelEditError, match=r"body 'nope'.*link, tip, world"):
        edit.body("arm", "nope")
    with pytest.raises(ModelEditError, match=r"geom 'nope'.*link_geom, tip_geom"):
        edit.geom("arm", body="link", geom="nope")


def test_world_body_parameter_edits_are_rejected() -> None:
    """``body_id=0`` is the static world; body parameters are not editable."""
    with pytest.raises(ModelEditError, match="world body"):
        _editor().body("arm", "world")


def test_geom_must_belong_to_the_named_body() -> None:
    """A geom attached to another body is rejected, not silently retargeted."""
    with pytest.raises(ModelEditError, match="belongs to body 'link'"):
        _editor().geom("arm", body="tip", geom="link_geom")


def test_unsupported_field_is_a_capability_error() -> None:
    """The editor refuses fields outside the backend's declared set."""
    edit = _editor(fields=frozenset({ModelEditField.BODY_MASS}))
    with pytest.raises(ModelEditCapabilityError, match=r"body.com.*body.mass"):
        edit.body("arm", "link").set(com=(0.0, 0.0, 0.0))
    with pytest.raises(ModelEditCapabilityError, match="geom.friction"):
        edit.geom("arm", body="link", geom="link_geom").set(friction=(1.0, 0.0, 0.0))


def test_env_ids_keyword_is_not_accepted() -> None:
    """The transaction is global-only: ``env_ids`` is a TypeError."""
    with pytest.raises(TypeError):
        _editor().body("arm", "link").set(mass=1.0, env_ids=[0])


def test_context_rolls_back_on_exception() -> None:
    """An escaping exception discards the pending writes."""
    recorder = _Recorder()
    edit = _editor(recorder=recorder)
    with pytest.raises(RuntimeError, match="boom"):
        with edit:
            edit.body("arm", "link").set(mass=2.0)
            raise RuntimeError("boom")

    assert recorder.calls == []
    assert edit.result is None
    assert edit.closed


def test_commit_failure_closes_and_records_nothing() -> None:
    """A backend refusal leaves the editor closed with no result."""
    recorder = _Recorder(fail=True)
    edit = _editor(recorder=recorder)
    edit.body("arm", "link").set(mass=2.0)

    with pytest.raises(ModelEditError, match="fake backend refused"):
        edit.commit()

    assert edit.result is None
    assert edit.closed
    with pytest.raises(ModelEditError, match="already closed"):
        edit.commit()


def test_rollback_and_double_close_are_guarded() -> None:
    """One editor is single-use; rollback after a clean commit is an error."""
    edit = _editor()
    edit.rollback()
    assert edit.closed
    with pytest.raises(ModelEditError, match="already closed"):
        edit.rollback()

    edit = _editor()
    edit.commit()
    with pytest.raises(ModelEditError, match="already closed"):
        edit.commit()


def test_stale_revision_is_rejected() -> None:
    """A commit after another model edit fails instead of overwriting."""
    current = [4]
    edit = _editor(current_revision=lambda: current[0])
    edit.body("arm", "link").set(mass=2.0)
    current[0] = 5

    with pytest.raises(ModelEditError, match="stale transaction"):
        edit.commit()


def test_closed_or_stepping_env_rejects_the_transaction() -> None:
    """The liveness probe guards opening and every handle mutation."""
    live = [True]
    edit = _editor(is_open=lambda: live[0])
    handle = edit.body("arm", "link")
    live[0] = False
    with pytest.raises(ModelEditError, match="closed or stepping"):
        handle.set(mass=2.0)
    with pytest.raises(ModelEditError, match="closed or stepping"):
        edit.body("arm", "tip")


def test_multi_articulation_requires_an_explicit_name_without_merge() -> None:
    """No merged IR means ``articulation=None`` cannot pick a primary."""
    edit = _editor(model_set=_set(single=False, merged=False))
    with pytest.raises(ModelEditError, match="several articulations"):
        edit.body(None, "link")
    edit.body("left", "link").set(mass=2.0)
    edit.body("right", "link").set(mass=3.0)


def test_single_articulation_none_selects_the_only_scope() -> None:
    """A one-articulation run binds its only IR when articulation is None."""
    recorder = _Recorder()
    with _editor(recorder=recorder) as edit:
        edit.body(None, "link").set(mass=2.0)
    (body,) = recorder.calls[0][0]
    assert body.articulation == "arm"


def test_result_values_mapping_is_frozen_against_later_mutation() -> None:
    """The result carries a snapshot of the staged values."""
    recorder = _Recorder()
    edit = _editor(recorder=recorder)
    handle = edit.body("arm", "link")
    handle.set(mass=2.0)
    edit.commit()
    (body,) = recorder.calls[0][0]
    assert isinstance(body.values, Mapping)
    assert dict(body.values)[ModelEditField.BODY_MASS] == 2.0


def test_duplicate_detection_uses_resolved_compiled_identity() -> None:
    """Different scopes reaching the same compiled body/geom share duplicate state."""
    edit = _editor(model_set=_set(single=False, merged=True))
    edit.body("left", "link").set(mass=1.0)
    with pytest.raises(ModelEditError, match="duplicate"):
        edit.body(None, "link").set(mass=2.0)

    edit = _editor(model_set=_set(single=False, merged=True))
    edit.geom("left", body="link", geom="link_geom").set(friction=(1.0, 0.0, 0.0))
    with pytest.raises(ModelEditError, match="duplicate"):
        edit.geom(None, body="link", geom="link_geom").set(friction=(0.5, 0.0, 0.0))
