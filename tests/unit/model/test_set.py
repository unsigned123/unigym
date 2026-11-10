"""ModelSet named lookup and the honest merged-face requirement."""

from __future__ import annotations

import pytest

from unigym.model import Body, ModelIR, ModelSet, ModelSetError, Table

_FACES = ("bodies", "joints", "meta")


def _ir(name: str) -> ModelIR:
    return ModelIR(
        name=name,
        backend=None,
        compiled=False,
        bodies=Table([Body(name="world")]),
    )


def test_model_set_lookup_and_iteration_has_no_primary_articulation():
    """Articulations are looked up by name and iterated in declaration order."""
    g1 = _ir("g1")
    g2 = _ir("g2")
    model_set = ModelSet({"g1": g1, "g2": g2})

    assert model_set["g1"] is g1
    assert model_set.names() == ["g1", "g2"]
    assert list(model_set) == ["g1", "g2"]
    assert len(model_set) == 2
    assert "g2" in model_set
    with pytest.raises(KeyError, match="missing"):
        model_set["missing"]

    empty = ModelSet({})
    assert len(empty) == 0
    assert empty.names() == []


@pytest.mark.parametrize("face", _FACES)
def test_model_set_table_face_reads_the_merged_ir(face):
    """An available merge answers every whole-scene face identically."""
    per_articulation = _ir("g1")
    merged = _ir("scene")
    model_set = ModelSet({"g1": per_articulation}, merged=merged)

    assert getattr(model_set, face) is getattr(merged, face)


@pytest.mark.parametrize("face", _FACES)
def test_model_set_table_face_fails_when_no_honest_merge_exists(face):
    """Without a merge, a whole-scene face fails and names the per-articulation path."""
    model_set = ModelSet({"g1": _ir("g1")})

    with pytest.raises(ModelSetError, match=face) as excinfo:
        getattr(model_set, face)
    message = str(excinfo.value)
    assert "per-articulation" in message
    assert "g1" in message
