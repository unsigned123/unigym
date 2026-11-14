"""Contract for the content-filler protocol rename and the ``objects.filler`` key."""

from __future__ import annotations

import pytest

from unigym.scenes.pipeline import (
    SceneBindingError,
    register_scene_content_filler,
    registered_scene_content_fillers,
    restore_scene_content_fillers,
    scene_content_filler,
    snapshot_scene_content_fillers,
)
from unigym.scenes.recipe import normalize_content_keys
from unigym.scenes.scene_ir import Scene


@pytest.fixture(autouse=True)
def _isolated_content_fillers():
    """Snapshot and restore the content-filler registry around each test."""
    snapshot = snapshot_scene_content_fillers()
    try:
        yield snapshot
    finally:
        restore_scene_content_fillers(snapshot)


def _tabletop_filler(**_kwargs):
    """Build a tabletop content filler."""
    return object()


def test_register_lookup_and_list_content_fillers() -> None:
    """A filler registers once, resolves by name and lists sorted."""
    register_scene_content_filler("tabletop_filler", _tabletop_filler)

    assert scene_content_filler("tabletop_filler") is _tabletop_filler
    assert registered_scene_content_fillers() == ["tabletop_filler"]


@pytest.mark.parametrize("name", ["", "Tabletop", "a.b", "_filler", "1filler"])
def test_content_filler_names_stay_snake_case(name: str) -> None:
    """Content fillers keep the closed snake_case grammar."""
    with pytest.raises(SceneBindingError, match="lowercase snake_case"):
        register_scene_content_filler(name, _tabletop_filler)


def test_content_filler_registration_rejects_non_callables_and_duplicates() -> None:
    """A non-callable and a duplicate name fail closed."""
    with pytest.raises(SceneBindingError, match="must be callable"):
        register_scene_content_filler("bad_filler", object())

    register_scene_content_filler("tabletop_filler", _tabletop_filler)
    with pytest.raises(SceneBindingError, match="registered twice"):
        register_scene_content_filler("tabletop_filler", _tabletop_filler)


def test_content_filler_has_no_hidden_default() -> None:
    """An undeclared filler never falls back to the old ``robocasa`` default."""
    with pytest.raises(SceneBindingError, match="not registered"):
        scene_content_filler("robocasa")


def test_scene_fill_requires_an_explicit_filler() -> None:
    """Neither the shared fill seam nor the mujoco router invents a filler."""
    from unigym.backends.mujoco.build import route_content
    from unigym.backends.shared.mjfamily.scene.scene_artifacts import (
        _fill_scene_content,
    )

    with pytest.raises(ValueError, match="requires a non-empty 'filler'"):
        _fill_scene_content(Scene(name="tabletop"), {"sets": {"cup": 1}}, None)

    with pytest.raises(ValueError, match="requires a non-empty 'filler'"):
        route_content({"sets": {"cup": 1}}, {})


def test_objects_filler_splits_into_the_content_spec() -> None:
    """``objects.filler`` (with sets/seed) is part of the content spec."""
    cfg = {
        "objects": {
            "types": [{"name": "cup"}],
            "filler": "tabletop_filler",
            "sets": {"cup": 1},
            "seed": 2,
        }
    }

    result = normalize_content_keys(cfg)

    assert result["props"] == {"types": [{"name": "cup"}]}
    assert result["content"] == {
        "filler": "tabletop_filler",
        "sets": {"cup": 1},
        "seed": 2,
    }


def test_objects_provider_is_rejected_with_the_replacement_key() -> None:
    """The legacy ``objects.provider`` spelling fails and names ``objects.filler``."""
    with pytest.raises(ValueError, match=r"objects\.filler"):
        normalize_content_keys({"objects": {"provider": "tabletop_filler"}})
