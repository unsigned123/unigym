"""Contract for the scene-provider registry (Commit A vocabulary split)."""

from __future__ import annotations

import pytest

from unigym.scenes import (
    ProviderOrigin,
    SceneProviderContext,
    build_scene_from_provider,
    register_scene_provider,
    registered_scene_providers,
    restore_scene_providers,
    scene_provider_origin,
    scene_provider_providers,
    snapshot_scene_providers,
)
from unigym.scenes.provider_params import ProviderParamSpec, SceneProviderParamError
from unigym.scenes.scene_ir import Scene
from unigym.world import World


@pytest.fixture(autouse=True)
def _isolated_scene_providers():
    """Snapshot and restore the scene-provider registries around each test."""
    snapshot = snapshot_scene_providers()
    try:
        yield snapshot
    finally:
        restore_scene_providers(snapshot)


def _table_top(params, context):
    """Build a tabletop scene."""
    return Scene(name="tabletop")


def test_dotted_provider_names_dispatch_params_and_context() -> None:
    """An adapter-grammar name registers and dispatches with data context."""
    calls: list[tuple[dict[str, object], SceneProviderContext]] = []

    def provider(params, context):
        """Record the dispatch and build a scene."""
        calls.append((dict(params), context))
        return Scene(name="tabletop")

    register_scene_provider("acme.table_top", provider)

    assert registered_scene_providers() == ["acme.table_top"]
    scene = build_scene_from_provider("acme.table_top", {"seed": 1}, preset="acme.tabletop")
    assert scene.name == "tabletop"
    assert calls == [
        (
            {"seed": 1},
            SceneProviderContext(name="acme.table_top", preset="acme.tabletop"),
        )
    ]


@pytest.mark.parametrize(
    "name",
    ["", "Acme", "9acme", "_acme", "acme..top", "acme.top-1", "acme/top"],
)
def test_provider_name_grammar_rejects_non_dotted_snake_case(name: str) -> None:
    """Path-like or non-adapter spellings fail at registration."""
    with pytest.raises(ValueError, match="lowercase snake_case"):
        register_scene_provider(name, _table_top)


def test_provider_registration_rejects_duplicates_factories_and_shapes() -> None:
    """Duplicate names, non-callables, missing docstrings and aliases fail."""
    with pytest.raises(ValueError, match="registered twice"):
        register_scene_provider("acme.table_top", _table_top)
        register_scene_provider("acme.table_top", _table_top)

    with pytest.raises(ValueError, match="must be callable"):
        register_scene_provider("acme.not_callable", object())  # type: ignore[arg-type]

    def undocumented(params, context):
        return Scene(name="undocumented")

    with pytest.raises(ValueError, match="requires a docstring"):
        register_scene_provider("acme.undocumented", undocumented)

    def second(params, context):
        """Reuse the first provider's factory under another name."""
        return Scene(name="second")

    register_scene_provider("acme.first", second)
    with pytest.raises(ValueError, match="one factory owns exactly one name"):
        register_scene_provider("acme.second", second)


def test_dispatch_fails_closed_on_unknown_names_and_bad_params() -> None:
    """An unknown provider lists the public set; params must be a mapping."""
    register_scene_provider("acme.table_top", _table_top)

    with pytest.raises(ValueError, match=r"acme\.table_top"):
        build_scene_from_provider("acme.missing", {})

    with pytest.raises(ValueError, match="params must be a mapping"):
        build_scene_from_provider("acme.table_top", [])  # type: ignore[arg-type]

    def not_a_scene(params, context):
        """Return the wrong output type."""
        return {"name": "not_a_scene"}

    register_scene_provider("acme.not_a_scene", not_a_scene)
    with pytest.raises(ValueError, match="must return a Scene"):
        build_scene_from_provider("acme.not_a_scene", {})


def test_registration_params_schema_fills_defaults_and_rejects_unknown_keys() -> None:
    """The code-form schema mirror fills defaults and validates before dispatch."""
    seen: list[dict[str, object]] = []

    def provider(params, context):
        """Record the filled parameter mapping."""
        seen.append(dict(params))
        return Scene(name="styled")

    register_scene_provider(
        "acme.styled",
        provider,
        params={
            "style": (int, 1),
            "seed": int,
            "layout": (str, "default", ("default", "dense")),
        },
    )

    build_scene_from_provider("acme.styled", {"seed": 3})
    assert seen == [{"style": 1, "seed": 3, "layout": "default"}]

    infos = scene_provider_providers()
    assert [info.name for info in infos] == ["acme.styled"]
    schema = infos[0].params_schema
    assert schema is not None
    assert schema["style"] == ProviderParamSpec(type="int", required=False, default=1)
    assert schema["seed"].required
    assert schema["layout"].choices == ("default", "dense")

    with pytest.raises(SceneProviderParamError, match="unknown parameter"):
        build_scene_from_provider("acme.styled", {"seed": 3, "extra": 1})

    with pytest.raises(SceneProviderParamError, match="missing required parameter"):
        build_scene_from_provider("acme.styled", {})

    with pytest.raises(SceneProviderParamError, match="must be int"):
        build_scene_from_provider("acme.styled", {"seed": True})

    with pytest.raises(SceneProviderParamError, match="acme.tabletop"):
        build_scene_from_provider(
            "acme.styled", {"seed": 3, "layout": "wide"}, preset="acme.tabletop"
        )


def test_inline_world_provider_stays_out_of_the_public_registry() -> None:
    """``World.inline`` registers privately, dispatches and reports its origin."""
    register_scene_provider("acme.table_top", _table_top)
    inline_world = World.inline(Scene(name="inline_scene"))
    inline_name = (inline_world.fragment().get("params") or {})["mj_scene"]
    assert isinstance(inline_name, str)

    assert registered_scene_providers() == ["acme.table_top"]
    infos = scene_provider_providers()
    assert [info.name for info in infos] == ["acme.table_top"]
    assert infos[0].kind == "scene"
    assert infos[0].origin == ProviderOrigin.unknown()
    assert scene_provider_providers(kind="robot") == ()

    assert scene_provider_origin(inline_name) == ProviderOrigin.inline()
    assert build_scene_from_provider(inline_name, {}).name == "inline_scene"


def test_origins_round_trip_and_unknown_is_explicit() -> None:
    """A supplied origin round-trips; a loose registration reports unknown."""
    origin = ProviderOrigin(
        kind="manifest",
        pack="acme",
        pack_version="1.2.0",
        root="/packs/acme",
        discovered_by="explicit",
        face="providers",
        item_id="acme.table_top",
        entrypoint="acme_scenes:table_top",
        activation_entrypoint="acme:register",
    )

    def other(params, context):
        """Build another scene."""
        return Scene(name="other")

    register_scene_provider("acme.table_top", _table_top, origin=origin)
    register_scene_provider("acme.other", other)

    assert scene_provider_origin("acme.table_top") == origin
    assert scene_provider_origin("acme.other") == ProviderOrigin.unknown()
    with pytest.raises(ValueError, match=r"acme\.table_top"):
        scene_provider_origin("acme.missing")


def test_snapshot_and_restore_are_total() -> None:
    """Restore keeps origins/params/inline and drops every later registration."""
    origin = ProviderOrigin(kind="pack-code", pack="acme", pack_version="1.2.0")
    register_scene_provider(
        "acme.table_top",
        _table_top,
        params={"style": (int, 1)},
        origin=origin,
    )
    inline_world = World.inline(Scene(name="inline_scene"))
    inline_name = (inline_world.fragment().get("params") or {})["mj_scene"]
    assert isinstance(inline_name, str)

    before = snapshot_scene_providers()

    def late(params, context):
        """Build a scene registered after the snapshot."""
        return Scene(name="late")

    register_scene_provider(
        "acme.late",
        late,
        params={"seed": int},
        origin=ProviderOrigin(kind="manifest"),
    )
    World.inline(Scene(name="late_inline"))

    restore_scene_providers(before)

    assert registered_scene_providers() == ["acme.table_top"]
    assert scene_provider_origin("acme.table_top") == origin
    assert scene_provider_providers()[0].params_schema == {
        "style": ProviderParamSpec(type="int", required=False, default=1)
    }
    assert build_scene_from_provider(inline_name, {}).name == "inline_scene"
    assert scene_provider_origin(inline_name) == ProviderOrigin.inline()
    with pytest.raises(ValueError, match="not registered"):
        build_scene_from_provider("acme.late", {})
    with pytest.raises(TypeError):
        restore_scene_providers({})  # type: ignore[arg-type]


def test_removed_public_names_are_absent() -> None:
    """The pre-split provider names are removed (no aliases)."""
    import unigym.pack_api as pack_api
    import unigym.scenes as scenes

    for name in (
        "SceneContext",
        "ContentProvider",
        "register_scene",
        "registered_scenes",
        "build_scene_definition",
        "PROVIDER_WHENS",
        "register_content_provider",
        "content_provider",
        "registered_content_providers",
    ):
        assert not hasattr(scenes, name), name

    assert not hasattr(pack_api, "register_scene")
    assert not hasattr(pack_api, "register_content_provider")
