"""MuJoCo-Warp consumes the same generic scene_provider keys (no mujoco block)."""

from __future__ import annotations

import pytest

from unigym.scenes.pipeline import (
    register_scene_content_filler,
    restore_scene_content_fillers,
    snapshot_scene_content_fillers,
)
from unigym.scenes.providers import (
    register_scene_provider,
    restore_scene_providers,
    snapshot_scene_providers,
)
from unigym.scenes.scene_ir import Scene


def _asset_plan():
    """An explicit empty asset boundary (the scene has no asset leaves)."""
    from unigym.assets.plan import AssetPlan
    from unigym.packs.runtime import AssetContext, discover

    return AssetPlan(AssetContext(discover([])))


@pytest.fixture
def _provider_registry():
    snapshot = snapshot_scene_providers()
    filler_snapshot = snapshot_scene_content_fillers()
    try:
        yield
    finally:
        restore_scene_providers(snapshot)
        restore_scene_content_fillers(filler_snapshot)


_CALLS: list[tuple[dict, object]] = []


def _table_top(params, context):
    """Record the dispatch and return an empty neutral scene."""
    _CALLS.append((dict(params), context))
    return Scene(name="tabletop")


def test_mjwarp_composes_the_scene_provider_with_its_backend_params(
    tiny_env_factory, _provider_registry
) -> None:
    _CALLS.clear()
    register_scene_provider("acme.table_top", _table_top)

    env = tiny_env_factory(
        params={
            "scene_provider": "acme.table_top",
            "scene_provider_params": {"seed": 2},
            "scene_provider_preset": "acme.tabletop",
            "base": {"joint": "fixed"},
        },
        asset_plan=_asset_plan(),
    )

    assert env.scene_provider == "acme.table_top"
    assert env._scene_provider_params == {"seed": 2}
    params, context = _CALLS[0]
    assert params == {"seed": 2}
    assert context.preset == "acme.tabletop"


@pytest.mark.parametrize("key", ["mj_scene", "mj_scene_params"])
def test_mjwarp_rejects_legacy_mj_scene_params(tiny_env_factory, key: str) -> None:
    """Direct construction must not silently ignore the removed scene keys."""
    from unigym.errors import ConfigError

    value = "acme.table_top" if key == "mj_scene" else {"seed": 1}
    with pytest.raises(ConfigError, match=f"{key} was removed.*scene_preset"):
        tiny_env_factory(params={key: value}, asset_plan=_asset_plan())


_FILLS: list[dict] = []


class _RecordingFiller:
    """Record the routed content spec and hand the build a fill record."""

    last_record = {"filled": ["cup"]}

    def fill(self, scene, spec, rng):
        _FILLS.append({"scene": scene.name, "spec": dict(spec)})
        return ()


def test_mjwarp_routes_a_content_fill_after_scene_composition(
    tiny_env_factory, tmp_path, _provider_registry
) -> None:
    _CALLS.clear()
    _FILLS.clear()
    register_scene_provider("acme.table_top", _table_top)
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "catalog.json").write_text(
        '{"schema": "unigym.catalog/1", "entries": []}', encoding="utf-8"
    )

    def _factory(catalog, generator_exists):
        return _RecordingFiller()

    _factory.catalog_ref = str(catalog_dir)
    register_scene_content_filler("test_filler", _factory)

    env = tiny_env_factory(
        params={
            "scene_provider": "acme.table_top",
            "scene_provider_params": {},
            "content": {"filler": "test_filler", "sets": {"cup": 1}, "seed": 1},
            "base": {"joint": "fixed"},
        },
        asset_plan=_asset_plan(),
    )

    assert env.scene_provider == "acme.table_top"
    assert env._content_spec == {"filler": "test_filler", "sets": {"cup": 1}, "seed": 1}
    assert _FILLS == [{"scene": "tabletop", "spec": {"sets": {"cup": 1}, "seed": 1}}]
    assert env._fill_record == {"filled": ["cup"]}
