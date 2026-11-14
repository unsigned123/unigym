"""Contract for scene preset resolution through the runtime ``scene_presets`` face."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from unigym.errors import ConfigError
from unigym.pack_api import discover
from unigym.packs import PackError
from unigym.run.api import Run
from unigym.run.spec import build_spec
from unigym.scene_presets import (
    ScenePresetError,
    apply_scene_preset,
    load_scene_preset,
    reject_legacy_scene_keys,
    serialize_scene_preset,
)
from unigym.scenes.providers import (
    register_scene_provider,
    restore_scene_providers,
    snapshot_scene_providers,
)
from unigym.scenes.scene_ir import Scene
from unigym.world import World

WORLD = "world.xml"
PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"

CONTENT_ONLY = """\
schema: unigym.scene_preset/1
name: tabletop
world:
  mjcf: world.xml
objects:
  types:
    - name: scene_box
backend_params:
  mujoco:
    base: {joint: fixed}
    initializer: {type: tabletop_initializer, params: {seed: 1}}
  mjwarp:
    base: {joint: fixed}
    initializer: {type: tabletop_initializer, params: {seed: 2}}
"""

PROVIDER_BACKED = """\
schema: unigym.scene_preset/1
name: tabletop
provider: acme.table_top
params:
  style: 1
world:
  mjcf: world.xml
objects:
  types:
    - name: scene_box
  filler: tabletop_filler
  sets: {cup: 1}
  seed: 1
backend_params:
  mujoco:
    base: {joint: fixed}
  mjwarp:
    base: {joint: fixed}
"""


def _pack_root(tmp_path: Path, preset_text: str) -> Path:
    pack = tmp_path / "acme_scene"
    (pack / "scene_presets").mkdir(parents=True, exist_ok=True)
    (pack / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        "name: acme_scene\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: code\n"
        "provides:\n"
        "  scene_presets:\n"
        "    - id: acme_scene.tabletop\n"
        "      resource: scene_presets/tabletop.yaml\n"
        "  providers:\n"
        "    - id: acme.table_top\n"
        "      kind: scene\n"
        "      entrypoint: acme_scenes:table_top\n"
        "      params:\n"
        "        style: {type: int, default: 1}\n"
        "        seed: {type: int}\n",
        encoding="utf-8",
    )
    (pack / "scene_presets" / "tabletop.yaml").write_text(preset_text, encoding="utf-8")
    (pack / WORLD).write_text(
        '<mujoco model="tabletop"><worldbody>'
        '<geom name="ground" type="plane" size="1 1 0.1"/>'
        "</worldbody></mujoco>\n",
        encoding="utf-8",
    )
    return pack


def _runtime(tmp_path: Path, preset_text: str = CONTENT_ONLY):
    return discover([_pack_root(tmp_path, preset_text)])


def _cfg(**overrides):
    cfg = {
        "run_name": "scene_run",
        "seed": 1,
        "env": {"backend": "mujoco", "num_envs": 1},
        "collect": {"steps": 10},
        "articulations": [
            {"name": "arm", "joints": [{"name": "shoulder"}, {"name": "elbow"}]},
        ],
        "controls": [
            {
                "name": "arm",
                "adapter": "manual",
                "articulation": "arm",
                "joints": ["shoulder", "elbow"],
                "drive": "position_pd",
                "target": "position",
            },
        ],
    }
    cfg.update(overrides)
    return cfg


def _args():
    return Namespace(config="run.yaml")


def _table_top(params, context):
    """Build a tabletop scene."""
    return Scene(name="tabletop")


@pytest.fixture
def _scene_registry():
    snapshot = snapshot_scene_providers()
    try:
        yield
    finally:
        restore_scene_providers(snapshot)


def test_preset_id_resolves_through_the_runtime_and_parses(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    preset = load_scene_preset("acme_scene.tabletop", runtime)

    assert preset.catalog_id == "acme_scene.tabletop"
    assert preset.name == "tabletop"
    assert preset.world is not None and preset.world.mjcf == WORLD
    assert preset.objects == {"types": [{"name": "scene_box"}]}
    assert set(preset.backend_params) == {"mujoco", "mjwarp"}


@pytest.mark.parametrize(
    ("preset_text", "match"),
    [
        ("key: [unclosed\n", "not valid YAML"),
        ("- one\n- two\n", "must be a mapping"),
    ],
    ids=["malformed-yaml", "non-mapping"],
)
def test_preset_resource_must_parse_as_a_mapping(
    tmp_path: Path, preset_text: str, match: str
) -> None:
    """A malformed or non-mapping preset resource fails at resolution."""
    runtime = _runtime(tmp_path, preset_text)

    with pytest.raises(ScenePresetError, match=match):
        load_scene_preset("acme_scene.tabletop", runtime)


def test_build_spec_routes_the_content_only_preset(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    spec = build_spec(
        _cfg(scene_preset="acme_scene.tabletop", objects={"types": [{"name": "run_box"}]}),
        _args(),
        runtime=runtime,
    )

    assert "scene_preset" not in spec.cfg
    assert spec.world_base == {"mjcf": WORLD}
    assert spec.props == {"types": [{"name": "run_box"}]}
    assert spec.backend_params["base"] == {"joint": "fixed"}
    assert spec.backend_params["initializer"] == {
        "type": "tabletop_initializer",
        "params": {"seed": 1},
    }


def test_run_initializer_partial_override_merges_over_the_preset(tmp_path: Path) -> None:
    """The run's env.params.initializer merges over preset defaults (no replace)."""
    runtime = _runtime(tmp_path)

    spec = build_spec(
        _cfg(
            scene_preset="acme_scene.tabletop",
            env={
                "backend": "mujoco",
                "num_envs": 1,
                "params": {"initializer": {"params": {"seed": 9}}},
            },
        ),
        _args(),
        runtime=runtime,
    )

    assert spec.backend_params["initializer"] == {
        "type": "tabletop_initializer",
        "params": {"seed": 9},
    }
    assert spec.backend_params["base"] == {"joint": "fixed"}


def test_preset_backend_block_cannot_route_actuators(tmp_path: Path) -> None:
    """Actuation is run-level IR; a preset backend block cannot route it."""
    text = CONTENT_ONLY.replace(
        "  mujoco:\n    base: {joint: fixed}\n",
        "  mujoco:\n    base: {joint: fixed}\n"
        "    actuators:\n      legs:\n        type: standard_pd\n",
    )
    runtime = _runtime(tmp_path, text)

    with pytest.raises(ConfigError, match="unsupported key.*actuators"):
        build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args(), runtime=runtime)


def test_backend_params_selects_the_run_backend_not_mujoco(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    spec = build_spec(
        _cfg(scene_preset="acme_scene.tabletop", env={"backend": "mjwarp", "num_envs": 1}),
        _args(),
        runtime=runtime,
    )

    assert spec.backend_params["initializer"]["params"] == {"seed": 2}


def test_provider_backed_preset_routes_content_and_params(tmp_path: Path, _scene_registry) -> None:
    register_scene_provider("acme.table_top", _table_top)
    runtime = _runtime(tmp_path, PROVIDER_BACKED)

    spec = build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args(), runtime=runtime)

    assert spec.backend_params["scene_provider"] == "acme.table_top"
    assert spec.backend_params["scene_provider_params"] == {"style": 1}
    assert spec.backend_params["scene_provider_preset"] == "acme_scene.tabletop"
    assert spec.cfg["content"] == {"filler": "tabletop_filler", "sets": {"cup": 1}, "seed": 1}
    assert spec.backend_params["content"] == {
        "filler": "tabletop_filler",
        "sets": {"cup": 1},
        "seed": 1,
    }


def test_inline_provider_form_uses_a_process_local_registry(
    tmp_path: Path, _scene_registry
) -> None:
    register_scene_provider("acme.table_top", _table_top)

    spec = build_spec(
        _cfg(
            scene_preset={
                "provider": "acme.table_top",
                "params": {"style": 2},
                "world": {"mjcf": "scene.xml"},
                "backend_params": {
                    "mujoco": {"initializer": {"type": "tabletop_initializer", "params": {}}}
                },
            }
        ),
        _args(),
    )

    assert spec.world_base == {"mjcf": "scene.xml"}
    assert spec.backend_params["scene_provider"] == "acme.table_top"
    assert spec.backend_params["scene_provider_params"] == {"style": 2}
    assert spec.backend_params["scene_provider_preset"] is None


def test_provider_backed_catalog_preset_round_trips_through_serialization(
    tmp_path: Path,
) -> None:
    """A catalog preset re-emits its declaration and resolves to the same run."""
    runtime = _runtime(tmp_path, PROVIDER_BACKED)
    first = build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args(), runtime=runtime)

    authored = serialize_scene_preset(first.cfg, first.scene_preset)
    # Mirror Run.to_config's surface spelling for the content block.
    authored["objects"] = authored.pop("props")

    assert first.scene_preset == "acme_scene.tabletop"
    assert authored["scene_preset"] == "acme_scene.tabletop"
    assert "scene_provider" not in authored["env"]["params"]
    assert "scene_provider_params" not in authored["env"]["params"]
    assert "scene_provider_preset" not in authored["env"]["params"]
    second = build_spec(authored, _args(), runtime=runtime)
    assert second.cfg == first.cfg
    assert second.scene_preset == first.scene_preset


def test_provider_backed_inline_run_round_trips_through_to_config(
    tmp_path: Path, _scene_registry
) -> None:
    """Run.to_config writes the authoring surface and reloads to the same run."""
    register_scene_provider("acme.table_top", _table_top)
    cfg = _cfg(
        scene_preset={
            "provider": "acme.table_top",
            "params": {"style": 2},
            "world": {"mjcf": str(tmp_path / WORLD)},
            "objects": {"types": [{"name": "run_box"}], "filler": "f", "sets": {"cup": 1}},
            "backend_params": {
                "mujoco": {
                    "base": {"joint": "fixed"},
                    "initializer": {"type": "tabletop_initializer", "params": {"seed": 3}},
                }
            },
        }
    )
    first = Run(cfg)

    written = first.to_config(tmp_path / "roundtrip.yaml")

    text = Path(written).read_text(encoding="utf-8")
    assert "scene_provider" not in text
    assert "scene_preset" in text
    reloaded = Run.from_config(written)
    assert reloaded.spec.scene_preset == first.spec.scene_preset
    assert reloaded.cfg == first.cfg


def test_content_only_preset_round_trips_through_serialization(tmp_path: Path) -> None:
    """A content-only preset keeps round-tripping once its declaration is re-emitted."""
    runtime = _runtime(tmp_path)
    first = build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args(), runtime=runtime)

    authored = serialize_scene_preset(first.cfg, first.scene_preset)
    # Mirror Run.to_config's surface spelling for the content block.
    authored["objects"] = authored.pop("props")

    assert authored["scene_preset"] == "acme_scene.tabletop"
    assert "scene_provider" not in authored["env"]["params"]
    second = build_spec(authored, _args(), runtime=runtime)
    assert second.cfg == first.cfg
    assert second.scene_preset == first.scene_preset


def test_unknown_preset_id_names_the_id_and_known_ids(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    with pytest.raises(PackError) as info:
        load_scene_preset("acme_scene.missing", runtime)

    assert "unknown id 'acme_scene.missing'" in str(info.value)
    assert "acme_scene.tabletop" in str(info.value)


def test_catalog_preset_without_runtime_is_a_teaching_error() -> None:
    with pytest.raises(ConfigError) as info:
        build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args())

    message = str(info.value)
    assert "'acme_scene.tabletop'" in message
    assert "requires a pack runtime" in message


def test_config_without_scene_preset_passes_through_without_a_runtime() -> None:
    cfg = {"run_name": "r", "env": {"backend": "mujoco"}}

    assert apply_scene_preset(cfg) is cfg


def test_legacy_run_scene_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="scene_preset"):
        reject_legacy_scene_keys({"scene": "acme.tabletop"})

    with pytest.raises(ConfigError, match="scene_preset"):
        reject_legacy_scene_keys({"env": {"params": {"mj_scene": "acme.tabletop"}}})


def test_content_fill_without_provider_is_rejected(tmp_path: Path) -> None:
    text = CONTENT_ONLY.replace(
        "objects:\n  types:\n    - name: scene_box\n",
        "objects:\n  types:\n    - name: scene_box\n  filler: tabletop_filler\n  sets: {cup: 1}\n",
    )
    runtime = _runtime(tmp_path, text)

    with pytest.raises(ScenePresetError, match="composed scene"):
        load_scene_preset("acme_scene.tabletop", runtime)


def test_initializer_name_shape_and_base_keys_are_rejected(tmp_path: Path) -> None:
    bad_initializer = CONTENT_ONLY.replace(
        "initializer: {type: tabletop_initializer, params: {seed: 1}}",
        "initializer: {name: tabletop_initializer, params: {seed: 1}}",
    )
    with pytest.raises(ScenePresetError, match=r"'type'.*'module'"):
        load_scene_preset("acme_scene.tabletop", _runtime(tmp_path, bad_initializer))

    mixed_initializer = CONTENT_ONLY.replace(
        "initializer: {type: tabletop_initializer, params: {seed: 1}}",
        "initializer: {type: tabletop_initializer, name: tabletop_initializer, params: {seed: 1}}",
    )
    with pytest.raises(ScenePresetError, match="removed key 'name'"):
        load_scene_preset("acme_scene.tabletop", _runtime(tmp_path, mixed_initializer))

    bad_base = CONTENT_ONLY.replace("base: {joint: fixed}", "base: {kind: scene}")
    with pytest.raises(ScenePresetError, match="unsupported key"):
        load_scene_preset("acme_scene.tabletop", _runtime(tmp_path, bad_base))


def test_world_from_config_rejects_a_direct_preset_file(tmp_path: Path) -> None:
    pack = _pack_root(tmp_path, CONTENT_ONLY)

    with pytest.raises(ValueError) as info:
        World.from_config(pack / "scene_presets" / "tabletop.yaml")

    assert "is a scene preset file" in str(info.value)
    assert "scene_presets" in str(info.value)


def test_world_from_config_and_from_preset_read_the_new_key(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    world = World.from_config({"scene_preset": "acme_scene.tabletop"}, runtime=runtime)

    assert world.fragment()["scene_preset"] == "acme_scene.tabletop"

    direct = World.from_preset("acme_scene.tabletop", runtime=runtime)
    assert direct.fragment()["scene_preset"] == "acme_scene.tabletop"


def test_world_from_config_mapping_without_runtime_fails() -> None:
    """A catalog ref in a mapping without a runtime fails naming the ref."""
    with pytest.raises(ValueError) as info:
        World.from_config({"scene_preset": "acme_scene.tabletop"})

    message = str(info.value)
    assert "'acme_scene.tabletop'" in message
    assert "requires a pack runtime" in message


def test_world_from_preset_requires_a_runtime(tmp_path: Path) -> None:
    """``World.from_preset`` takes the runtime as a required keyword."""
    runtime = _runtime(tmp_path)

    world = World.from_preset("acme_scene.tabletop", runtime=runtime)

    assert world.fragment()["scene_preset"] == "acme_scene.tabletop"
    with pytest.raises(TypeError):
        World.from_preset("acme_scene.tabletop")


def test_world_base_declares_each_format_neutral_asset() -> None:
    """``World.base`` emits any declared subset of ``mjcf`` / ``urdf`` / ``usd``."""
    assert World.base(mjcf="w.xml").fragment() == {"world_base": {"mjcf": "w.xml"}}
    assert World.base(urdf="w.urdf").fragment() == {"world_base": {"urdf": "w.urdf"}}
    assert World.base(usd="w.usd").fragment() == {"world_base": {"usd": "w.usd"}}
    assert World.base(urdf="w.urdf", usd="w.usd").fragment() == {
        "world_base": {"urdf": "w.urdf", "usd": "w.usd"}
    }


def test_world_base_requires_at_least_one_format() -> None:
    """A world base with no declared format is a teaching error."""
    with pytest.raises(ValueError, match="at least one of"):
        World.base()


def test_world_base_urdf_without_a_compatible_backend_fails(tmp_path: Path) -> None:
    text = CONTENT_ONLY.replace("  mjcf: world.xml", "  urdf: world.urdf")
    runtime = _runtime(tmp_path, text)

    with pytest.raises(ConfigError, match="no compatible world format"):
        build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args(), runtime=runtime)


def test_unknown_backend_params_block_names_the_known_backends(tmp_path: Path) -> None:
    text = CONTENT_ONLY.replace(
        "backend_params:\n", "backend_params:\n  nope:\n    base: {joint: fixed}\n"
    )
    runtime = _runtime(tmp_path, text)

    with pytest.raises(ConfigError, match="unknown backend 'nope'"):
        build_spec(_cfg(scene_preset="acme_scene.tabletop"), _args(), runtime=runtime)


def test_provider_preset_loads_every_declared_field(tmp_path: Path) -> None:
    """The loader maps every surface key of a provider-backed document."""
    runtime = _runtime(tmp_path, PROVIDER_BACKED)

    preset = load_scene_preset("acme_scene.tabletop", runtime)

    assert preset.provider == "acme.table_top"
    assert dict(preset.params) == {"style": 1}
    assert preset.world is not None and preset.world.mjcf == WORLD
    assert preset.objects == {
        "types": [{"name": "scene_box"}],
        "filler": "tabletop_filler",
        "sets": {"cup": 1},
        "seed": 1,
    }
    assert dict(preset.backend_params["mujoco"]) == {"base": {"joint": "fixed"}}


def test_content_only_fixture_preset_builds_without_activation() -> None:
    runtime = discover([PACKS_ROOT / "acme_content"])

    spec = build_spec(
        _cfg(scene_preset="acme_content.tabletop"),
        _args(),
        runtime=runtime,
    )

    assert spec.cfg["world_base"]["mjcf"].endswith("acme_content/worlds/table.xml")
    assert spec.props == {"types": [{"name": "table_plate"}]}
    assert "scene_provider" not in spec.backend_params


def test_catalog_preset_provider_must_be_declared_by_the_same_pack(tmp_path: Path) -> None:
    """A dynamically registered provider is not referenceable from a preset."""
    text = PROVIDER_BACKED.replace("provider: acme.table_top", "provider: acme.other")
    runtime = _runtime(tmp_path, text)

    with pytest.raises(ScenePresetError, match="does not declare as a kind: scene provider"):
        load_scene_preset("acme_scene.tabletop", runtime)


def test_catalog_preset_provider_owned_by_another_pack_names_both(tmp_path: Path) -> None:
    """A cross-pack provider reference names the preset pack and the owner."""
    preset_pack = tmp_path / "preset_pack"
    (preset_pack / "scene_presets").mkdir(parents=True)
    (preset_pack / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        "name: preset_pack\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides:\n"
        "  scene_presets:\n"
        "    - id: preset_pack.tabletop\n"
        "      resource: scene_presets/tabletop.yaml\n",
        encoding="utf-8",
    )
    (preset_pack / "scene_presets" / "tabletop.yaml").write_text(PROVIDER_BACKED, encoding="utf-8")
    other_pack = tmp_path / "other_pack"
    other_pack.mkdir()
    (other_pack / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        "name: other_pack\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: code\n"
        "provides:\n"
        "  providers:\n"
        "    - id: acme.table_top\n"
        "      kind: scene\n"
        "      entrypoint: acme_scenes:table_top\n",
        encoding="utf-8",
    )
    runtime = discover([preset_pack, other_pack])

    with pytest.raises(ScenePresetError) as info:
        load_scene_preset("preset_pack.tabletop", runtime)

    message = str(info.value)
    assert "preset_pack" in message
    assert "other_pack" in message
    assert "acme.table_top" in message


def test_catalog_preset_params_are_validated_against_the_manifest_schema(
    tmp_path: Path,
) -> None:
    """Declared preset values are checked for type and unknown keys at load."""
    bad_type = PROVIDER_BACKED.replace("  style: 1", '  style: "one"')
    with pytest.raises(ScenePresetError, match="must be int"):
        load_scene_preset("acme_scene.tabletop", _runtime(tmp_path, bad_type))

    unknown = PROVIDER_BACKED.replace("  style: 1", "  style: 1\n  extra: 2")
    with pytest.raises(ScenePresetError, match="unknown parameter"):
        load_scene_preset("acme_scene.tabletop", _runtime(tmp_path, unknown))


def test_catalog_preset_may_defer_a_required_param_to_a_run_override(tmp_path: Path) -> None:
    """Load validates declared values but a required parameter may come later."""
    runtime = _runtime(tmp_path, PROVIDER_BACKED)

    preset = load_scene_preset("acme_scene.tabletop", runtime)

    assert dict(preset.params) == {"style": 1}
