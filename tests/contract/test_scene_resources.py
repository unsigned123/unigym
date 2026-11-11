"""Contract for scene resolution through the runtime ``scenes`` face.

A run config's ``scene:`` value is an exact catalog id resolved by
``PackRuntime.resource_for("scenes", id)``.  The legacy family/name directory
scan and direct ``scene.yaml`` references are gone, so a ref without a runtime
and an unknown id both fail with teaching errors.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from unigym.pack_api import discover
from unigym.packs import PackError
from unigym.run.spec import build_spec
from unigym.scene_packs import apply_scene_bundle, load_scene_bundle
from unigym.world import World

MANIFEST = """\
schema: unigym.pack/1
name: acme_scene
version: 1.0.0
api: 1
trust: content
provides:
  scenes:
    - id: acme_scene.tabletop
      resource: scenes/tabletop.yaml
"""

SCENE = """\
name: tabletop
backend:
  mujoco:
    world_base:
      mjcf: scene.xml
    mj_scene: table_top
    mj_scene_params:
      x: 1
    base:
      kind: scene
    actuators:
      legs:
        drive: position_pd
        kp: 100
objects:
  types:
    - name: scene_box
initializer:
  name: scene_initializer
  params:
    mode: scene
    seed: 1
"""


def _pack_root(tmp_path: Path, scene_text: str = SCENE) -> Path:
    """Write one content pack with a scene resource and return its root."""
    pack = tmp_path / "acme_scene"
    (pack / "scenes").mkdir(parents=True)
    (pack / "unigym.pack.yaml").write_text(MANIFEST, encoding="utf-8")
    (pack / "scenes" / "tabletop.yaml").write_text(scene_text, encoding="utf-8")
    return pack


def _runtime(tmp_path: Path, scene_text: str = SCENE):
    """Discover a runtime over the temporary pack."""
    return discover([_pack_root(tmp_path, scene_text)])


def test_scene_id_resolves_through_the_runtime_and_parses(tmp_path: Path) -> None:
    """An exact catalog id resolves to its declared, parsed YAML document."""
    runtime = _runtime(tmp_path)

    bundle = load_scene_bundle("acme_scene.tabletop", runtime)

    assert bundle["name"] == "tabletop"
    assert bundle["backend"]["mujoco"]["mj_scene"] == "table_top"
    assert runtime.resource_for("scenes", "acme_scene.tabletop").name == "tabletop.yaml"


def test_scene_bundle_routes_blocks_and_run_values_win(tmp_path: Path) -> None:
    """Backend blocks route as before and run overrides merge over scene defaults."""
    runtime = _runtime(tmp_path)
    cfg = {
        "scene": "acme_scene.tabletop",
        "world_base": {"mjcf": "run.xml"},
        "env": {
            "params": {
                "actuators": {"legs": {"kp": 200}},
                "initializer": {"params": {"seed": 9}},
            }
        },
        "objects": {"types": [{"name": "run_box"}]},
    }

    merged = apply_scene_bundle(cfg, runtime)

    params = merged["env"]["params"]
    assert merged["world_base"] == {"mjcf": "run.xml"}
    assert params["mj_scene"] == "table_top"
    assert params["mj_scene_params"] == {"x": 1}
    assert params["base"] == {"kind": "scene"}
    assert params["actuators"]["legs"] == {"drive": "position_pd", "kp": 200}
    assert params["initializer"] == {
        "name": "scene_initializer",
        "params": {"mode": "scene", "seed": 9},
    }
    assert merged["props"] == {"types": [{"name": "run_box"}]}


def test_build_spec_consumes_the_scene_ref_and_routes_the_bundle(tmp_path: Path) -> None:
    """A scene-bearing run builds successfully and leaves no string ref behind."""
    runtime = _runtime(tmp_path)
    cfg = {
        "run_name": "scene_run",
        "seed": 1,
        "scene": "acme_scene.tabletop",
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

    spec = build_spec(cfg, Namespace(config="run.yaml"), runtime=runtime)

    assert "scene" not in spec.cfg
    assert spec.world_base == {"mjcf": "scene.xml"}
    assert spec.backend_params["mj_scene"] == "table_top"
    assert spec.backend_params["initializer"]["name"] == "scene_initializer"


def test_unknown_scene_id_names_the_id_and_known_ids(tmp_path: Path) -> None:
    """An unknown catalog id fails with the runtime's known-id list."""
    runtime = _runtime(tmp_path)

    with pytest.raises(PackError) as info:
        apply_scene_bundle({"scene": "acme_scene.missing"}, runtime)

    message = str(info.value)
    assert "unknown id 'acme_scene.missing'" in message
    assert "acme_scene.tabletop" in message


def test_scene_ref_without_runtime_is_a_teaching_error() -> None:
    """A scene ref without a runtime names the ref instead of falling back."""
    with pytest.raises(ValueError) as info:
        apply_scene_bundle({"scene": "acme_scene.tabletop"})

    message = str(info.value)
    assert "'acme_scene.tabletop'" in message
    assert "requires a pack runtime" in message


@pytest.mark.parametrize(
    ("scene_text", "match"),
    [
        ("key: [unclosed\n", "not valid YAML"),
        ("- one\n- two\n", "must be a mapping"),
    ],
    ids=["malformed-yaml", "non-mapping"],
)
def test_scene_resource_must_parse_as_a_mapping(
    tmp_path: Path, scene_text: str, match: str
) -> None:
    """A malformed or non-mapping scene resource fails at resolution."""
    runtime = _runtime(tmp_path, scene_text)

    with pytest.raises(ValueError, match=match):
        load_scene_bundle("acme_scene.tabletop", runtime)


def test_legacy_family_name_ref_does_not_resolve(tmp_path: Path) -> None:
    """The removed ``<family>/<name>`` form is an unknown catalog id, not a scan."""
    runtime = _runtime(tmp_path)

    with pytest.raises(PackError) as info:
        apply_scene_bundle({"scene": "tabletop/tabletop"}, runtime)

    message = str(info.value)
    assert "unknown id 'tabletop/tabletop'" in message
    assert "acme_scene.tabletop" in message


def test_world_from_config_rejects_a_direct_scene_bundle_file(tmp_path: Path) -> None:
    """A direct ``scene.yaml`` bundle file is no longer a scene reference."""
    pack = _pack_root(tmp_path)

    with pytest.raises(ValueError) as info:
        World.from_config(pack / "scenes" / "tabletop.yaml")

    message = str(info.value)
    assert "is a scene bundle file" in message
    assert "'scenes' catalog ids" in message


def test_world_from_config_resolves_a_scene_mapping_with_a_runtime(tmp_path: Path) -> None:
    """A scene-bearing run mapping resolves when a runtime is supplied."""
    runtime = _runtime(tmp_path)

    world = World.from_config({"scene": "acme_scene.tabletop"}, runtime=runtime)

    assert world.fragment() == {"scene": "acme_scene.tabletop"}


def test_world_from_config_scene_mapping_without_runtime_fails() -> None:
    """A scene-bearing mapping without a runtime fails naming the ref."""
    with pytest.raises(ValueError) as info:
        World.from_config({"scene": "acme_scene.tabletop"})

    message = str(info.value)
    assert "'acme_scene.tabletop'" in message
    assert "requires a pack runtime" in message


def test_world_from_pack_requires_a_runtime(tmp_path: Path) -> None:
    """``World.from_pack`` takes the runtime as a required keyword."""
    runtime = _runtime(tmp_path)

    world = World.from_pack("acme_scene.tabletop", runtime=runtime)

    assert world.fragment() == {"scene": "acme_scene.tabletop"}
    with pytest.raises(TypeError):
        World.from_pack("acme_scene.tabletop")


def test_config_without_scene_passes_through_without_a_runtime() -> None:
    """A config with no ``scene:`` key is returned unchanged and needs no runtime."""
    cfg = {"run_name": "r", "env": {"backend": "mujoco"}}

    result = apply_scene_bundle(cfg)

    assert result is cfg
    assert result == {"run_name": "r", "env": {"backend": "mujoco"}}
