"""Contract for the boundary-resolved asset plan.

The plan is the one asset boundary of Phase 7b-2: pack-qualified mesh and
texture declarations become absolute before any backend sees them, documents
resolve once, and removed anchors or missing contexts fail with teaching
errors.  The backend renderer's absolute-path guard is exercised directly; no
real simulator is started.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from unigym.assets.plan import AssetPlan
from unigym.core.paths import PlaceholderError, expand_placeholders
from unigym.pack_api import AssetContext, PackRuntime, discover
from unigym.props.asset_ir import AssetDefinition, Body, MeshDef, TextureDef
from unigym.run.spec import build_spec

PACK = "acme_assets"
MESH_REF = f"pack:{PACK}/assets/meshes/cup.stl"
TEXTURE_REF = f"pack:{PACK}/assets/generators/tex/a/wood.png"
CANONICAL_TEXTURE = "assets/generators/tex/a/wood/wood.png"


def _write_pack(tmp_path: Path) -> Path:
    """Write a minimal content pack with one mesh and one canonical texture."""
    root = tmp_path / PACK
    root.mkdir()
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": PACK,
                "version": "1.0.0",
                "api": 1,
                "trust": "content",
                "provides": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    mesh = root / "assets" / "meshes" / "cup.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("solid cup\n", encoding="utf-8")
    texture = root / CANONICAL_TEXTURE
    texture.parent.mkdir(parents=True)
    texture.write_text("png\n", encoding="utf-8")
    return root


@pytest.fixture()
def runtime(tmp_path: Path) -> PackRuntime:
    return discover([_write_pack(tmp_path)])


@pytest.fixture()
def plan(runtime: PackRuntime) -> AssetPlan:
    return AssetPlan(AssetContext(runtime))


def _asset() -> AssetDefinition:
    return AssetDefinition(
        name="cup",
        root=Body(name="cup"),
        meshes=(MeshDef(name="cup_mesh", file=MESH_REF, scale=(1.0, 2.0, 3.0)),),
        textures=(TextureDef(name="wood", kind="2d", file=TEXTURE_REF),),
    )


def test_asset_plan_resolves_pack_mesh_and_texture_to_absolute(
    plan: AssetPlan, runtime: PackRuntime
) -> None:
    """A pack-qualified mesh and texture resolve to absolute package paths."""
    resolved = plan.resolve_definition(_asset())

    assert resolved.meshes[0].file == str(runtime.root_by_pack[PACK] / "assets/meshes/cup.stl")
    assert resolved.textures[0].file == str(runtime.resource(PACK, CANONICAL_TEXTURE))


def test_asset_plan_preserves_every_other_definition_field(plan: AssetPlan) -> None:
    """Resolving declaration files changes only the file strings."""
    original = _asset()

    resolved = plan.resolve_definition(original)

    assert resolved.name == original.name
    assert resolved.root == original.root
    assert resolved.meshes[0].name == "cup_mesh"
    assert resolved.meshes[0].scale == (1.0, 2.0, 3.0)
    assert resolved.textures[0].name == "wood"
    assert resolved.textures[0].kind == "2d"


def test_asset_plan_resolves_document_placeholders_and_pack_leaves(
    plan: AssetPlan, runtime: PackRuntime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document expands its token and resolves its ``pack:`` file leaf."""
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(tmp_path))
    document = (
        '<mujoco model="probe"><asset>'
        '<mesh file="pack:' + PACK + '/assets/meshes/cup.stl"/>'
        '<texture name="t" type="2d" file="${UNIGYM_DEPS_ROOT}/own.png"/>'
        "</asset></mujoco>"
    )

    resolved = plan.resolve_document(document)

    assert str(runtime.root_by_pack[PACK] / "assets/meshes/cup.stl") in resolved
    assert str(tmp_path / "own.png") in resolved
    assert "pack:" not in resolved
    assert "${" not in resolved


def test_pack_reference_without_context_is_a_teaching_error() -> None:
    """A pack reference cannot resolve without an explicit asset context."""
    with pytest.raises(PlaceholderError, match="explicit asset context"):
        expand_placeholders(MESH_REF)


@pytest.mark.parametrize("token", ["UNIGYM_ASSET_ROOT", "UNIGYM_ASSET_PACK"])
def test_removed_asset_tokens_are_rejected(token: str, plan: AssetPlan) -> None:
    """The removed asset anchors are a teaching error, even with a context."""
    with pytest.raises(PlaceholderError, match="was removed by the asset cutover"):
        expand_placeholders("${" + token + "}/generators/x.png", plan.context)


@pytest.mark.parametrize("marker", ["${UNIGYM_REPO}", "pack:" + PACK + "/assets/meshes/cup.stl"])
def test_load_spec_rejects_an_unresolved_document(marker: str, tmp_path: Path) -> None:
    """The MJCF loader rejects a document that skipped the boundary."""
    from unigym.backends.shared.mjfamily.scene.mjcf_io import load_spec

    document = tmp_path / "fragment.xml"
    document.write_text(
        '<mujoco model="probe"><asset><mesh file="' + marker + '"/></asset></mujoco>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved asset reference"):
        load_spec(document)


def _run_config(asset: str) -> dict:
    return {
        "run_name": "r",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "controls": [
            {
                "name": "arm",
                "adapter": "manual",
                "articulation": "arm",
                "joints": ["shoulder"],
                "drive": "position_pd",
                "target": "position",
            }
        ],
        "objects": {"types": [{"name": "cup", "type": "articulation", "asset": asset}]},
    }


def test_run_config_pack_reference_without_runtime_is_a_teaching_error(tmp_path: Path) -> None:
    """A run config carrying a pack reference needs an explicit runtime."""
    args = Namespace(config=str(tmp_path / "run.yaml"))

    with pytest.raises(PlaceholderError, match="explicit asset context"):
        build_spec(_run_config(MESH_REF), args)


def test_run_config_pack_reference_expands_with_a_runtime(
    tmp_path: Path, runtime: PackRuntime
) -> None:
    """With a runtime, the run config's pack reference expands to an absolute path."""
    args = Namespace(config=str(tmp_path / "run.yaml"))

    spec = build_spec(_run_config(MESH_REF), args, runtime=runtime)

    assert spec.props["types"][0]["asset"] == str(
        runtime.root_by_pack[PACK] / "assets/meshes/cup.stl"
    )


TEMPLATE_NAME = "phase7b2_pack_prop"


@pytest.fixture()
def pack_template():
    """Register one generated-prop template and remove it afterwards.

    The template registry is a process-global closed registry without a
    public unregister operation.  Restoring its private module tables in the
    fixture teardown is the only way to keep this test from leaking the
    template into later tests; the registry is touched nowhere else.
    """
    from unigym.props.templates import (
        _TEMPLATE_METADATA,
        _TEMPLATES,
        PropTemplateResult,
        register_prop_template,
    )

    def provider(prop_name, params):
        """Return a generated payload carrying pack-qualified declaration files."""
        return PropTemplateResult(
            spec={"name": prop_name, "type": "free_articulation", "joints": {}},
            asset=AssetDefinition(
                name=prop_name,
                root=Body(name=prop_name),
                meshes=(MeshDef(name="cup_mesh", file=MESH_REF),),
                textures=(TextureDef(name="wood", kind="2d", file=TEXTURE_REF),),
            ),
        )

    register_prop_template(
        TEMPLATE_NAME,
        params={"variant": {"type": "enum", "values": ["plain"], "required": False}},
    )(provider)
    try:
        yield TEMPLATE_NAME
    finally:
        _TEMPLATES.pop(TEMPLATE_NAME, None)
        _TEMPLATE_METADATA.pop(TEMPLATE_NAME, None)


def _template_config(name: str) -> dict:
    return {"types": [{"name": "cup", "template": name, "params": {}}]}


def test_prop_template_resolves_generated_asset_files(
    pack_template: str, plan: AssetPlan, runtime: PackRuntime
) -> None:
    """A template's pack-qualified mesh and texture become absolute at the boundary."""
    from unigym.props.templates import resolve_prop_entries

    entries = resolve_prop_entries(_template_config(pack_template), plan)

    assert len(entries) == 1
    _entry, asset = entries[0]
    assert asset is not None
    assert Path(asset.meshes[0].file).is_absolute()
    assert asset.meshes[0].file == str(runtime.root_by_pack[PACK] / "assets/meshes/cup.stl")
    assert asset.textures[0].file == str(runtime.resource(PACK, CANONICAL_TEXTURE))


def test_prop_manager_stores_absolute_generated_asset_files(
    pack_template: str, plan: AssetPlan, runtime: PackRuntime, tmp_path: Path
) -> None:
    """PropManager.asset_definitions carries the boundary-resolved files."""
    from unigym.props import build_prop_manager

    manager = build_prop_manager(
        _template_config(pack_template),
        num_envs=1,
        env_spacing=2.0,
        device="cpu",
        base_dir=str(tmp_path),
        backend="mujoco",
        asset_plan=plan,
    )

    assert manager is not None
    asset = manager.asset_definitions["cup"]
    assert asset.meshes[0].file == str(runtime.root_by_pack[PACK] / "assets/meshes/cup.stl")
    assert asset.textures[0].file == str(runtime.resource(PACK, CANONICAL_TEXTURE))


def test_prop_template_generated_asset_requires_a_plan(pack_template: str) -> None:
    """A generated payload without an asset plan is a teaching error."""
    from unigym.props.templates import resolve_prop_entries

    with pytest.raises(ValueError, match="requires an asset plan"):
        resolve_prop_entries(_template_config(pack_template))


@pytest.mark.parametrize(
    "value",
    ["assets/meshes/cup.stl", MESH_REF, "${UNIGYM_REPO}/assets/meshes/cup.stl"],
    ids=["relative", "pack", "placeholder"],
)
def test_render_boundary_rejects_unresolved_asset_file(value: str) -> None:
    """The backend render boundary refuses any file the plan did not resolve."""
    from unigym.backends.shared.mjfamily.scene.asset_render import _absolute_asset_file

    with pytest.raises(ValueError, match="is not absolute"):
        _absolute_asset_file("mesh", "cup_mesh", value)


def test_render_boundary_accepts_an_absolute_asset_file(tmp_path: Path) -> None:
    """The backend render boundary accepts the plan's absolute path verbatim."""
    from unigym.backends.shared.mjfamily.scene.asset_render import _absolute_asset_file

    absolute = str(tmp_path / "assets" / "meshes" / "cup.stl")

    assert _absolute_asset_file("mesh", "cup_mesh", absolute) == absolute
