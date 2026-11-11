"""Contract for the Phase 7b-1 asset-resolver foundation.

A temporary ``unigym.pack/1`` content pack proves pack-qualified mesh and
texture resolution, absolute pass-through, teaching errors and the delegating
``AssetContext``.  No production consumer is switched here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from unigym.pack_api import AssetContext, PackRuntime, discover
from unigym.packs import PackError

PACK = "acme_assets"
MESH_REF = f"pack:{PACK}/assets/meshes/cup.stl"
TEXTURE_REF = f"pack:{PACK}/assets/generators/tex/a/wood.png"
CANONICAL_TEXTURE = "assets/generators/tex/a/wood/wood.png"


def _write_pack(
    tmp_path: Path,
    *,
    trust: str = "content",
    entrypoint: str | None = None,
) -> Path:
    """Write a minimal content pack with one mesh and one canonical texture."""
    root = tmp_path / PACK
    root.mkdir()
    document: dict[str, object] = {
        "schema": "unigym.pack/1",
        "name": PACK,
        "version": "1.0.0",
        "api": 1,
        "trust": trust,
        "provides": {},
    }
    if entrypoint is not None:
        document["entrypoint"] = entrypoint
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
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


def test_pack_qualified_mesh_resolves_under_the_pack_root(runtime: PackRuntime) -> None:
    """A pack-qualified mesh resolves inside its owning pack root."""
    expected = runtime.root_by_pack[PACK] / "assets" / "meshes" / "cup.stl"

    assert runtime.asset_path(MESH_REF) == expected


def test_pack_qualified_texture_leaf_is_canonicalized(runtime: PackRuntime) -> None:
    """A declared texture leaf resolves to its own canonical asset directory."""
    expected = runtime.resource(PACK, CANONICAL_TEXTURE)

    resolved = runtime.asset_path(TEXTURE_REF)

    assert resolved == expected
    assert resolved.name == "wood.png"
    assert resolved.parent.name == "wood"


def test_absolute_asset_path_passes_through_without_existence(
    runtime: PackRuntime, tmp_path: Path
) -> None:
    """An absolute path is a run-owned asset and needs no pack or existence."""
    own = tmp_path / "run" / "own_texture.png"

    assert runtime.asset_path(str(own)) == own
    assert not own.exists()


def test_home_relative_asset_path_expands_to_an_absolute_path(
    runtime: PackRuntime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``~`` path is expanded to an absolute path without a pack lookup."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert runtime.asset_path("~/textures/own.png") == tmp_path / "textures" / "own.png"


@pytest.mark.parametrize("value", ["assets/meshes/cup.stl", "assets/generators/tex/a/wood.png"])
def test_bare_relative_asset_path_is_rejected(runtime: PackRuntime, value: str) -> None:
    """A bare relative reference has no owning pack and is rejected."""
    with pytest.raises(PackError) as info:
        runtime.asset_path(value)

    message = str(info.value)
    assert value in message
    assert "pack-qualified or absolute" in message
    assert "pack:<pack>/<rel>" in message


@pytest.mark.parametrize("value", ["", None, 0, Path("assets/meshes/cup.stl")])
def test_non_string_or_empty_asset_reference_is_rejected(
    runtime: PackRuntime, value: object
) -> None:
    """Only a non-empty string is a valid asset reference."""
    with pytest.raises(PackError, match="non-empty string"):
        runtime.asset_path(value)


def test_missing_resource_names_the_pack_and_reference(runtime: PackRuntime) -> None:
    """An absent pack resource fails naming the pack and the original ref."""
    ref = f"pack:{PACK}/assets/meshes/absent.stl"

    with pytest.raises(PackError) as info:
        runtime.asset_path(ref)

    message = str(info.value)
    assert PACK in message
    assert ref in message
    assert "does not exist" in message


def test_unknown_pack_names_the_pack_and_reference(runtime: PackRuntime) -> None:
    """An unknown pack in an asset reference fails naming pack and ref."""
    ref = "pack:absent_pack/assets/meshes/cup.stl"

    with pytest.raises(PackError) as info:
        runtime.asset_path(ref)

    message = str(info.value)
    assert "absent_pack" in message
    assert ref in message


def test_asset_context_delegates_to_runtime_asset_path(runtime: PackRuntime) -> None:
    """``AssetContext.resolve`` returns the same path as ``runtime.asset_path``."""
    context = AssetContext(runtime=runtime)

    assert context.resolve(TEXTURE_REF) == str(runtime.asset_path(TEXTURE_REF))


def test_asset_resolution_does_not_import_pack_python(tmp_path: Path) -> None:
    """Resolving an asset never activates or imports the owning pack's code."""
    root = _write_pack(tmp_path, trust="code", entrypoint="acme_asset_sentinel:register")
    python_dir = root / "python"
    python_dir.mkdir()
    (python_dir / "acme_asset_sentinel.py").write_text(
        "raise AssertionError('pack python imported during asset resolution')\n",
        encoding="utf-8",
    )
    sys.modules.pop("acme_asset_sentinel", None)

    runtime = discover([root])
    runtime.asset_path(MESH_REF)

    assert "acme_asset_sentinel" not in sys.modules
