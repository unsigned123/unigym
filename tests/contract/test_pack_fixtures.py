"""Contract for the shared Phase 2 fixture packs.

The fixtures under ``tests/_support/packs`` are real ``unigym.pack/1``
documents, not full domain schemas.  These tests pin their exact shape and
prove that manifest and catalog loading never import fixture pack Python.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

from unigym.pack_api import (
    Face,
    ItemStatus,
    Pack,
    PackSet,
    load_manifest_v1,
)

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"
FIXTURE_ROOTS = (CONTENT_ROOT, CODE_ROOT)

EXPECTED_YAML_FILES = {
    "acme_code/unigym.pack.yaml",
    "acme_content/assets/cup/asset.yaml",
    "acme_content/runs/demo.yaml",
    "acme_content/scene_presets/tabletop.yaml",
    "acme_content/unigym.pack.yaml",
    "facade_mjcf/robots/tiny_arm.yaml",
    "facade_mjcf/unigym.pack.yaml",
}


def _yaml_documents() -> list[Path]:
    """Return every fixture YAML document in a stable order."""
    return sorted(path for path in PACKS_ROOT.rglob("*.yaml") if "__pycache__" not in path.parts)


def _string_values(node: object) -> list[str]:
    """Collect every string value and key from a parsed YAML tree."""
    values: list[str] = []
    if isinstance(node, str):
        values.append(node)
    elif isinstance(node, dict):
        for key, value in node.items():
            values.extend(_string_values(key))
            values.extend(_string_values(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(_string_values(value))
    return values


def test_content_fixture_loads_with_exact_items() -> None:
    manifest = load_manifest_v1(CONTENT_ROOT)

    assert isinstance(manifest, Pack)
    assert manifest.schema == "unigym.pack/1"
    assert manifest.name == "acme_content"
    assert manifest.version == "1.0.0"
    assert manifest.api == 1
    assert manifest.trust == "content"
    assert manifest.entrypoint is None
    assert dict(manifest.requires) == {}
    assert manifest.root == CONTENT_ROOT.resolve()
    assert [
        (item.face, item.name, item.kind, item.resource, item.entrypoint, item.params)
        for item in manifest.items
    ] == [
        (Face.RUNS, "acme_content.demo", None, "runs/demo.yaml", None, None),
        (
            Face.SCENE_PRESETS,
            "acme_content.tabletop",
            None,
            "scene_presets/tabletop.yaml",
            None,
            None,
        ),
        (Face.ASSETS, "acme_content.cup", None, "assets/cup/asset.yaml", None, None),
    ]
    assert all(item.status is ItemStatus.DECLARED for item in manifest.items)


def test_code_fixture_loads_with_trust_and_provider_entrypoint() -> None:
    manifest = load_manifest_v1(CODE_ROOT)

    assert manifest.schema == "unigym.pack/1"
    assert manifest.name == "acme_code"
    assert manifest.version == "1.0.0"
    assert manifest.api == 1
    assert manifest.trust == "code"
    assert manifest.entrypoint is not None
    assert manifest.entrypoint.raw == "acme_pack:register"
    assert manifest.root == CODE_ROOT.resolve()
    assert [
        (item.face, item.name, item.kind, item.resource, item.entrypoint, item.params)
        for item in manifest.items
    ] == [
        (
            Face.PROVIDERS,
            "acme_code.dynamic_assets",
            "asset_generator",
            None,
            "acme_pack:register",
            None,
        ),
        (
            Face.PROVIDERS,
            "acme_code.table_top",
            "scene",
            None,
            "acme_pack:table_top",
            {
                "style": {"type": "int", "default": 1},
                "seed": {"type": "int"},
            },
        ),
    ]


def test_pack_set_combines_fixtures_in_order_and_filters_by_face() -> None:
    content = load_manifest_v1(CONTENT_ROOT)
    code = load_manifest_v1(CODE_ROOT)

    pack_set = PackSet((content, code))

    assert [pack.name for pack in pack_set.packs()] == ["acme_content", "acme_code"]
    assert pack_set.get("acme_content") is content
    assert pack_set.get("acme_code") is code
    assert [
        (item.pack, item.face, item.name, item.kind, item.resource, item.entrypoint)
        for item in pack_set.items()
    ] == [
        ("acme_content", Face.RUNS, "acme_content.demo", None, "runs/demo.yaml", None),
        (
            "acme_content",
            Face.SCENE_PRESETS,
            "acme_content.tabletop",
            None,
            "scene_presets/tabletop.yaml",
            None,
        ),
        ("acme_content", Face.ASSETS, "acme_content.cup", None, "assets/cup/asset.yaml", None),
        (
            "acme_code",
            Face.PROVIDERS,
            "acme_code.dynamic_assets",
            "asset_generator",
            None,
            "acme_pack:register",
        ),
        (
            "acme_code",
            Face.PROVIDERS,
            "acme_code.table_top",
            "scene",
            None,
            "acme_pack:table_top",
        ),
    ]
    assert pack_set.items(face=Face.SCENE_PRESETS) == (
        content.item(Face.SCENE_PRESETS, "acme_content.tabletop"),
    )
    assert [item.name for item in pack_set.items(face=Face.PROVIDERS)] == [
        "acme_code.dynamic_assets",
        "acme_code.table_top",
    ]
    assert pack_set.items(face=Face.TERMINATIONS) == ()


def test_fixture_items_are_unique_by_face_and_name() -> None:
    pack_set = PackSet((load_manifest_v1(CONTENT_ROOT), load_manifest_v1(CODE_ROOT)))

    keys = [(item.face, item.name) for item in pack_set.items()]

    assert len(keys) == len(set(keys)) == 5


def test_manifest_and_catalog_loading_does_not_import_pack_python() -> None:
    module_file = CODE_ROOT / "python" / "acme_pack" / "__init__.py"
    assert module_file.is_file()
    assert "acme_pack" not in sys.modules

    content = load_manifest_v1(CONTENT_ROOT)
    code = load_manifest_v1(CODE_ROOT)
    pack_set = PackSet((content, code))

    entrypoint = pack_set.get("acme_code").entrypoint
    assert entrypoint is not None and entrypoint.raw == "acme_pack:register"
    assert "acme_pack" not in sys.modules


def test_code_fixture_python_dir_is_not_on_import_path() -> None:
    python_dir = (CODE_ROOT / "python").resolve()
    import_roots = {Path(entry or ".").resolve() for entry in sys.path}

    assert python_dir not in import_roots


def test_fixture_yaml_contains_no_absolute_or_parent_paths() -> None:
    documents = _yaml_documents()

    assert {path.relative_to(PACKS_ROOT).as_posix() for path in documents} == EXPECTED_YAML_FILES

    offenses: list[str] = []
    for path in documents:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for value in _string_values(document):
            posix = PurePosixPath(value)
            if posix.is_absolute() or value.startswith("~") or ".." in posix.parts:
                offenses.append(f"{path.relative_to(PACKS_ROOT)}: {value!r}")

    assert not offenses, f"absolute or parent-relative fixture paths: {offenses}"


def test_fixture_resources_resolve_under_their_pack_roots() -> None:
    for root in FIXTURE_ROOTS:
        manifest = load_manifest_v1(root)
        for item in manifest.items:
            if item.resource is None:
                continue
            resource = root / item.resource
            assert resource.is_file()
            assert resource.resolve().is_relative_to(root.resolve())


def test_fixtures_live_under_support_and_are_not_collected_as_tests(
    pytestconfig: pytest.Config,
) -> None:
    assert PACKS_ROOT == Path(__file__).resolve().parents[1] / "_support" / "packs"
    assert PACKS_ROOT.is_dir()
    assert PACKS_ROOT.parent.name == "_support"

    fixture_files = sorted(path for path in PACKS_ROOT.rglob("*") if path.is_file())
    assert fixture_files
    assert all(path.is_relative_to(PACKS_ROOT) for path in fixture_files)
    assert not [
        path
        for path in PACKS_ROOT.rglob("*")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    ]

    collected_roots = [
        (pytestconfig.rootpath / selector).resolve()
        for selector in pytestconfig.getini("testpaths")
    ]
    for path in fixture_files:
        resolved = path.resolve()
        assert not any(
            collected == resolved or collected in resolved.parents for collected in collected_roots
        ), f"fixture {path} lies under a collected test path"
