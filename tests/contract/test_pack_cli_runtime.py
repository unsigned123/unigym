"""Contract for the runtime-backed ``unigym pack`` CLI.

The CLI reads the same manifests, resources and activation path as the rest of
the system: there is no repo pack, no mount table and no legacy helper.  The
tests use real ``unigym.pack/1`` documents on ``tmp_path`` and drive the real
argparse command group.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

import unigym
from unigym.packs import PackError
from unigym.packs.cli import add_pack_subparser
from unigym.packs.schema import MANIFEST_NAME

_CODE_MODULE = "cli_code_pack"

_DELETED_MODULES = (
    "unigym.packs.load",
    "unigym.packs.manifest",
    "unigym.packs.mount",
    "unigym.packs.resolve",
)
_DELETED_IDENTIFIERS = (
    "load_mounted_code",
    "pack_roots",
    "repo_pack",
    "code_pack_root",
    "load_table",
)


@pytest.fixture(autouse=True)
def _isolate_runtime(monkeypatch: pytest.MonkeyPatch):
    """Clear ambient discovery and restore sys.path/modules after a code pack."""
    monkeypatch.delenv("UNIGYM_PACKS", raising=False)
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    path_snapshot = list(sys.path)
    yield
    sys.path[:] = path_snapshot
    for name in [
        name for name in sys.modules if name == _CODE_MODULE or name.startswith(_CODE_MODULE + ".")
    ]:
        del sys.modules[name]


def _content_pack(root: Path) -> Path:
    """Write a content pack with one resource on two faces."""
    (root / "runs").mkdir(parents=True)
    (root / "scene_presets").mkdir()
    (root / "runs" / "demo.yaml").write_text("run: demo\n", encoding="utf-8")
    (root / "scene_presets" / "tabletop.yaml").write_text(
        "scene preset: tabletop\n", encoding="utf-8"
    )
    (root / MANIFEST_NAME).write_text(
        "schema: unigym.pack/1\n"
        "name: cli_content\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides:\n"
        "  runs:\n"
        "    - id: cli_content.demo\n"
        "      resource: runs/demo.yaml\n"
        "  scene_presets:\n"
        "    - id: cli_content.tabletop\n"
        "      resource: scene_presets/tabletop.yaml\n",
        encoding="utf-8",
    )
    return root


def _code_pack(root: Path) -> Path:
    """Write a code pack whose activation records a marker beside its root."""
    module = root / "python" / _CODE_MODULE / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        '"""CLI fixture pack; activation writes one marker file."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from unigym.pack_api import PackContext\n"
        "\n"
        "\n"
        "def register(context: PackContext) -> None:\n"
        '    """Record that the runtime activated this pack."""\n'
        '    (context.root / "activated.txt").write_text("active", encoding="utf-8")\n',
        encoding="utf-8",
    )
    (root / MANIFEST_NAME).write_text(
        "schema: unigym.pack/1\n"
        "name: cli_code\n"
        "version: 2.1.0\n"
        "api: 1\n"
        "trust: code\n"
        f"entrypoint: {_CODE_MODULE}:register\n"
        "provides:\n"
        "  providers:\n"
        "    - id: cli_code.dynamic\n"
        "      kind: asset_generator\n"
        f"      entrypoint: {_CODE_MODULE}:register\n",
        encoding="utf-8",
    )
    _write_inventory(root)
    return root


def _write_inventory(root: Path, *, size_override: int | None = None) -> None:
    """Write the declared tree inventory the build tool would produce."""
    lines: list[str] = []
    for rel in (MANIFEST_NAME, f"python/{_CODE_MODULE}/__init__.py"):
        kind = "manifest" if rel == MANIFEST_NAME else rel.split("/", 1)[0]
        size = (root / rel).stat().st_size if size_override is None else size_override
        lines.append(f"{kind} {rel} {size}")
    (root / "BUILD_MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    """Drive the real ``pack`` argparse group and capture stdout."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    add_pack_subparser(sub)
    args = parser.parse_args(["pack", *argv])
    code = args.fn(args, [])
    return code, capsys.readouterr().out


def test_pack_cli_list_reports_runtime_manifest_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _content_pack(tmp_path / "cli_content")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "list", "--json")

    assert code == 0
    document = json.loads(out)
    assert document["ok"] is True
    rows = {row["name"]: row for row in document["packs"]}
    assert rows["cli_content"] == {
        "name": "cli_content",
        "version": "1.0.0",
        "api": 1,
        "trust": "content",
        "provides": {"runs": 1, "scene_presets": 1},
        "root": str(root.resolve()),
    }
    assert "repo" not in rows


def test_pack_cli_path_prints_root_and_face_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _content_pack(tmp_path / "cli_content")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "path", "cli_content")
    assert code == 0
    assert out.strip() == str(root.resolve())

    code, out = _run_cli(capsys, "path", "cli_content", "scene_presets")
    assert code == 0
    assert (
        out.strip()
        == f"cli_content.tabletop: {(root / 'scene_presets' / 'tabletop.yaml').resolve()}"
    )


def test_pack_cli_path_prints_code_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _code_pack(tmp_path / "cli_code")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "path", "cli_code", "providers")

    assert code == 0
    assert out.strip() == f"cli_code.dynamic: {_CODE_MODULE}:register"
    assert not (root / "activated.txt").exists()


def test_pack_cli_doctor_checks_inventory_and_activates_code_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _code_pack(tmp_path / "cli_code")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "doctor", "--name", "cli_code", "--json")

    assert code == 0, out
    document = json.loads(out)
    assert document["ok"] is True
    checks = document["packs"][0]["checks"]
    assert checks[0].startswith("  ok   manifest:")
    assert any("BUILD_MANIFEST.txt: 2 entries match" in line for line in checks)
    assert any(f"activation: {_CODE_MODULE}:register" in line for line in checks)
    assert (root / "activated.txt").read_text(encoding="utf-8") == "active"


def test_pack_cli_doctor_fails_on_inventory_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _code_pack(tmp_path / "cli_code")
    _write_inventory(root, size_override=1)
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "doctor", "--name", "cli_code", "--json")

    assert code == 1
    document = json.loads(out)
    assert document["ok"] is False
    checks = document["packs"][0]["checks"]
    assert any("FAIL BUILD_MANIFEST.txt" in line for line in checks)
    assert any("size changed" in line for line in checks)


def test_legacy_pack_modules_are_not_importable() -> None:
    import unigym.packs

    assert unigym.packs.__all__ == ["PackError"]
    for name in _DELETED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_production_tree_has_no_legacy_loader_references() -> None:
    root = Path(unigym.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in _DELETED_IDENTIFIERS:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(root)}: {token}")
        for module in _DELETED_MODULES:
            if module in text:
                offenders.append(f"{path.relative_to(root)}: {module}")
    assert offenders == []


def test_pack_error_is_still_the_public_pack_error() -> None:
    from unigym.packs.runtime import PackError as runtime_error

    assert runtime_error is PackError
