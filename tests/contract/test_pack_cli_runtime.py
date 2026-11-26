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
from unigym.actuator_law import restore_actuator_laws, snapshot_actuator_laws
from unigym.adapters import restore_adapter_registry, snapshot_adapter_registry
from unigym.initializers.base import restore_initializers, snapshot_initializers
from unigym.packs import PackError
from unigym.packs.cli import add_pack_subparser
from unigym.packs.schema import FACE_NAMES, MANIFEST_NAME
from unigym.props.templates import restore_prop_templates, snapshot_prop_templates
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.scenes.pipeline import (
    restore_scene_content_fillers,
    snapshot_scene_content_fillers,
)
from unigym.scenes.providers import restore_scene_providers, snapshot_scene_providers
from unigym.terminations import (
    restore_termination_registry,
    snapshot_termination_registry,
)

_CODE_MODULE = "cli_code_pack"
_FILLER_MODULE = "cli_filler_pack"

_DELETED_MODULES = (
    "unigym.packs.catalog",
    "unigym.packs.load",
    "unigym.packs.runtime",
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
    """Clear ambient discovery and restore path, modules and registries."""
    monkeypatch.delenv("UNIGYM_PACKS", raising=False)
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    path_snapshot = list(sys.path)
    robots = snapshot_robot_registry()
    adapters = snapshot_adapter_registry()
    scene_providers = snapshot_scene_providers()
    content_fillers = snapshot_scene_content_fillers()
    templates = snapshot_prop_templates()
    terminations = snapshot_termination_registry()
    laws = snapshot_actuator_laws()
    initializers = snapshot_initializers()
    yield
    restore_robot_registry(robots)
    restore_adapter_registry(adapters)
    restore_scene_providers(scene_providers)
    restore_scene_content_fillers(content_fillers)
    restore_prop_templates(templates)
    restore_termination_registry(terminations)
    restore_actuator_laws(laws)
    restore_initializers(initializers)
    sys.path[:] = path_snapshot
    for name in [
        name
        for name in sys.modules
        if any(
            name == module or name.startswith(module + ".")
            for module in (_CODE_MODULE, _FILLER_MODULE)
        )
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


def _params_pack(root: Path) -> Path:
    """Write a declared scene-provider pack with one non-empty params schema."""
    root.mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(
        "schema: unigym.pack/1\n"
        "name: cli_params\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: code\n"
        "provides:\n"
        "  providers:\n"
        "    - id: cli_params.scene\n"
        "      kind: scene\n"
        "      entrypoint: cli_params_pack:scene\n"
        "      params:\n"
        "        style:\n"
        "          type: int\n"
        "          default: 1\n",
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


def _write_tree_inventory(root: Path) -> None:
    """Write the declared inventory matching the current fixture tree."""
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "BUILD_MANIFEST.txt":
            continue
        first = rel.split("/", 1)[0]
        kind = first if first in (*FACE_NAMES, "python") else "manifest"
        lines.append(f"{kind} {rel} {path.stat().st_size}")
    (root / "BUILD_MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filler_code_pack(root: Path, *, register_filler: bool = True) -> Path:
    """Write a code pack with one manifest robot and an optional pack-code filler."""
    (root / "robots").mkdir(parents=True)
    (root / "robots" / "arm.yaml").write_text(
        "schema: unigym.robot/1\n"
        "name: cli_filler_arm\n"
        "joint_sets:\n"
        "  body: [shoulder]\n"
        "default_joint_pos:\n"
        "  shoulder: 0.0\n",
        encoding="utf-8",
    )
    register_line = (
        "    register_scene_content_filler('cli_filler', cli_filler)\n"
        if register_filler
        else "    return None\n"
    )
    module = root / "python" / _FILLER_MODULE / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        '"""CLI fixture pack; activation may register one content filler."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from unigym.pack_api import PackContext, register_scene_content_filler\n"
        "\n"
        "\n"
        "def cli_filler(**_kwargs):\n"
        "    return object()\n"
        "\n"
        "\n"
        "def register(context: PackContext) -> None:\n"
        f"{register_line}",
        encoding="utf-8",
    )
    (root / MANIFEST_NAME).write_text(
        "schema: unigym.pack/1\n"
        "name: cli_filler\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: code\n"
        f"entrypoint: {_FILLER_MODULE}:register\n"
        "provides:\n"
        "  providers:\n"
        "    - id: cli_filler_arm\n"
        "      kind: robot\n"
        "      resource: robots/arm.yaml\n",
        encoding="utf-8",
    )
    _write_tree_inventory(root)
    return root


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
        "origin": {
            "kind": "pack",
            "pack": "cli_content",
            "pack_version": "1.0.0",
            "root": str(root.resolve()),
            "discovered_by": "env:UNIGYM_PACKS",
            "face": None,
            "item_id": None,
            "resource": None,
            "entrypoint": None,
            "activation_entrypoint": None,
            "location_imported": False,
        },
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


def test_pack_cli_show_json_lists_every_declared_item_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default ``show --json`` is a manifest-only, face-ordered inventory."""
    root = _content_pack(tmp_path / "cli_content")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "show", "cli_content", "--json")

    assert code == 0
    document = json.loads(out)
    assert document["schema"] == "unigym.pack_inventory/1"
    assert document["ok"] is True
    assert document["runtime"] is None
    assert document["pack"]["name"] == "cli_content"
    assert document["pack"]["version"] == "1.0.0"
    assert document["pack"]["api"] == 1
    assert document["pack"]["trust"] == "content"
    assert document["pack"]["root"] == str(root.resolve())
    assert document["pack"]["entrypoint"] is None
    assert document["pack"]["requires"] == {}
    assert document["pack"]["face_counts"] == {"runs": 1, "scene_presets": 1}
    assert document["pack"]["origin"]["kind"] == "pack"
    assert [item["face"] for item in document["items"]] == ["runs", "scene_presets"]
    assert [item["id"] for item in document["items"]] == [
        "cli_content.demo",
        "cli_content.tabletop",
    ]
    for item in document["items"]:
        assert item["params"] == {}
        assert item["status"] == "declared"
        assert item["origin"]["kind"] == "manifest"
        assert item["origin"]["pack"] == "cli_content"


def test_pack_cli_show_json_serializes_mapping_proxy_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-empty deep-frozen params serialize through the JSON inventory."""
    root = _params_pack(tmp_path / "cli_params")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "show", "cli_params", "--json")

    assert code == 0
    item = json.loads(out)["items"][0]
    assert item["entrypoint"] == "cli_params_pack:scene"
    assert item["params"] == {"style": {"type": "int", "default": 1}}


def test_pack_cli_show_default_does_not_activate_pack_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A default ``show`` on a code pack resolves declarations only."""
    root = _code_pack(tmp_path / "cli_code")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "show", "cli_code", "--json")

    assert code == 0
    document = json.loads(out)
    assert document["runtime"] is None
    assert document["pack"]["entrypoint"] == f"{_CODE_MODULE}:register"
    assert not (root / "activated.txt").exists()


def test_pack_cli_show_runtime_json_distinguishes_manifest_and_pack_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``show --runtime`` reports one manifest and one pack-code registration."""
    root = _filler_code_pack(tmp_path / "cli_filler")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "show", "cli_filler", "--runtime", "--json")

    assert code == 0, out
    document = json.loads(out)
    runtime = document["runtime"]
    assert runtime["activated"] is True
    registrations = runtime["registrations"]
    assert set(registrations) == {"robots", "scene_content_fillers"}

    robot = registrations["robots"][0]
    assert robot["name"] == "cli_filler_arm"
    assert robot["kind"] == "robot"
    assert robot["live"] is True
    assert robot["origin"]["kind"] == "manifest"
    assert robot["origin"]["item_id"] == "cli_filler_arm"

    filler = registrations["scene_content_fillers"][0]
    assert filler["name"] == "cli_filler"
    assert filler["kind"] == "content_filler"
    assert filler["live"] is True
    assert filler["origin"]["kind"] == "pack-code"
    assert filler["origin"]["item_id"] is None


def test_pack_cli_doctor_strict_json_fails_on_pack_code_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Strict doctor fails and names the registry, item and ``item_id=None``."""
    root = _filler_code_pack(tmp_path / "cli_filler")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "doctor", "--name", "cli_filler", "--strict", "--json")

    assert code == 1
    document = json.loads(out)
    assert document["ok"] is False
    checks = document["packs"][0]["checks"]
    assert any(line.startswith("  FAIL strict:") for line in checks)
    assert any(
        "scene_content_fillers:" in line and "cli_filler" in line and "item_id=None" in line
        for line in checks
    )


def test_pack_cli_doctor_strict_passes_for_declared_only_code_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live manifest registration passes strict; no pack-code record exists."""
    root = _filler_code_pack(tmp_path / "cli_filler", register_filler=False)
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "doctor", "--name", "cli_filler", "--strict", "--json")

    assert code == 0, out
    checks = json.loads(out)["packs"][0]["checks"]
    assert any(line.startswith("  ok   strict:") for line in checks)


def test_pack_cli_show_runtime_reports_one_manifest_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A declared-only code pack's live record is ``manifest``, never ``pack-code``."""
    root = _filler_code_pack(tmp_path / "cli_filler", register_filler=False)
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "show", "cli_filler", "--runtime", "--json")

    assert code == 0, out
    registrations = json.loads(out)["runtime"]["registrations"]
    assert set(registrations) == {"robots"}
    assert registrations["robots"][0]["origin"]["kind"] == "manifest"
    assert all(
        record["origin"]["kind"] != "pack-code"
        for records in registrations.values()
        for record in records
    )


def test_pack_cli_doctor_default_ignores_pack_code_registrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--strict`` the existing doctor checks stay unchanged."""
    root = _filler_code_pack(tmp_path / "cli_filler")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    code, out = _run_cli(capsys, "doctor", "--name", "cli_filler", "--json")

    assert code == 0, out
    checks = json.loads(out)["packs"][0]["checks"]
    assert not any("strict:" in line for line in checks)


def test_legacy_pack_modules_are_not_importable() -> None:
    import unigym.packs

    assert unigym.packs.__all__ == [
        "AssetResourceRef",
        "EntrypointLocator",
        "Face",
        "ItemStatus",
        "Locator",
        "Pack",
        "PackError",
        "PackItem",
        "PackLoader",
        "PackSet",
        "ProviderOrigin",
        "ProviderRef",
        "ResourceLocator",
        "ScenePresetRef",
        "SceneProviderRef",
        "discover_set",
        "installed",
        "require",
    ]
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
                if re.search(rf"\b{re.escape(module)}\b", text):
                    offenders.append(f"{path.relative_to(root)}: {module}")
    assert offenders == []


def test_pack_error_is_still_the_public_pack_error() -> None:
    from unigym.packs.loader import PackError as runtime_error

    assert runtime_error is PackError
