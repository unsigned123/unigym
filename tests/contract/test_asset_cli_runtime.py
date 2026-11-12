"""Contract for the asset CLI's runtime-backed selection, pack and generator faces.

The public core has no asset catalog collector, so these tests exercise the
runtime boundary through ``cmd_selection_check`` with an empty selection
directory and through the focused lookup helpers.  Every pack is a real
``unigym.pack/1`` document in ``tmp_path`` discovered through
``UNIGYM_PACKS`` or explicit roots; no internal function under test is
replaced.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import pytest
import yaml

from unigym.commands.asset_cli import (
    AssetCliError,
    _generator_template,
    _pack_root,
    _print_generator,
    _selection_dirs,
    cmd_selection_check,
)
from unigym.pack_api import discover

_MODULE_NAME = "acme_generator_pack"

_GENERATOR_SOURCE = '''"""Temporary code pack for the asset CLI generator contract."""

from __future__ import annotations

from unigym.pack_api import PackContext, register_prop_template


def build_probe(asset_id: str, params: dict) -> dict:
    """Build the fixture prop."""
    return {}


def register(context: PackContext) -> None:
    """Register the fixture template through the public pack surface."""
    register_prop_template("runtime_probe", params={"size": {"type": "size_spec"}})(
        build_probe
    )
'''

_FAILING_SOURCE = '''"""Temporary code pack whose entrypoint fails during activation."""

from __future__ import annotations

from unigym.pack_api import PackContext


def register(context: PackContext) -> None:
    """Fail instead of registering anything."""
    raise RuntimeError("boom")
'''


def _write_pack(
    root: Path,
    *,
    name: str,
    selection: list[tuple[str, str]] | None = None,
) -> Path:
    """Write one content pack manifest and return its root."""
    root.mkdir(parents=True)
    provides: dict[str, list[dict[str, str]]] = {}
    if selection is not None:
        provides["selection"] = [
            {"id": item_id, "resource": resource} for item_id, resource in selection
        ]
    manifest = {
        "schema": "unigym.pack/1",
        "name": name,
        "version": "1.0.0",
        "api": 1,
        "trust": "content",
        "provides": provides,
    }
    (root / "unigym.pack.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return root


def _write_code_pack(root: Path, *, name: str, source: str) -> Path:
    """Write one code pack whose entrypoint is ``acme_generator_pack:register``."""
    root.mkdir(parents=True)
    package = root / "python" / _MODULE_NAME
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")
    manifest = {
        "schema": "unigym.pack/1",
        "name": name,
        "version": "1.0.0",
        "api": 1,
        "trust": "code",
        "entrypoint": f"{_MODULE_NAME}:register",
        "provides": {},
    }
    (root / "unigym.pack.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return root


def _drop_pack_module() -> None:
    """Remove the temporary pack module so each test observes a fresh import."""
    for name in [
        name for name in sys.modules if name == _MODULE_NAME or name.startswith(_MODULE_NAME + ".")
    ]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep configured discovery off ambient packs, entry points and overlays."""
    monkeypatch.delenv("UNIGYM_PACKS", raising=False)
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: importlib.metadata.EntryPoints(())
    )
    monkeypatch.setattr("unigym.packs.runtime.repo_root", lambda: tmp_path)


@pytest.fixture
def _isolate_pack_code() -> None:
    """Restore pack import state and the closed template registry."""
    from unigym.props import templates

    path_snapshot = list(sys.path)
    _drop_pack_module()
    try:
        yield
    finally:
        sys.path[:] = path_snapshot
        _drop_pack_module()
        templates._TEMPLATES.pop("runtime_probe", None)
        templates._TEMPLATE_METADATA.pop("runtime_probe", None)


def _point_discovery_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Point configured discovery at one temporary pack root only."""
    monkeypatch.setenv("UNIGYM_PACKS", str(root))


def test_selection_check_discovers_declared_selection_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A declared selection directory is discovered and checked without a catalog."""
    pack = _write_pack(
        tmp_path / "acme",
        name="acme",
        selection=[("acme.picks", "selection")],
    )
    (pack / "selection").mkdir()
    _point_discovery_at(monkeypatch, pack)

    code = cmd_selection_check(argparse.Namespace(dir=None))

    captured = capsys.readouterr()
    assert code == 0
    assert "selection: OK (0 spec(s) pinned)" in captured.out


def test_selection_check_ignores_undeclared_selection_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pack-root ``selection/`` directory without a catalog item is not discovered."""
    pack = _write_pack(tmp_path / "acme", name="acme", selection=None)
    (pack / "selection").mkdir()
    _point_discovery_at(monkeypatch, pack)

    code = cmd_selection_check(argparse.Namespace(dir=None))

    captured = capsys.readouterr()
    assert code == 0
    assert "no mounted pack declares a 'selection' face" in captured.out
    assert "OK (0 spec(s) pinned)" not in captured.out


def test_selection_dirs_dedupe_two_resources_in_one_directory(tmp_path: Path) -> None:
    """Two selection resources resolving to one directory yield one entry."""
    pack = _write_pack(
        tmp_path / "acme",
        name="acme",
        selection=[
            ("acme.one", "selection/one.yaml"),
            ("acme.two", "selection/two.yaml"),
        ],
    )
    selection_dir = pack / "selection"
    selection_dir.mkdir()
    (selection_dir / "one.yaml").write_text("name: one\n", encoding="utf-8")
    (selection_dir / "two.yaml").write_text("name: two\n", encoding="utf-8")
    runtime = discover([pack])

    directories = _selection_dirs(None, runtime)

    assert directories == [selection_dir.resolve()]


def test_pack_root_resolves_inside_the_runtime_pack_root(tmp_path: Path) -> None:
    """The pack lookup resolves the asset directory through the runtime resource owner."""
    pack = _write_pack(tmp_path / "acme", name="acme", selection=None)
    (pack / "assets" / "cup").mkdir(parents=True)
    runtime = discover([pack])

    root = _pack_root("acme", "pack:acme/assets/cup", runtime)

    assert root == (pack / "assets" / "cup").resolve()


def test_pack_root_rejects_an_asset_directory_that_escapes_the_pack_root(
    tmp_path: Path,
) -> None:
    """The runtime resource owner keeps symlink-escape checks in one place."""
    pack = _write_pack(tmp_path / "acme", name="acme", selection=None)
    (pack / "assets").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (pack / "assets" / "escape").symlink_to(outside)
    runtime = discover([pack])

    with pytest.raises(AssetCliError, match="escapes the pack root"):
        _pack_root("acme", "pack:acme/assets/escape", runtime)


def test_pack_root_unknown_namespace_names_available_packs(tmp_path: Path) -> None:
    """An unknown asset namespace fails closed and names the mounted packs."""
    pack = _write_pack(tmp_path / "acme", name="acme", selection=None)
    runtime = discover([pack])

    with pytest.raises(AssetCliError) as info:
        _pack_root("ghost", "pack:ghost/assets/cup", runtime)

    message = str(info.value)
    assert "no mounted pack owns namespace 'ghost'" in message
    assert "mounted packs: ['acme']" in message
    assert "UNIGYM_PACKS" in message


def test_selection_dirs_requires_a_runtime_for_default_discovery() -> None:
    """Default selection discovery cannot silently fall back to a root scan."""
    with pytest.raises(AssetCliError, match="needs a pack runtime"):
        _selection_dirs(None, None)


def test_selection_dirs_explicit_directory_needs_no_runtime(tmp_path: Path) -> None:
    """An explicit ``--dir`` keeps working without configured pack discovery."""
    directory = tmp_path / "selection"
    directory.mkdir()

    assert _selection_dirs(str(directory), None) == [directory.resolve()]


def test_generator_template_activates_the_code_pack_and_resolves_its_template(
    tmp_path: Path, _isolate_pack_code: None
) -> None:
    """Activation runs the pack entrypoint, which registers the named template."""
    pack = _write_code_pack(
        tmp_path / "acme_generator", name="acme_generator", source=_GENERATOR_SOURCE
    )
    runtime = discover([pack])

    template, params = _generator_template("runtime_probe", runtime)

    assert template == f"{_MODULE_NAME}:build_probe"
    assert params == ["size"]


def test_print_generator_reports_activation_failure_without_a_traceback(
    tmp_path: Path,
    _isolate_pack_code: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing code-pack entrypoint is a teaching CLI error, not a traceback."""
    pack = _write_code_pack(tmp_path / "acme_broken", name="acme_broken", source=_FAILING_SOURCE)
    runtime = discover([pack])

    code = _print_generator("pack:acme_broken/generator/failing_probe", runtime)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("[unigym asset] ")
    assert "acme_broken" in captured.out
    assert f"{_MODULE_NAME}:register" in captured.out
    assert "boom" in captured.out
    assert "Traceback" not in captured.out + captured.err
