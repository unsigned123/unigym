"""Contract for the explicit static-root pack runtime.

Discovery reads fixture manifests only; resource lookup stays inside each
pack root; activation is explicit normal Python and cached per runtime.
Negative fixtures are copied into ``tmp_path`` so the shared support packs
stay untouched.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from unigym.pack_api import ActivationHandle, PackRuntime, discover
from unigym.packs import PackError

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"
CODE_PYTHON_ROOT = (CODE_ROOT / "python").resolve()


def _drop_pack_modules() -> None:
    """Remove fixture pack modules so every test observes a fresh import."""
    for name in [
        name for name in sys.modules if name == "acme_pack" or name.startswith("acme_pack.")
    ]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_pack_modules() -> None:
    """Keep fixture import state and sys.path from leaking between tests."""
    path_snapshot = list(sys.path)
    _drop_pack_modules()
    yield
    sys.path[:] = path_snapshot
    _drop_pack_modules()


def _spy_entrypoint(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Import the real fixture module and replace its entrypoint with a spy.

    The fixture declares the Phase 3 no-argument ``register()`` entrypoint.
    The spy records the activation call while proving the runtime imports the
    entrypoint from the fixture's ``python/`` boundary.
    """
    monkeypatch.syspath_prepend(str(CODE_PYTHON_ROOT))
    module = importlib.import_module("acme_pack")
    module_file = Path(module.__file__).resolve()
    assert module_file.is_relative_to(CODE_PYTHON_ROOT)
    calls: list[str] = []

    def record() -> None:
        calls.append("register")

    monkeypatch.setattr(module, "register", record)
    return calls


def test_discover_content_and_code_roots_in_order() -> None:
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    assert isinstance(runtime, PackRuntime)
    assert runtime.catalog.packs() == ("acme_content", "acme_code")
    assert [entry.id for entry in runtime.catalog.entries()] == [
        "acme_content.demo",
        "acme_content.tabletop",
        "acme_content.cup",
        "acme_code.dynamic_assets",
    ]
    assert runtime.root_by_pack["acme_content"] == CONTENT_ROOT.resolve()
    assert runtime.root_by_pack["acme_code"] == CODE_ROOT.resolve()


def test_resource_resolves_under_the_pack_root() -> None:
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    assert (
        runtime.resource("acme_content", "runs/demo.yaml")
        == (CONTENT_ROOT / "runs" / "demo.yaml").resolve()
    )
    assert runtime.resource("acme_content", "assets/cup/asset.yaml").is_file()


def test_discover_dedupes_canonical_duplicate_roots() -> None:
    runtime = discover([CONTENT_ROOT, CONTENT_ROOT, f"{CONTENT_ROOT}/."])

    assert runtime.catalog.packs() == ("acme_content",)
    assert len(runtime.catalog.manifests) == 1


def test_discover_rejects_duplicate_pack_names(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    shutil.copytree(CONTENT_ROOT, first)
    shutil.copytree(CONTENT_ROOT, second)

    with pytest.raises(PackError, match="duplicate pack name 'acme_content'"):
        discover([first, second])


def test_discover_rejects_missing_root_or_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(PackError, match="missing pack manifest"):
        discover([empty])
    with pytest.raises(PackError, match="missing pack manifest"):
        discover([tmp_path / "absent"])


@pytest.mark.parametrize(
    ("rel", "reason"),
    [
        ("", "non-empty relative path"),
        ("/etc/passwd", "must be relative to the pack root"),
        ("../outside.yaml", "must not contain '..'"),
        ("missing.yaml", "does not exist"),
    ],
)
def test_resource_rejects_illegal_or_missing_paths(tmp_path: Path, rel: str, reason: str) -> None:
    root = tmp_path / "acme_content"
    shutil.copytree(CONTENT_ROOT, root)
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.resource("acme_content", rel)

    message = str(info.value)
    assert "acme_content" in message
    assert reason in message


def test_resource_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "acme_content"
    shutil.copytree(CONTENT_ROOT, root)
    outside = tmp_path / "outside.yaml"
    outside.write_text("kind: outside\n", encoding="utf-8")
    (root / "escape.yaml").symlink_to(outside)
    runtime = discover([root])

    with pytest.raises(PackError, match="escapes the pack root"):
        runtime.resource("acme_content", "escape.yaml")


def test_discovery_does_not_import_pack_python() -> None:
    assert "acme_pack" not in sys.modules

    discover([CODE_ROOT, CONTENT_ROOT])

    assert "acme_pack" not in sys.modules


def test_discovered_roots_are_fixture_relative() -> None:
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    for root in runtime.root_by_pack.values():
        assert root.is_relative_to(PACKS_ROOT.resolve())


def test_code_activation_imports_and_calls_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_entrypoint(monkeypatch)
    runtime = discover([CODE_ROOT])

    handle = runtime.activate("acme_code")

    assert handle == ActivationHandle(pack="acme_code", entrypoint="acme_pack:register")
    assert calls == ["register"]
    assert str(CODE_PYTHON_ROOT) in sys.path


def test_fixture_code_entrypoint_accepts_no_arguments() -> None:
    runtime = discover([CODE_ROOT])

    handle = runtime.activate("acme_code")

    assert handle == ActivationHandle(pack="acme_code", entrypoint="acme_pack:register")


def test_content_activation_is_a_noop() -> None:
    runtime = discover([CONTENT_ROOT])

    handle = runtime.activate("acme_content")

    assert handle == ActivationHandle(pack="acme_content", entrypoint=None)
    assert "acme_pack" not in sys.modules


def test_activation_failure_wraps_pack_and_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "acme_code"
    shutil.copytree(CODE_ROOT, root)
    (root / "python" / "acme_pack" / "__init__.py").write_text(
        '"""Broken fixture entrypoint."""\n\n'
        "from __future__ import annotations\n\n\n"
        "def register() -> None:\n"
        '    raise RuntimeError("boom")\n',
        encoding="utf-8",
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("acme_code")

    message = str(info.value)
    assert "acme_code" in message
    assert "acme_pack:register" in message
    assert "boom" in message
    assert isinstance(info.value.__cause__, RuntimeError)


def test_repeated_activation_returns_the_cached_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_entrypoint(monkeypatch)
    runtime = discover([CODE_ROOT])

    first = runtime.activate("acme_code")
    second = runtime.activate("acme_code")

    assert first is second
    assert calls == ["register"]


def test_runtime_module_has_no_legacy_loader_bindings() -> None:
    import unigym.packs.runtime as runtime_module

    for name in ("MountedPack", "load_pack_code", "mount", "pack_roots"):
        assert name not in runtime_module.__dict__
