"""Contract for configured pack-source discovery.

Explicit roots stay authoritative; ``roots=None`` reads the configured
environment, local-overlay and deps sources in the documented order.  Every
boundary is a real environment value or file on disk; no runtime internals are
patched.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from unigym.pack_api import discover
from unigym.packs import PackError

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"
CODE_PYTHON_ROOT = (CODE_ROOT / "python").resolve()


def _drop_pack_modules() -> None:
    """Remove fixture pack modules so no test observes stale import state."""
    for name in [
        name for name in sys.modules if name == "acme_pack" or name.startswith("acme_pack.")
    ]:
        del sys.modules[name]


def _copy_pack(source: Path, target: Path, name: str) -> Path:
    """Copy *source* to *target* and rename its manifest identity and item ids."""
    shutil.copytree(source, target)
    manifest = target / "unigym.pack.yaml"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace("acme_content", name).replace("acme_code", name), encoding="utf-8"
    )
    return target


@pytest.fixture(autouse=True)
def _isolate_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient config and keep fixture import state from leaking."""
    monkeypatch.delenv("UNIGYM_PACKS", raising=False)
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    path_snapshot = list(sys.path)
    _drop_pack_modules()
    yield
    sys.path[:] = path_snapshot
    _drop_pack_modules()


def test_env_roots_are_ordered_and_canonically_deduped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "UNIGYM_PACKS",
        os.pathsep.join([str(CONTENT_ROOT), str(CODE_ROOT), f"{CONTENT_ROOT}/."]),
    )

    runtime = discover(config_path=tmp_path / "unigym.local.yaml")

    assert runtime.catalog.packs() == ("acme_content", "acme_code")
    assert len(runtime.catalog.manifests) == 2
    assert runtime.root_by_pack["acme_content"] == CONTENT_ROOT.resolve()
    assert runtime.root_by_pack["acme_code"] == CODE_ROOT.resolve()


def test_configured_sources_are_collected_in_documented_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = _copy_pack(CONTENT_ROOT, tmp_path / "env_pack", "env_pack")
    yaml_root = _copy_pack(CONTENT_ROOT, tmp_path / "yaml_pack", "yaml_pack")
    deps = tmp_path / "deps"
    _copy_pack(CODE_ROOT, deps / "unigym_packs" / "deps_pack", "deps_pack")
    overlay = tmp_path / "unigym.local.yaml"
    overlay.write_text("packs:\n  - yaml_pack\n", encoding="utf-8")
    monkeypatch.setenv("UNIGYM_PACKS", str(env_root))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(deps))

    runtime = discover(config_path=overlay)

    assert runtime.catalog.packs() == ("env_pack", "yaml_pack", "deps_pack")
    assert runtime.root_by_pack["yaml_pack"] == yaml_root.resolve()


def test_explicit_roots_ignore_configured_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = tmp_path / "deps"
    shutil.copytree(CODE_ROOT, deps / "unigym_packs" / "acme_code")
    overlay = tmp_path / "unigym.local.yaml"
    overlay.write_text(f"packs:\n  - {CODE_ROOT}\n", encoding="utf-8")
    monkeypatch.setenv("UNIGYM_PACKS", str(CODE_ROOT))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(deps))

    runtime = discover([CONTENT_ROOT], config_path=overlay)

    assert runtime.catalog.packs() == ("acme_content",)


def test_deps_root_discovers_sorted_pack_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = tmp_path / "deps"
    top = deps / "unigym_packs"
    shutil.copytree(CONTENT_ROOT, top / "acme_content")
    shutil.copytree(CODE_ROOT, top / "acme_code")
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(deps))

    runtime = discover(config_path=tmp_path / "unigym.local.yaml")

    assert runtime.catalog.packs() == ("acme_code", "acme_content")
    assert runtime.root_by_pack["acme_content"] == (top / "acme_content").resolve()


@pytest.mark.parametrize("deps_state", ["unset", "absent-root", "no-pack-dir"])
def test_missing_deps_sources_yield_no_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deps_state: str
) -> None:
    if deps_state == "absent-root":
        monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(tmp_path / "absent"))
    elif deps_state == "no-pack-dir":
        (tmp_path / "deps").mkdir()
        monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(tmp_path / "deps"))

    runtime = discover(config_path=tmp_path / "unigym.local.yaml")

    assert runtime.catalog.packs() == ()
    assert dict(runtime.root_by_pack) == {}


def test_configured_root_without_manifest_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("UNIGYM_PACKS", str(empty))

    with pytest.raises(PackError, match="missing pack manifest"):
        discover(config_path=tmp_path / "unigym.local.yaml")


def test_configured_root_with_malformed_manifest_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "acme_content"
    shutil.copytree(CONTENT_ROOT, root)
    (root / "unigym.pack.yaml").write_text("schema: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))

    with pytest.raises(PackError, match="unreadable pack manifest"):
        discover(config_path=tmp_path / "unigym.local.yaml")


def test_deps_pack_directory_without_manifest_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = tmp_path / "deps"
    (deps / "unigym_packs" / "broken").mkdir(parents=True)
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(deps))

    with pytest.raises(PackError, match="missing pack manifest"):
        discover(config_path=tmp_path / "unigym.local.yaml")


def test_local_yaml_packs_resolve_relative_to_the_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(CONTENT_ROOT, config_dir / "packs" / "acme_content")
    shutil.copytree(CODE_ROOT, config_dir / "packs" / "acme_code")
    overlay = config_dir / "unigym.local.yaml"
    overlay.write_text(
        "packs:\n  - packs/acme_code\n  - packs/acme_content\n",
        encoding="utf-8",
    )

    runtime = discover(config_path=overlay)

    assert runtime.catalog.packs() == ("acme_code", "acme_content")
    assert runtime.root_by_pack["acme_content"] == (config_dir / "packs" / "acme_content").resolve()


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ("packs: [unclosed\n", "unreadable local pack config"),
        ("- acme_content\n", "must be a mapping"),
        ("packs: acme_content\n", "'packs' must be a list"),
        ("packs:\n  - ''\n", "must be a non-empty path"),
    ],
)
def test_malformed_local_yaml_fails(tmp_path: Path, document: str, reason: str) -> None:
    overlay = tmp_path / "unigym.local.yaml"
    overlay.write_text(document, encoding="utf-8")

    with pytest.raises(PackError, match=reason):
        discover(config_path=overlay)


def test_configured_discovery_does_not_import_pack_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNIGYM_PACKS", os.pathsep.join([str(CODE_ROOT), str(CONTENT_ROOT)]))
    assert "acme_pack" not in sys.modules

    runtime = discover(config_path=tmp_path / "unigym.local.yaml")

    assert runtime.catalog.packs() == ("acme_code", "acme_content")
    assert "acme_pack" not in sys.modules
    assert CODE_PYTHON_ROOT not in {Path(entry or ".").resolve() for entry in sys.path}


@pytest.mark.parametrize(
    "value",
    [
        os.pathsep.join([str(CONTENT_ROOT), "", str(CODE_ROOT)]),
        os.pathsep.join([str(CONTENT_ROOT), "   ", str(CODE_ROOT)]),
        "",
    ],
)
def test_invalid_env_pack_entries_fail_with_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("UNIGYM_PACKS", value)

    with pytest.raises(PackError) as info:
        discover(config_path=tmp_path / "unigym.local.yaml")

    message = str(info.value)
    assert "UNIGYM_PACKS" in message
    assert "entry" in message
    assert "empty" in message


def test_empty_explicit_roots_do_not_fall_back_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = tmp_path / "deps"
    shutil.copytree(CONTENT_ROOT, deps / "unigym_packs" / "acme_content")
    overlay = tmp_path / "unigym.local.yaml"
    overlay.write_text(f"packs:\n  - {CODE_ROOT}\n", encoding="utf-8")
    monkeypatch.setenv("UNIGYM_PACKS", str(CODE_ROOT))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(deps))

    runtime = discover([], config_path=overlay)

    assert runtime.catalog.packs() == ()
    assert dict(runtime.root_by_pack) == {}
