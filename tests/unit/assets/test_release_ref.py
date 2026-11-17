"""``release:`` asset references resolve only at the boundary."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from unigym.assets import release
from unigym.assets.asset_ref import AssetRefError
from unigym.scenes.articulation import normalize_articulation


def test_is_release_ref_requires_the_whole_string_prefix() -> None:
    assert release.is_release_ref("release:go2-mjcf")
    assert not release.is_release_ref("pack:acme/robot.xml")
    assert not release.is_release_ref("/abs/release:go2-mjcf")
    assert not release.is_release_ref(None)


def test_resolve_release_ref_returns_an_existing_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "go2.xml"
    asset.write_text("<mujoco/>\n", encoding="utf-8")
    requested: list[str] = []

    def path_resolver(asset_id: str) -> Path:
        requested.append(asset_id)
        return asset

    monkeypatch.setattr(release, "_resolver", lambda: path_resolver)

    assert release.resolve_release_ref("release:go2-mjcf") == str(asset.resolve())
    assert requested == ["go2-mjcf"]


def test_resolve_release_ref_reports_a_missing_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release, "_resolver", lambda: lambda asset_id: tmp_path / "absent.xml")

    with pytest.raises(AssetRefError, match="not an existing file") as info:
        release.resolve_release_ref("release:go2-mjcf")

    assert "unigym-robots download" in str(info.value)


def test_resolve_release_ref_without_a_resolver_is_a_teaching_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "_entry_point_resolver", lambda: None)
    monkeypatch.setattr(release, "_robots_resolver", lambda: None)

    with pytest.raises(AssetRefError, match="install unigym-robots"):
        release.resolve_release_ref("release:go2-mjcf")


def test_articulation_model_keeps_the_release_declaration() -> None:
    """Spec normalization is pure data; resolution belongs to the build path."""
    spec = normalize_articulation({"name": "go2", "model": {"mjcf": "release:go2-mjcf"}})

    assert spec.model is not None
    assert spec.model.mjcf == "release:go2-mjcf"


def test_installed_entry_point_provider_is_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real ``unigym.asset_release`` distribution on sys.path is used."""
    (tmp_path / "unigym_ep_provider.py").write_text(
        "from pathlib import Path\n"
        "def resolve(asset_id):\n"
        "    return Path(__file__).parent / 'assets' / f'{asset_id}.xml'\n",
        encoding="utf-8",
    )
    asset = tmp_path / "assets" / "tiny.xml"
    asset.parent.mkdir(parents=True)
    asset.write_text("<mujoco/>\n", encoding="utf-8")
    dist_info = tmp_path / "unigym_ep_provider-0.0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unigym-ep-provider\nVersion: 0.0.1\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[unigym.asset_release]\nep = unigym_ep_provider:resolve\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert release.resolve_release_ref("release:tiny") == str(asset.resolve())


def test_robots_fallback_reports_a_broken_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed-but-broken ``unigym_robots`` is not "not installed"."""
    broken = tmp_path / "unigym_robots"
    broken.mkdir()
    (broken / "__init__.py").write_text(
        "raise RuntimeError('broken resolver install')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.delitem(sys.modules, "unigym_robots", raising=False)

    with pytest.raises(AssetRefError, match="failed to import"):
        release._robots_resolver()
