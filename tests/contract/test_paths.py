"""Contract for the single-owner path tokens and asset path policy.

Phase 9 splits the generic placeholder helpers into ``unigym.core.paths`` and
the asset path mapping into ``unigym.assets.refs``, then deletes the legacy
asset anchors.  These tests prove the supported token vocabulary, recursive
expansion, teaching errors and the unchanged texture spelling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import unigym
from unigym.assets.refs import canonical_asset_path, texture_payload_rel
from unigym.core.paths import (
    PlaceholderError,
    deps_root,
    expand_placeholders,
    known_tokens,
    repo_root,
)
from unigym.errors import ConfigError


def test_repo_root_is_the_checkout_root() -> None:
    """``repo_root`` follows the imported checkout."""
    expected = str(Path(unigym.__file__).resolve().parents[1])

    assert repo_root() == expected


def test_deps_root_reads_the_environment_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment value wins over a yaml declaration."""
    (tmp_path / "unigym.yaml").write_text("unigym_deps_root: /from/yaml\n", encoding="utf-8")
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(tmp_path / "env-deps"))

    assert deps_root(str(tmp_path)) == str(tmp_path / "env-deps")


def test_deps_root_local_overlay_wins_over_committed_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local overlay is read before the committed yaml."""
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    (tmp_path / "unigym.yaml").write_text("unigym_deps_root: /from/yaml\n", encoding="utf-8")
    (tmp_path / "unigym.local.yaml").write_text(
        "unigym_deps_root: /from/overlay\n", encoding="utf-8"
    )

    assert deps_root(str(tmp_path)) == "/from/overlay"


def test_deps_root_without_an_anchor_is_a_teaching_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset deps root fails with the anchor hint, not a machine default."""
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)

    with pytest.raises(ConfigError, match="no dependency-root anchor"):
        deps_root(str(tmp_path))


def test_known_tokens_are_only_the_two_path_anchors() -> None:
    """Only the two generic path anchors remain in the token vocabulary."""
    assert set(known_tokens()) == {"UNIGYM_REPO", "UNIGYM_DEPS_ROOT"}


def test_known_tokens_expand_recursively_over_a_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token value that still carries a placeholder is re-expanded."""
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", "${UNIGYM_REPO}/deps")
    document = {
        "root": "${UNIGYM_DEPS_ROOT}/unigym_packs",
        "paths": ["${UNIGYM_REPO}/config", 7, None],
    }

    assert expand_placeholders(document) == {
        "root": f"{repo_root()}/deps/unigym_packs",
        "paths": [f"{repo_root()}/config", 7, None],
    }


def test_unknown_token_is_a_teaching_error() -> None:
    """An unknown token fails and names the known vocabulary."""
    with pytest.raises(PlaceholderError) as info:
        expand_placeholders("${UNIGYM_NOT_A_TOKEN}/x")

    message = str(info.value)
    assert "unknown path placeholder ${UNIGYM_NOT_A_TOKEN}" in message
    assert "UNIGYM_DEPS_ROOT" in message
    assert "UNIGYM_REPO" in message


@pytest.mark.parametrize("token", ["UNIGYM_ASSET_ROOT", "UNIGYM_ASSET_PACK"])
def test_removed_asset_tokens_are_teaching_errors(token: str) -> None:
    """The removed asset anchors stay a cutover error, not an unknown typo."""
    with pytest.raises(PlaceholderError, match="was removed by the asset cutover"):
        expand_placeholders("${" + token + "}/generators/x.png")


def test_pack_reference_without_context_is_a_teaching_error() -> None:
    """A whole-string ``pack:`` reference cannot resolve without a context."""
    with pytest.raises(PlaceholderError, match="explicit asset context"):
        expand_placeholders("pack:acme/assets/meshes/cup.stl")


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("tex/a/wood.png", "tex/a/wood/wood.png"),
        ("tex/gentex/d/tex123.png", "tex/gentex/d/tex123/tex123.png"),
        ("tex/a/wood.obj", "tex/a/wood.obj"),
        ("meshes/cup.png", "meshes/cup.png"),
    ],
)
def test_texture_payload_rel_maps_only_tex_image_leaves(declared: str, expected: str) -> None:
    """A ``tex/`` image leaf names its own stem directory."""
    assert texture_payload_rel(Path(declared)) == Path(expected)


def test_canonical_asset_path_maps_the_generators_tail() -> None:
    """The canonical texture spelling is applied to the declared leaf."""
    declared = "pack:acme/assets/generators/tex/a/wood.png"
    canonical = "pack:acme/assets/generators/tex/a/wood/wood.png"

    assert canonical_asset_path(declared) == canonical


def test_canonical_asset_path_leaves_other_paths_unchanged() -> None:
    """A path without a ``/generators/`` segment is returned verbatim."""
    value = "pack:acme/assets/tex/a/wood.png"

    assert canonical_asset_path(value) == value


def test_legacy_asset_root_module_is_deleted() -> None:
    """The retired asset-root module has no compatibility shim."""
    assert importlib.util.find_spec("unigym.core.asset_root") is None
