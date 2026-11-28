"""Backend resolution at the launcher/CLI boundary is canonical-only.

``spawn_spec.resolve_backend`` validates both the override and the yaml value
through ``capabilities.build_name``: the five canonical names pass, while the
removed ``isaaclab``/``isaaclab3`` aliases, unknown spellings and a missing
``env.backend`` raise a teaching ``ValueError``.  ``launch`` duplicates the
name tuple for its import-light ``--help`` path; the two must stay identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from unigym.capabilities.traits import CANONICAL_BACKENDS
from unigym.scripts import launch
from unigym.scripts.spawn_spec import resolve_backend


def _write_cfg(tmp_path: Path, backend: str | None, name: str = "run.yaml") -> Path:
    path = tmp_path / name
    if backend is None:
        path.write_text("run_name: demo\nenv: {}\n", encoding="utf-8")
    else:
        path.write_text(f"run_name: demo\nenv:\n  backend: {backend}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("backend", CANONICAL_BACKENDS)
def test_resolve_backend_accepts_each_canonical_name(tmp_path: Path, backend: str) -> None:
    cfg_path = _write_cfg(tmp_path, backend)

    assert resolve_backend(cfg_path, None) == backend


@pytest.mark.parametrize("alias", ["isaaclab", "isaaclab3", "not-a-backend"])
def test_resolve_backend_rejects_removed_aliases_and_unknown_names(
    tmp_path: Path, alias: str
) -> None:
    cfg_path = _write_cfg(tmp_path, alias)

    with pytest.raises(ValueError) as excinfo:
        resolve_backend(cfg_path, None)

    message = str(excinfo.value)
    for canonical in CANONICAL_BACKENDS:
        assert canonical in message
    assert "aliases were removed" in message


def test_resolve_backend_requires_env_backend(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, None)

    with pytest.raises(ValueError) as excinfo:
        resolve_backend(cfg_path, None)

    message = str(excinfo.value)
    assert "env.backend" in message
    for canonical in CANONICAL_BACKENDS:
        assert canonical in message


def test_override_wins_over_the_yaml_backend(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, "mujoco")

    assert resolve_backend(cfg_path, "isaaclab3_newton") == "isaaclab3_newton"


@pytest.mark.parametrize("alias", ["isaaclab", "isaaclab3"])
def test_override_is_validated_too(tmp_path: Path, alias: str) -> None:
    cfg_path = _write_cfg(tmp_path, "mujoco")

    with pytest.raises(ValueError) as excinfo:
        resolve_backend(cfg_path, alias)

    message = str(excinfo.value)
    assert "aliases were removed" in message
    assert "isaaclab3_physx" in message


def test_launch_canonical_names_match_the_registry() -> None:
    names = getattr(launch, "_CANONICAL_BACKENDS", None)

    assert names is not None, "launch._CANONICAL_BACKENDS is missing"
    assert tuple(names) == CANONICAL_BACKENDS


def _load_collect_cli(valid_cfg: Path):
    """Import collect_cli with a valid --config (its module scope peeks it)."""
    saved = list(sys.argv)
    sys.argv = ["unigym-collect", "--config", str(valid_cfg)]
    try:
        return __import__("unigym.scripts.collect_cli", fromlist=["_peek_backend"])
    finally:
        sys.argv = saved


def test_collect_cli_peek_rejects_a_removed_alias(tmp_path: Path) -> None:
    valid_cfg = _write_cfg(tmp_path, "mujoco", name="valid.yaml")
    alias_cfg = _write_cfg(tmp_path, "isaaclab", name="alias.yaml")

    module = _load_collect_cli(valid_cfg)

    with pytest.raises(SystemExit) as excinfo:
        module._peek_backend(str(alias_cfg))

    message = str(excinfo.value)
    assert "canonical" in message
    assert "isaaclab3_physx" in message
