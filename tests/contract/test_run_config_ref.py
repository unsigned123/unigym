"""Contract for run config references through absolute paths or catalog ids.

A run config ref is either an absolute file path or an exact PackRuntime
``runs`` catalog id. Relative paths and globs are rejected; the CLI and daemon
share the same resolver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.pack_api import discover
from unigym.packs import PackError
from unigym.run.config_ref import RunConfigRefError, resolve_run_config

MANIFEST = """\
schema: unigym.pack/1
name: acme_runs
version: 1.0.0
api: 1
trust: content
provides:
  runs:
    - id: acme_runs.demo
      resource: runs/demo.yaml
    - id: acme_runs.directory
      resource: runs_dir
"""

RUN = """\
run_name: acme_demo
env:
  backend: mujoco
"""


def _pack_root(tmp_path: Path) -> Path:
    """Write one content pack with a run resource and a directory resource."""
    pack = tmp_path / "acme_runs"
    (pack / "runs").mkdir(parents=True)
    (pack / "runs_dir").mkdir()
    (pack / "unigym.pack.yaml").write_text(MANIFEST, encoding="utf-8")
    (pack / "runs" / "demo.yaml").write_text(RUN, encoding="utf-8")
    return pack


def _runtime(tmp_path: Path):
    return discover([_pack_root(tmp_path)])


def test_absolute_path_resolves_without_a_runtime(tmp_path: Path) -> None:
    """An absolute run config file resolves with no pack runtime."""
    config = _pack_root(tmp_path) / "runs" / "demo.yaml"

    resolved = resolve_run_config(str(config))

    assert resolved == config.resolve()
    assert resolved.is_file()


def test_exact_runs_id_resolves_through_the_runtime(tmp_path: Path) -> None:
    """An exact ``runs`` catalog id resolves to its declared file."""
    runtime = _runtime(tmp_path)

    resolved = resolve_run_config("acme_runs.demo", runtime)

    assert resolved.name == "demo.yaml"
    assert resolved == (tmp_path / "acme_runs" / "runs" / "demo.yaml").resolve()
    assert resolved.is_file()


def test_unknown_id_names_the_id_and_known_ids(tmp_path: Path) -> None:
    """An unknown catalog id fails with the runtime's known-id list."""
    runtime = _runtime(tmp_path)

    with pytest.raises(PackError) as info:
        resolve_run_config("acme_runs.missing", runtime)

    message = str(info.value)
    assert "unknown id 'acme_runs.missing'" in message
    assert "acme_runs.demo" in message
    assert "acme_runs.directory" in message


@pytest.mark.parametrize(
    "spec",
    ["runs/demo.yaml", "demo.yaml", "*.yaml", "runs/*.yaml", "./runs/demo.yaml"],
    ids=["subpath", "filename", "glob", "subpath-glob", "dot-relative"],
)
def test_relative_path_and_glob_are_rejected(tmp_path: Path, spec: str) -> None:
    """Relative paths and globs teach the two accepted forms instead."""
    runtime = _runtime(tmp_path)

    with pytest.raises(RunConfigRefError) as info:
        resolve_run_config(spec, runtime)

    message = str(info.value)
    assert "is not supported" in message
    assert "absolute file path" in message
    assert "'runs' pack catalog id" in message
    assert "known ids" not in message


@pytest.mark.parametrize("spec", ["", "   "], ids=["empty", "whitespace"])
def test_empty_ref_is_rejected(spec: str) -> None:
    """An empty run config ref is a teaching error."""
    with pytest.raises(RunConfigRefError) as info:
        resolve_run_config(spec)

    assert "non-empty string" in str(info.value)


def test_catalog_id_without_runtime_fails_fast() -> None:
    """The catalog id form requires a runtime and names the ref."""
    with pytest.raises(RunConfigRefError) as info:
        resolve_run_config("acme_runs.demo")

    message = str(info.value)
    assert "'acme_runs.demo'" in message
    assert "requires a pack runtime" in message


def test_directory_resource_is_rejected_as_not_a_file(tmp_path: Path) -> None:
    """A catalog resource that is a directory is not a run config file."""
    runtime = _runtime(tmp_path)

    with pytest.raises(RunConfigRefError) as info:
        resolve_run_config("acme_runs.directory", runtime)

    assert "is not a file" in str(info.value)
