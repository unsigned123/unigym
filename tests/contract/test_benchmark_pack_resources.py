"""Contract for benchmark pack discovery through the runtime ``benchmarks`` face.

A benchmark pack is declared as one runtime resource; ``discover_packs`` loads
the builtin namespace plus every resource-backed ``benchmarks`` catalog entry
through the unchanged ``load_pack`` domain parser.  There is no run-config
pack list, repository scan or dependency scan, and ``benchmark.packs`` is a
teaching error.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from unigym.benchmarks.base import build_tasks
from unigym.benchmarks.packs import discover_packs
from unigym.errors import ConfigError
from unigym.pack_api import discover
from unigym.packs import PackError
from unigym.run.spec import build_spec

MANIFEST = """\
schema: unigym.pack/1
name: acme_bench
version: 1.0.0
api: 1
trust: content
provides:
  benchmarks:
    - id: acme_bench.hold
      resource: benchmarks/hold
"""

PACK_YAML = """\
name: acme_hold
version: 0.1.0
description: temporary contract benchmark pack
"""

TASK_YAML = """\
description: fraction of frames inside the target band
metrics:
  - id: near_zero
    term: frame_fraction
    key: time_in_episode
    target: 0.0
    tol: 1.0
success:
  all:
    - metric: near_zero
      gte: 0.9
"""


def _write_pack(
    tmp_path: Path,
    *,
    manifest: str = MANIFEST,
    pack_yaml: str = PACK_YAML,
    task_yaml: str = TASK_YAML,
    directory: str = "hold",
) -> Path:
    """Write one content pack with a resource-backed benchmark pack."""
    pack = tmp_path / "acme_bench"
    bench = pack / "benchmarks" / directory
    (bench / "tasks").mkdir(parents=True)
    (pack / "unigym.pack.yaml").write_text(manifest, encoding="utf-8")
    (bench / "pack.yaml").write_text(pack_yaml, encoding="utf-8")
    (bench / "tasks" / "hold.yaml").write_text(task_yaml, encoding="utf-8")
    return pack


def _runtime(tmp_path: Path, pack: Path | None = None):
    """Discover a runtime over the temporary content pack."""
    return discover([pack if pack is not None else _write_pack(tmp_path)])


def test_benchmark_catalog_id_resolves_to_a_directory_resource(tmp_path: Path) -> None:
    """A ``benchmarks`` catalog id resolves to the declared pack directory."""
    runtime = _runtime(tmp_path)

    path = runtime.resource_for("benchmarks", "acme_bench.hold")

    assert path.is_dir()
    assert path.name == "hold"


def test_discover_packs_includes_builtin_and_the_declared_pack(tmp_path: Path) -> None:
    """Discovery returns the builtin namespace first, then the catalog pack."""
    runtime = _runtime(tmp_path)

    index = discover_packs(runtime)

    assert list(index.packs) == ["builtin", "acme_hold"]
    assert (
        index.packs["acme_hold"].root == (tmp_path / "acme_bench" / "benchmarks" / "hold").resolve()
    )


def test_task_names_resolve_and_wildcards_expand(tmp_path: Path) -> None:
    """Canonical task names resolve and ``<pack>/*`` expands to the pack tasks."""
    index = discover_packs(_runtime(tmp_path))

    definition, pack_name = index.resolve_task("acme_hold/hold")

    assert pack_name == "acme_hold"
    assert definition.canonical == "acme_hold/hold"
    assert definition.kind == "formula"
    assert index.expand_wildcards([{"name": "acme_hold/*"}]) == [{"name": "acme_hold/hold"}]


def test_build_tasks_builds_the_pack_task(tmp_path: Path) -> None:
    """``build_tasks`` instantiates the declared FormulaTask through the runtime."""
    runtime = _runtime(tmp_path)

    tasks = build_tasks([{"name": "acme_hold/*"}], {"articulations": {}}, runtime)

    assert [task.name for task in tasks] == ["acme_hold/hold"]


def test_discover_packs_without_a_runtime_fails() -> None:
    """Discovery requires the runtime; there is no implicit source."""
    with pytest.raises(TypeError):
        discover_packs()


def test_build_tasks_without_a_runtime_fails() -> None:
    """Task construction requires the runtime; there is no pack list fallback."""
    with pytest.raises(TypeError):
        build_tasks([{"name": "acme_hold/hold"}], {})


def test_entrypoint_only_benchmark_item_fails_naming_the_id(tmp_path: Path) -> None:
    """An entrypoint-only benchmarks item is not a benchmark pack directory."""
    manifest = """\
schema: unigym.pack/1
name: acme_ep
version: 1.0.0
api: 1
trust: code
provides:
  benchmarks:
    - id: acme_ep.tasks
      entrypoint: acme_ep:register
"""
    pack = tmp_path / "acme_ep"
    pack.mkdir()
    (pack / "unigym.pack.yaml").write_text(manifest, encoding="utf-8")
    runtime = discover([pack])

    with pytest.raises(PackError) as info:
        discover_packs(runtime)

    assert "acme_ep.tasks" in str(info.value)


def test_file_benchmark_resource_fails_naming_the_id(tmp_path: Path) -> None:
    """A benchmarks resource that is a file is rejected, not loaded."""
    manifest = """\
schema: unigym.pack/1
name: acme_file
version: 1.0.0
api: 1
trust: content
provides:
  benchmarks:
    - id: acme_file.tasks
      resource: benchmarks/notes.txt
"""
    pack = tmp_path / "acme_file"
    (pack / "benchmarks").mkdir(parents=True)
    (pack / "unigym.pack.yaml").write_text(manifest, encoding="utf-8")
    (pack / "benchmarks" / "notes.txt").write_text("not a pack\n", encoding="utf-8")
    runtime = discover([pack])

    with pytest.raises(PackError) as info:
        discover_packs(runtime)

    assert "acme_file.tasks" in str(info.value)


def test_missing_benchmark_resource_fails_naming_the_id(tmp_path: Path) -> None:
    """A declared resource that does not exist fails naming its catalog id."""
    manifest = """\
schema: unigym.pack/1
name: acme_missing
version: 1.0.0
api: 1
trust: content
provides:
  benchmarks:
    - id: acme_missing.tasks
      resource: benchmarks/absent
"""
    pack = tmp_path / "acme_missing"
    pack.mkdir()
    (pack / "unigym.pack.yaml").write_text(manifest, encoding="utf-8")

    with pytest.raises(PackError) as info:
        discover([pack])

    assert "acme_missing.tasks" in str(info.value)


def test_duplicate_benchmark_pack_names_fail(tmp_path: Path) -> None:
    """Two catalog entries naming the same pack directory fail loudly."""
    manifest = """\
schema: unigym.pack/1
name: acme_bench
version: 1.0.0
api: 1
trust: content
provides:
  benchmarks:
    - id: acme_bench.one
      resource: benchmarks/one
    - id: acme_bench.two
      resource: benchmarks/two
"""
    pack = _write_pack(tmp_path, manifest=manifest, directory="one")
    (pack / "benchmarks" / "two" / "tasks").mkdir(parents=True)
    (pack / "benchmarks" / "two" / "pack.yaml").write_text(PACK_YAML, encoding="utf-8")
    (pack / "benchmarks" / "two" / "tasks" / "hold.yaml").write_text(TASK_YAML, encoding="utf-8")
    runtime = discover([pack])

    with pytest.raises(ValueError, match="declared twice"):
        discover_packs(runtime)


def test_unconfigured_benchmark_directory_is_not_discovered(tmp_path: Path) -> None:
    """A pack directory absent from the runtime catalog is not discovered."""
    pack = _write_pack(tmp_path)
    unused = pack / "benchmarks" / "unused"
    (unused / "tasks").mkdir(parents=True)
    (unused / "pack.yaml").write_text(
        PACK_YAML.replace("acme_hold", "unused_pack"), encoding="utf-8"
    )
    (unused / "tasks" / "hold.yaml").write_text(TASK_YAML, encoding="utf-8")

    index = discover_packs(discover([pack]))

    assert "unused_pack" not in index.packs
    assert list(index.packs) == ["builtin", "acme_hold"]


def test_build_spec_rejects_benchmark_packs(tmp_path: Path) -> None:
    """``benchmark.packs`` is a teaching error, not a second discovery source."""
    cfg = {
        "run_name": "r",
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "benchmark": {"packs": [str(tmp_path / "some_pack")]},
    }

    with pytest.raises(ConfigError) as info:
        build_spec(cfg, Namespace(config="run.yaml"))

    message = str(info.value)
    assert "benchmark.packs" in message
    assert "benchmarks" in message
