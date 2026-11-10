"""Lane registry invariants: every test file belongs to exactly one lane."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "ci" / "compat.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
SUPPORTED_VERSION = 1
PROVISIONS = {"uv", "conda"}
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _tables(document: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    value = document.get(key)
    assert isinstance(value, dict), f"{key!r} must be a TOML table"
    tables: dict[str, dict[str, object]] = {}
    for name, entry in value.items():
        assert isinstance(name, str), f"{key!r} must use string keys, got {name!r}"
        assert isinstance(entry, dict), f"{key}.{name} must be a TOML table"
        tables[name] = entry
    return tables


def _nested_table(document: dict[str, object], *keys: str) -> dict[str, object]:
    current = document
    for key in keys:
        value = current.get(key)
        assert isinstance(value, dict), f"missing [{'.'.join(keys)}] table"
        current = value
    return current


def _is_repo_relative(text: str) -> bool:
    path = PurePosixPath(text)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and not _WINDOWS_ABSOLUTE.match(text)
    )


def _looks_like_machine_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("~")
        or bool(_WINDOWS_ABSOLUTE.match(value))
        or "/home/" in value
        or "/data" in value
    )


def _lane_selectors(registry: dict[str, object]) -> dict[str, list[str]]:
    selectors: dict[str, list[str]] = {}
    for name, lane in _tables(registry, "lanes").items():
        value = lane.get("selectors")
        selectors[name] = (
            [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
        )
    return selectors


def _covers(selector: str, test_path: str) -> bool:
    selector_path = PurePosixPath(selector)
    test = PurePosixPath(test_path)
    return test == selector_path or selector_path in test.parents


def _test_files() -> list[str]:
    return sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("test_*.py"))


@pytest.fixture(scope="module")
def registry() -> dict[str, object]:
    return _load_toml(REGISTRY_PATH)


@pytest.fixture(scope="module")
def environments(registry: dict[str, object]) -> dict[str, dict[str, object]]:
    return _tables(registry, "environments")


def test_registry_parses_with_supported_version(registry):
    """The lane registry is TOML with the schema version the runner supports."""
    version = registry.get("version")
    assert isinstance(version, int) and not isinstance(
        version, bool
    ), f"registry version must be an integer, got {version!r}"
    assert version == SUPPORTED_VERSION


def test_every_lane_references_a_declared_environment(registry, environments):
    """A lane names an environment that the registry actually declares."""
    lanes = _tables(registry, "lanes")
    assert lanes, "registry must declare at least one lane"
    undeclared = sorted(
        f"{name} -> {lane.get('environment')!r}"
        for name, lane in lanes.items()
        if lane.get("environment") not in environments
    )
    assert not undeclared, f"lanes referencing undeclared environments: {undeclared}"


def test_every_environment_declares_a_known_provisioner(environments):
    """Each environment states how it is provisioned."""
    offenders = {
        name: environment.get("provision")
        for name, environment in environments.items()
        if environment.get("provision") not in PROVISIONS
    }
    assert not offenders, f"environments with an unknown provisioner: {offenders}"


def test_uv_environments_declare_extras(environments):
    """A uv environment lists the non-empty extras its lane installs."""
    offenders = []
    for name, environment in environments.items():
        if environment.get("provision") != "uv":
            continue
        extras = environment.get("extras")
        if (
            not isinstance(extras, list)
            or not extras
            or not all(isinstance(extra, str) and extra for extra in extras)
        ):
            offenders.append(name)
    assert not offenders, f"uv environments without a non-empty extras list: {offenders}"


def test_conda_environments_declare_env_var_manifest_and_strict(environments):
    """A conda environment is anchored by a variable, a manifest and strictness."""
    offenders = []
    for name, environment in environments.items():
        if environment.get("provision") != "conda":
            continue
        env_var = environment.get("env_var")
        manifest = environment.get("manifest")
        strict = environment.get("strict")
        if not isinstance(env_var, str) or not env_var.startswith("UNIGYM_"):
            offenders.append((name, "env_var", env_var))
        if not isinstance(manifest, str) or not _is_repo_relative(manifest):
            offenders.append((name, "manifest", manifest))
        if not isinstance(strict, bool):
            offenders.append((name, "strict", strict))
    assert not offenders, f"conda environments without full anchoring: {offenders}"


def test_every_lane_declares_repo_relative_existing_selectors(registry):
    """Lane selectors are repo-relative paths that exist on disk."""
    offenders = []
    for name, lane in _tables(registry, "lanes").items():
        selectors = lane.get("selectors")
        if not isinstance(selectors, list) or not selectors:
            offenders.append((name, selectors))
            continue
        for selector in selectors:
            if (
                not isinstance(selector, str)
                or not _is_repo_relative(selector)
                or not (ROOT / selector).exists()
            ):
                offenders.append((name, selector))
    assert not offenders, f"lane selectors that are missing or not repo-relative: {offenders}"


def test_lane_selectors_do_not_overlap(registry):
    """No two lanes own the same path or a nested path."""
    selectors = _lane_selectors(registry)
    names = list(selectors)
    overlaps = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for left_selector in selectors[left]:
                for right_selector in selectors[right]:
                    if _covers(left_selector, right_selector) or _covers(
                        right_selector, left_selector
                    ):
                        overlaps.append((left, left_selector, right, right_selector))
    assert not overlaps, f"overlapping lane selectors: {overlaps}"


def test_every_test_file_belongs_to_exactly_one_lane(registry):
    """Path enumeration assigns every test file to exactly one lane selector."""
    selectors = _lane_selectors(registry)
    unassigned = []
    ambiguous = []
    for test_file in _test_files():
        owners = [
            name
            for name, lane_selectors in selectors.items()
            if any(_covers(selector, test_file) for selector in lane_selectors)
        ]
        if not owners:
            unassigned.append(test_file)
        elif len(owners) > 1:
            ambiguous.append((test_file, owners))
    assert not unassigned, f"test files outside every lane selector: {unassigned}"
    assert not ambiguous, f"test files matched by more than one lane: {ambiguous}"


def test_default_testpaths_equal_d0_selectors(registry):
    """A bare pytest run collects exactly the d0 lane selectors."""
    d0 = _lane_selectors(registry).get("d0")
    assert d0 is not None, "registry must declare the d0 lane"
    options = _nested_table(_load_toml(PYPROJECT_PATH), "tool", "pytest", "ini_options")
    testpaths = options.get("testpaths")
    assert isinstance(testpaths, list), "pyproject testpaths must be a list"
    assert testpaths == d0, f"pyproject testpaths {testpaths} must equal the d0 selectors {d0}"


def test_conda_manifests_exist_and_contain_no_machine_paths(environments):
    """Referenced manifests exist and record versions, not machine paths."""
    offenders = []
    for name, environment in environments.items():
        if environment.get("provision") != "conda":
            continue
        manifest = environment.get("manifest")
        if not isinstance(manifest, str):
            offenders.append((name, manifest))
            continue
        path = ROOT / manifest
        if not path.is_file():
            offenders.append((name, manifest))
            continue
        for key, value in _load_toml(path).items():
            if isinstance(value, str) and _looks_like_machine_path(value):
                offenders.append((name, key, value))
    assert not offenders, f"conda manifests missing or containing machine paths: {offenders}"
