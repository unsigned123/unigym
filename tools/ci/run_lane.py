"""Execute one pytest lane declared in ``ci/compat.toml``.

The registry owns lane selectors and the environment that provisions them.
This runner turns one lane entry into its declared command and executes it with
``shell=False``.  It installs nothing; a ``no-sync`` uv environment is
provisioned out of band from its pinned manifest.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NoReturn, NotRequired, TypedDict

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path("ci/compat.toml")
SUPPORTED_VERSION = 1


class Lane(TypedDict):
    """One lane entry: the owning environment plus its test selectors.

    ``variant_of`` marks a versioned lane that reuses another lane's test
    ownership in a different environment; it is not a second owner.
    ``pytest_args`` are extra arguments passed verbatim to pytest.
    """

    environment: str
    selectors: list[str]
    pytest_args: NotRequired[list[str]]
    variant_of: NotRequired[str]


class Environment(TypedDict, total=False):
    """One environment entry from the registry.

    ``uv`` environments carry ``extras`` and an optional ``python`` version.
    A pre-provisioned uv environment sets ``no_sync`` instead of ``extras`` and
    may name a repo-relative pinned ``manifest``.  ``conda`` environments
    carry ``env_var``, ``manifest`` and ``strict``.
    """

    provision: str
    extras: list[str]
    python: str
    no_sync: bool
    env_var: str
    manifest: str
    strict: bool


class Registry(TypedDict):
    """The validated lane registry."""

    environments: dict[str, Environment]
    lanes: dict[str, Lane]


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"run_lane: {message}")


def _named_tables(document: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    """Return one named table per registry entry under ``key``."""
    value = document.get(key)
    if not isinstance(value, dict):
        _fail(f"{REGISTRY_PATH}: missing [{key}] table")
    tables: dict[str, dict[str, object]] = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            _fail(f"{REGISTRY_PATH}: [{key}] entries must be named tables")
        tables[name] = entry
    return tables


def _str_list(value: object, *, field: str) -> list[str]:
    """Return a non-empty list of non-empty strings, or fail fast."""
    if not isinstance(value, list) or not value:
        _fail(f"{REGISTRY_PATH}: {field} must be a non-empty list")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            _fail(f"{REGISTRY_PATH}: {field} must contain only non-empty strings")
        strings.append(item)
    return strings


def _parse_environment(name: str, table: dict[str, object]) -> Environment:
    """Validate and normalize one environment entry."""
    provision = table.get("provision")
    if not isinstance(provision, str) or provision not in {"uv", "conda"}:
        _fail(f"{REGISTRY_PATH}: environments.{name}.provision must be 'uv' or 'conda'")

    environment: Environment = {"provision": provision}
    if provision == "uv":
        no_sync = table.get("no_sync")
        if no_sync is not None and not isinstance(no_sync, bool):
            _fail(f"{REGISTRY_PATH}: environments.{name}.no_sync must be a boolean")
        if no_sync:
            if "extras" in table:
                _fail(
                    f"{REGISTRY_PATH}: environments.{name} cannot declare extras "
                    "together with no_sync"
                )
            environment["no_sync"] = True
        else:
            environment["extras"] = _str_list(
                table.get("extras"), field=f"environments.{name}.extras"
            )

        python = table.get("python")
        if python is not None:
            if not isinstance(python, str) or not python:
                _fail(f"{REGISTRY_PATH}: environments.{name}.python must be a non-empty string")
            environment["python"] = python

        manifest = table.get("manifest")
        if manifest is not None:
            if (
                not isinstance(manifest, str)
                or not manifest
                or Path(manifest).is_absolute()
                or ".." in Path(manifest).parts
            ):
                _fail(f"{REGISTRY_PATH}: environments.{name}.manifest must be a repo-relative path")
            if not (ROOT / manifest).is_file():
                _fail(f"{REGISTRY_PATH}: environments.{name}.manifest does not exist: {manifest}")
            environment["manifest"] = manifest
        return environment

    env_var = table.get("env_var")
    if not isinstance(env_var, str) or not env_var:
        _fail(f"{REGISTRY_PATH}: environments.{name}.env_var must be a non-empty string")

    manifest = table.get("manifest")
    if (
        not isinstance(manifest, str)
        or not manifest
        or Path(manifest).is_absolute()
        or ".." in Path(manifest).parts
    ):
        _fail(f"{REGISTRY_PATH}: environments.{name}.manifest must be a repo-relative path")

    strict = table.get("strict")
    if not isinstance(strict, bool):
        _fail(f"{REGISTRY_PATH}: environments.{name}.strict must be a boolean")

    environment["env_var"] = env_var
    environment["manifest"] = manifest
    environment["strict"] = strict
    return environment


def _parse_lane(name: str, table: dict[str, object]) -> Lane:
    """Validate and normalize one lane entry."""
    environment = table.get("environment")
    if not isinstance(environment, str) or not environment:
        _fail(f"{REGISTRY_PATH}: lanes.{name}.environment must be a non-empty string")
    selectors = _str_list(table.get("selectors"), field=f"lanes.{name}.selectors")
    lane: Lane = {"environment": environment, "selectors": selectors}
    pytest_args = table.get("pytest_args")
    if pytest_args is not None:
        lane["pytest_args"] = _str_list(pytest_args, field=f"lanes.{name}.pytest_args")
    variant_of = table.get("variant_of")
    if variant_of is not None:
        if not isinstance(variant_of, str) or not variant_of:
            _fail(f"{REGISTRY_PATH}: lanes.{name}.variant_of must be a non-empty string")
        lane["variant_of"] = variant_of
    return lane


def _load_registry() -> Registry:
    """Load and minimally validate ``ci/compat.toml``."""
    path = ROOT / REGISTRY_PATH
    try:
        with path.open("rb") as handle:
            raw: dict[str, object] = tomllib.load(handle)
    except FileNotFoundError:
        _fail(f"lane registry not found: {REGISTRY_PATH}")

    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SUPPORTED_VERSION:
        _fail(
            f"{REGISTRY_PATH}: unsupported registry version {version!r}; "
            f"this runner supports version {SUPPORTED_VERSION}"
        )

    environments = {
        name: _parse_environment(name, table)
        for name, table in _named_tables(raw, "environments").items()
    }
    lanes = {name: _parse_lane(name, table) for name, table in _named_tables(raw, "lanes").items()}
    if not environments or not lanes:
        _fail(f"{REGISTRY_PATH}: registry must declare at least one environment and lane")

    for name, lane in lanes.items():
        if lane["environment"] not in environments:
            _fail(
                f"{REGISTRY_PATH}: lane {name!r} references undeclared "
                f"environment {lane['environment']!r}"
            )

    return {"environments": environments, "lanes": lanes}


def _conda_executable() -> str:
    """Resolve conda, honoring the documented ``UNIGYM_CONDA_EXE`` override."""
    override = os.environ.get("UNIGYM_CONDA_EXE")
    if override:
        return override
    return shutil.which("conda") or "conda"


def _build_command(
    name: str, lane: Lane, environment: Environment, *, require_env: bool
) -> list[str]:
    """Build the exact argument vector for one lane."""
    provision = environment["provision"]
    if provision == "uv":
        command = ["uv", "run"]
        if environment.get("no_sync"):
            command.append("--no-sync")
        python = environment.get("python")
        if python is not None:
            command.extend(("--python", python))
        if environment.get("no_sync"):
            command.extend(("python", "-m", "pytest"))
        else:
            command.append("--frozen")
            for extra in environment["extras"]:
                command.extend(("--extra", extra))
            command.extend(("python", "-m", "pytest"))
    else:
        env_var = environment["env_var"]
        env_name = os.environ.get(env_var, "")
        if not env_name:
            if require_env:
                kind = "strict lane" if environment["strict"] else "lane"
                _fail(
                    f"lane {name!r} needs {env_var} to name its conda environment; "
                    f"the {kind} requires a provisioned environment"
                )
            env_name = f"${env_var}"
        command = [
            _conda_executable(),
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            "-m",
            "pytest",
        ]
    command.extend(lane.get("pytest_args", []))
    command.extend(lane["selectors"])
    command.append("-q")
    return command


def _format_command(command: list[str]) -> str:
    """Render an argument vector, keeping an unresolved env reference expandable."""
    rendered = [
        token if token.startswith("$") and token[1:].isidentifier() else shlex.quote(token)
        for token in command
    ]
    return " ".join(rendered)


def _print_lanes(registry: Registry) -> None:
    """Print every lane with its environment, provisioner and selectors."""
    for name, lane in registry["lanes"].items():
        environment_name = lane["environment"]
        environment = registry["environments"][environment_name]
        selectors = " ".join(lane["selectors"])
        details = ""
        if "python" in environment:
            details += f" python={environment['python']}"
        if environment.get("no_sync"):
            details += " no_sync=true"
        if "manifest" in environment:
            details += f" manifest={environment['manifest']}"
        if "variant_of" in lane:
            details += f" variant_of={lane['variant_of']}"
        if "pytest_args" in lane:
            details += f" pytest_args={lane['pytest_args']}"
        print(
            f"{name:<12} environment={environment_name} "
            f"provision={environment['provision']}{details} selectors={selectors}"
        )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run or describe one lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_lanes",
        help="print lanes, environments and selectors",
    )
    parser.add_argument("--lane", metavar="NAME", help="lane to run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact command without executing it",
    )
    args = parser.parse_args(argv)

    registry = _load_registry()

    if args.list_lanes:
        if args.lane is not None:
            parser.error("--list and --lane are mutually exclusive")
        _print_lanes(registry)
        return 0

    if args.lane is None:
        parser.error("one of --list or --lane NAME is required")

    lane = registry["lanes"].get(args.lane)
    if lane is None:
        _fail(f"unknown lane {args.lane!r}; declared lanes: " f"{', '.join(registry['lanes'])}")

    environment = registry["environments"][lane["environment"]]
    command = _build_command(args.lane, lane, environment, require_env=not args.dry_run)

    if args.dry_run:
        print(_format_command(command))
        return 0

    run_env = os.environ.copy()
    if environment.get("strict"):
        run_env["UNIGYM_TEST_ENV_STRICT"] = "1"
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False, env=run_env)
    except FileNotFoundError as exc:
        _fail(f"lane {args.lane!r} needs executable {command[0]!r} on PATH: {exc}")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
