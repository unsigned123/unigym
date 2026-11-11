"""Run config references: absolute file paths or exact ``runs`` catalog ids.

The resolver is shared by the CLI and the daemon so both faces accept exactly
the same two forms:

```text
/absolute/path/to/run.yaml     -> used verbatim (must be a file)
acme.demo                      -> exact PackRuntime "runs" catalog id
```

There is no implicit ``config/`` directory, no CWD-relative path, no glob
expansion and no repository fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from unigym.packs import PackError

if TYPE_CHECKING:
    from unigym.packs.runtime import PackRuntime

__all__ = ["RunConfigRefError", "resolve_run_config"]

_GLOB_CHARS = ("*", "?", "[")
_ACCEPTED_FORMS = (
    "an absolute file path (e.g. /path/to/run.yaml) or an exact "
    "'runs' pack catalog id (e.g. acme.demo)"
)


class RunConfigRefError(ValueError):
    """A run config reference is not one of the two accepted forms."""


def _has_glob(spec: str) -> bool:
    return any(char in spec for char in _GLOB_CHARS)


def _looks_like_path(spec: str) -> bool:
    return (
        "/" in spec
        or "\\" in spec
        or spec in (".", "..")
        or spec.lower().endswith((".yaml", ".yml"))
    )


def _rejected(spec: str, reason: str | None = None) -> RunConfigRefError:
    message = f"run config ref {spec!r} is not supported"
    if reason is not None:
        message += f": {reason}"
    return RunConfigRefError(f"{message}; use {_ACCEPTED_FORMS}")


def _require_file(path: Path, spec: str) -> Path:
    if not path.is_file():
        raise RunConfigRefError(f"run config ref {spec!r} resolved to {path}, which is not a file")
    return path


def resolve_run_config(spec: str, runtime: PackRuntime | None = None) -> Path:
    """Resolve *spec* to one run config file.

    Args:
        spec: An absolute path, a ``~`` path, or an exact ``runs`` catalog id.
        runtime: The pack runtime that owns the ``runs`` catalog.  Required for
            the catalog-id form; an absolute path never needs one.

    Returns:
        The resolved absolute config path.

    Raises:
        RunConfigRefError: *spec* is empty or not a string, is a relative path
            or glob, names a catalog id without a runtime, or resolves to
            something that is not a file.
        PackError: *spec* is an unknown ``runs`` catalog id; the error names the
            id and the known ids.
    """
    if not isinstance(spec, str):
        raise RunConfigRefError(
            f"run config ref must be a string, got {type(spec).__name__}; " f"use {_ACCEPTED_FORMS}"
        )
    if not spec.strip():
        raise RunConfigRefError(f"run config ref must be a non-empty string; use {_ACCEPTED_FORMS}")

    path = Path(spec).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        if _has_glob(spec):
            raise _rejected(spec, "globs are not expanded")
        raise RunConfigRefError(
            f"run config ref {spec!r} is not a file: {path}; use {_ACCEPTED_FORMS}"
        )

    if _has_glob(spec):
        raise _rejected(spec, "globs are not expanded")
    if runtime is None:
        raise RunConfigRefError(
            f"run config ref {spec!r} is not an absolute path and the catalog "
            f"id form requires a pack runtime; use {_ACCEPTED_FORMS}"
        )

    try:
        path = runtime.resource_for("runs", spec)
    except PackError as error:
        if _looks_like_path(spec):
            raise _rejected(spec, "relative paths are not resolved") from error
        raise
    return _require_file(path, spec)
