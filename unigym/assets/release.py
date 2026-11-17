"""Runtime resolution of ``release:<id>`` robot asset references.

A ``release:`` reference names a pinned asset published outside any pack
(e.g. the ``unigym-robots`` GitHub Release tarballs).  It is pure declaration
data: the reference is resolved to an absolute on-disk file only at the
composition boundary, through an explicit provider.

Provider discovery is deterministic and import-light:

1. an installed ``unigym.asset_release`` entry point is called with the asset
   id and must return a path;
2. otherwise the optional ``unigym_robots`` package's ``asset_path`` helper is
   used when importable.

There is no network access and no download here: the resolver reports the
installed path and fails with an actionable error when the asset tarball has
not been extracted yet.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable
from pathlib import Path

from unigym.assets.asset_ref import AssetRefError

__all__ = ["RELEASE_PREFIX", "is_release_ref", "resolve_release_ref"]

RELEASE_PREFIX = "release:"
"""The whole-string prefix of a release asset reference."""

_ENTRY_POINT_GROUP = "unigym.asset_release"


def is_release_ref(value: object) -> bool:
    """Whether *value* is a whole-string ``release:<id>`` reference."""
    return isinstance(value, str) and value.startswith(RELEASE_PREFIX)


def _entry_point_resolver() -> Callable[[str], object] | None:
    """The first installed ``unigym.asset_release`` resolver, if any."""
    for entry in importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP):
        try:
            target = entry.load()
        except Exception as err:  # noqa: BLE001 - provider import failures are reported
            raise AssetRefError(
                f"release asset resolver entry point {entry.name!r} failed to load: {err}"
            ) from err
        if callable(target):
            return target
    return None


def _robots_resolver() -> Callable[[str], object] | None:
    """The optional ``unigym_robots.asset_path`` fallback, when importable.

    A missing ``unigym_robots`` package (or a missing ``asset_path`` symbol)
    means the fallback is unavailable, so discovery continues to the teaching
    "install unigym-robots" error.  Any other import-time failure -- a broken
    dependency of an installed package, or a raising module body -- is
    reported as an :class:`AssetRefError`: silently treating it as "not
    installed" would send the caller chasing the wrong fix.
    """
    try:
        from unigym_robots import asset_path
    except ImportError as err:
        missing = err.name or ""
        if missing == "unigym_robots" or missing.startswith("unigym_robots."):
            return None
        raise AssetRefError(
            f"unigym_robots is installed but its resolver failed to import: "
            f"{err} | hint: repair the unigym-robots installation"
        ) from err
    except Exception as err:  # noqa: BLE001 - provider import failures are reported
        raise AssetRefError(f"unigym_robots failed to import: {err}") from err
    return asset_path


def _resolver() -> Callable[[str], object]:
    """The one release resolver for this process."""
    resolver = _entry_point_resolver() or _robots_resolver()
    if resolver is None:
        raise AssetRefError(
            "release: asset references need a resolver; install unigym-robots "
            "(or provide a 'unigym.asset_release' entry point) before building "
            "a profile that declares one"
        )
    return resolver


def resolve_release_ref(value: str) -> str:
    """Resolve one ``release:<id>`` reference to an existing absolute path.

    Raises:
        AssetRefError: The reference is malformed, no resolver is available,
            the resolver returns a non-path, or the asset file is absent
            (the message names the download command).
    """
    if not is_release_ref(value):
        raise AssetRefError(f"{value!r} is not a release: asset reference")
    asset_id = value[len(RELEASE_PREFIX) :]
    if not asset_id or asset_id != asset_id.strip():
        raise AssetRefError(
            f"release asset reference {value!r} must be 'release:<asset-id>' "
            "with a non-empty id and no surrounding whitespace"
        )
    resolver = _resolver()
    try:
        target = resolver(asset_id)
    except AssetRefError:
        raise
    except Exception as err:  # noqa: BLE001 - provider errors are reported as ref errors
        raise AssetRefError(f"release asset {asset_id!r} could not be resolved: {err}") from err
    if not isinstance(target, (str, os.PathLike)):
        raise AssetRefError(
            f"release asset {asset_id!r} resolver returned {type(target).__name__}, "
            "expected a filesystem path"
        )
    path = Path(target).expanduser()
    if not path.is_file():
        raise AssetRefError(
            f"release asset {asset_id!r} resolved to {path!s}, which is not an "
            "existing file; download it with `unigym-robots download --all` "
            "(or set UNIGYM_ROBOTS_ROOT to an installed asset root)"
        )
    return str(path.resolve())
