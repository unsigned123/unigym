"""Backend-aware asset selection for :class:`~unigym.robots.base.RobotProfile`.

A profile may carry one :class:`~unigym.assets.asset_ref.AssetRef` plus the
``asset_backends`` allow-list saying which backends may consume each declared
format.  This module combines those declarations with a backend's own format
preference order.  It never converts, falls back to an incompatible format or
touches the filesystem.

``asset_options`` stays metadata in this slice: :func:`asset_options_for`
exposes the selected format's mapping to a realizer's import boundary, and no
backend behavior is invented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from unigym._typing import JsonValue
from unigym.assets.asset_ref import ASSET_FORMATS, AssetRef, AssetRefError

from .base import RobotProfile

__all__ = ["asset_options_for", "select_backend_asset", "select_robot_asset"]


def _ordered_formats(backend_formats: Sequence[str]) -> tuple[str, ...]:
    """Validate a backend preference list with the AssetRef format rules."""
    if isinstance(backend_formats, (str, bytes)) or not isinstance(backend_formats, Sequence):
        raise AssetRefError(
            f"backend format preference list must be a sequence of formats, "
            f"got {backend_formats!r}"
        )
    ordered = tuple(str(value) for value in backend_formats)
    if not ordered:
        raise AssetRefError("asset format preference list must not be empty")
    unknown = sorted(set(ordered) - set(ASSET_FORMATS))
    if unknown:
        raise AssetRefError(
            f"asset format preference list declares unknown format(s) {unknown}; "
            f"allowed formats are {list(ASSET_FORMATS)}"
        )
    if len(set(ordered)) != len(ordered):
        raise AssetRefError(f"asset format preference list repeats a format: {list(ordered)}")
    return ordered


def _validated_backend(backend: str) -> str:
    """Return a non-empty backend name or raise a teaching error."""
    if not isinstance(backend, str) or not backend.strip():
        raise AssetRefError(f"backend must be a non-empty string, got {backend!r}")
    return backend


def _validated_restrictions(
    asset_backends: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    """Validate the ``asset_backends`` allow-list independently of RobotProfile.

    ``select_backend_asset`` is public API, so it cannot trust a profile-built
    mapping: a bare string must not substring-match a backend name, and empty
    or duplicate backend lists are configuration errors.
    """
    if not isinstance(asset_backends, Mapping):
        raise AssetRefError(f"asset_backends must be a mapping, got {asset_backends!r}")
    validated: dict[str, tuple[str, ...]] = {}
    for asset_format, backends in asset_backends.items():
        if asset_format not in ASSET_FORMATS:
            raise AssetRefError(
                f"asset_backends declares unknown format {asset_format!r}; "
                f"allowed formats are {list(ASSET_FORMATS)}"
            )
        if isinstance(backends, (str, bytes)) or not isinstance(backends, Sequence):
            raise AssetRefError(
                f"asset_backends[{asset_format!r}] must be a sequence of backend "
                f"names, got {backends!r}"
            )
        if not backends:
            raise AssetRefError(f"asset_backends[{asset_format!r}] must not be empty")
        names = tuple(backends)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise AssetRefError(
                f"asset_backends[{asset_format!r}] backend names must be "
                f"non-empty strings, got {names!r}"
            )
        if len(set(names)) != len(names):
            raise AssetRefError(
                f"asset_backends[{asset_format!r}] backend names must be unique, " f"got {names!r}"
            )
        validated[asset_format] = names
    return validated


def select_backend_asset(
    assets: AssetRef | None,
    *,
    asset_backends: Mapping[str, object],
    backend: str,
    backend_formats: Sequence[str],
) -> tuple[str, str]:
    """Select ``(format, resource)`` for one backend without a RobotProfile.

    This is the same rule as :func:`select_robot_asset`, for callers that hold
    a run-level ``AssetRef`` override while the profile still owns the
    per-format backend restrictions.

    Args:
        assets: the effective neutral asset reference.
        asset_backends: format -> backends allowed to use that format; an
            absent format is unrestricted beyond backend format support.
        backend: the backend name (``mujoco``, ``mjwarp``, ``isaaclab2`` ...).
        backend_formats: the backend's declared format preference order.

    Returns:
        The first backend-preferred format with a non-empty resource that the
        backend is allowed to use, and that resource.

    Raises:
        AssetRefError: The preference list or backend name is malformed, no
            asset reference is declared, or no declared format is both
            supported by the backend and allowed by ``asset_backends``.  The
            message names the declared formats, the backend formats and the
            per-format restrictions.
    """
    ordered = _ordered_formats(backend_formats)
    name = _validated_backend(backend)
    if assets is None:
        raise AssetRefError(f"no asset reference is declared; backend formats are {list(ordered)}")
    restrictions = _validated_restrictions(asset_backends)
    for asset_format in ordered:
        resource = getattr(assets, asset_format)
        if not resource:
            continue
        allowed = restrictions.get(asset_format)
        if allowed is not None and name not in allowed:
            continue
        return asset_format, str(resource)
    declared = [asset_format for asset_format in ASSET_FORMATS if getattr(assets, asset_format)]
    reported = {
        asset_format: allowed
        for asset_format, allowed in restrictions.items()
        if getattr(assets, asset_format)
    }
    raise AssetRefError(
        f"asset reference declares format(s) {declared}, the backend {name!r} "
        f"supports {list(ordered)}, and per-format backend restrictions are "
        f"{reported}; no compatible format"
    )


def select_robot_asset(
    profile: RobotProfile,
    *,
    backend: str,
    backend_formats: Sequence[str],
) -> tuple[str, str]:
    """Return ``(format, resource)`` for this backend or raise ``AssetRefError``.

    The backend's preference order decides which declared format wins; a
    format missing a resource is skipped, and a format restricted by
    ``profile.asset_backends`` is skipped unless it names *backend*.  There is
    no fallback to an incompatible format and no conversion.
    """
    return select_backend_asset(
        profile.assets,
        asset_backends=profile.asset_backends,
        backend=backend,
        backend_formats=backend_formats,
    )


def asset_options_for(profile: RobotProfile, asset_format: str) -> Mapping[str, JsonValue]:
    """Return the selected format's ``asset_options`` mapping (metadata only).

    An absent mapping is the empty mapping.  Core does not interpret the
    values; a backend realizer may forward them to its own import boundary.
    """
    if asset_format not in ASSET_FORMATS:
        raise AssetRefError(
            f"unknown asset format {asset_format!r}; allowed formats are " f"{list(ASSET_FORMATS)}"
        )
    return profile.asset_options.get(asset_format, {})
