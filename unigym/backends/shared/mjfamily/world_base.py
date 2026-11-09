"""Resolve the run-level ``world_base`` declaration to a MJCF path.

The world asset is no longer read from ``env.params.mj_model``; it is
declared by the run config's top-level ``world_base: {mjcf: <path>}``
(scene bundles declare the same key under ``backend.mujoco.world_base``).
The backend constructors receive that declaration as an explicit
``world_base`` keyword and resolve it here.

There is deliberately **no default model and no alias**: an absent or
malformed declaration fails fast with a teaching error, and the legacy
``env.params.mj_model`` key is rejected at every door.
"""
from __future__ import annotations

from collections.abc import Mapping

from unigym.errors import ConfigError


def resolve_world_base_mjcf(world_base) -> str:
    """Return the MJCF path declared by ``world_base``.

    Args:
        world_base: ``{mjcf: <path>}`` (the config shape) or a bare path
            string (the direct-constructor convenience).

    Returns:
        The declared MJCF path string.

    Raises:
        ConfigError: The declaration is absent or malformed.  There is no
            implicit default model.
    """
    if world_base is None:
        raise ConfigError(
            "world_base is required: declare `world_base: {mjcf: <path>}` "
            "at the run config top level (or "
            "`backend.mujoco.world_base` in a scene bundle). "
            "env.params.mj_model was removed and has no fallback")
    if isinstance(world_base, Mapping):
        mjcf = world_base.get("mjcf")
        if mjcf is None or str(mjcf) == "":
            raise ConfigError(
                "world_base must declare `mjcf: <path>`; got keys "
                f"{sorted(map(str, world_base))}")
        return str(mjcf)
    if isinstance(world_base, str):
        if not world_base:
            raise ConfigError("world_base path must be a non-empty string")
        return world_base
    raise ConfigError(
        "world_base must be a path string or a mapping {mjcf: <path>}, "
        f"got {type(world_base).__name__}")


def reject_legacy_mj_model(params, *, source: str = "env.params") -> None:
    """Fail fast when the removed ``mj_model`` key is still declared.

    Args:
        params: The backend params mapping (``env.params``).
        source: Where the mapping came from (for the teaching message).

    Raises:
        ConfigError: ``mj_model`` is present; the message names
            ``world_base`` as the replacement.
    """
    if isinstance(params, Mapping) and "mj_model" in params:
        raise ConfigError(
            f"{source}.mj_model was removed in v101 A1; declare the "
            "run-level `world_base: {mjcf: <path>}` instead "
            "(scene bundles: `backend.mujoco.world_base`) -- there is no "
            "alias and no fallback")
