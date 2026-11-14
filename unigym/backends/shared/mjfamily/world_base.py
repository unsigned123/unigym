"""Resolve the run-level ``world_base`` declaration to a MJCF path.

The world asset is no longer read from ``env.params.mj_model``; it is
declared by the run config's top-level ``world_base: {mjcf: <path>}`` (a
scene preset declares the same asset under its top-level ``world:`` block).
The backend constructors receive that declaration as an explicit
``world_base`` keyword and resolve it here.

There is deliberately **no default model and no alias**: an absent or
malformed declaration fails fast with a teaching error, and the legacy
``env.params.mj_model`` / ``env.params.mj_scene`` keys are rejected at
every door.
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
            "at the run config top level (or a scene preset's top-level "
            "`world:` block). env.params.mj_model was removed and has no "
            "fallback"
        )
    if isinstance(world_base, Mapping):
        mjcf = world_base.get("mjcf")
        if mjcf is None or str(mjcf) == "":
            raise ConfigError(
                "world_base must declare `mjcf: <path>`; got keys "
                f"{sorted(map(str, world_base))}"
            )
        return str(mjcf)
    if isinstance(world_base, str):
        if not world_base:
            raise ConfigError("world_base path must be a non-empty string")
        return world_base
    raise ConfigError(
        "world_base must be a path string or a mapping {mjcf: <path>}, "
        f"got {type(world_base).__name__}"
    )


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
            f"{source}.mj_model was removed; declare the run-level "
            "`world_base: {mjcf: <path>}` instead (or a scene preset's "
            "top-level `world:` block) -- there is no alias and no fallback"
        )


def reject_legacy_scene_params(params, *, source: str = "env.params") -> None:
    """Fail fast when the removed scene-routing keys are still declared.

    ``mj_scene`` / ``mj_scene_params`` were the pre-preset scene provider
    channel.  A backend constructor now composes the scene from
    ``env.params.scene_provider``, which only :func:`apply_scene_preset`
    writes; a direct declaration is dead and must fail instead of being
    silently ignored.

    Args:
        params: The backend params mapping (``env.params``).
        source: Where the mapping came from (for the teaching message).

    Raises:
        ConfigError: A removed scene key is present; the message names the
            ``scene_preset`` replacement.
    """
    if not isinstance(params, Mapping):
        return
    for key in ("mj_scene", "mj_scene_params"):
        if key in params:
            raise ConfigError(
                f"{source}.{key} was removed; declare the run-level "
                "'scene_preset:' (catalog id or inline provider) instead -- "
                "there is no alias and no fallback"
            )
