"""IsaacLab run-construction hooks (``BackendInfo`` faces).

The neutral ``unigym.run.build`` path calls these through the registry's
lazy factories; every isaaclab-specific import (kit, terrain, actuators,
DR events) stays inside this module / this package.
"""

from __future__ import annotations


def _reject_scene_params(env_cfg: dict) -> None:
    """Reject scene-provider/content routing keys at the isaaclab boundary."""
    params = env_cfg.get("params")
    if not isinstance(params, dict):
        return
    for key in ("scene_provider", "scene_provider_params", "scene_provider_preset", "content"):
        if key in params:
            raise ValueError(
                f"isaaclab backend: env.params.{key} has no consumer here; "
                "the run layer must reject a scene preset on this backend "
                "before construction"
            )


def env_kwargs(
    env_cfg: dict,
    artifacts_dir,
    world_base,
    spec=None,
    *,
    controls=None,
    profiles=None,
    robot_realizations=None,
    backend_options=None,
    actuation_plans=None,
) -> dict:
    """Builder kwargs ``IsaaclabEnv`` accepts.

    The isaaclab env takes neither ``backend_params`` nor ``world_base``
    (the world is the cloned scene); it takes the native ``terrain`` IR
    and the ``static_shapes`` declarations from the run spec.
    ``controls`` / ``profiles`` carry the neutral control runtimes and the
    per-articulation :class:`~unigym.robots.base.RobotProfile` objects;
    ``robot_realizations`` carries the backend-owned articulation configs
    built by this backend's robot realizer.  ``actuation_plans`` is accepted
    for the common run-layer contract and ignored here: D4 does not wire the
    generic Isaac Lab actuator plan into the G1 realizer (D5 does).
    """
    _reject_scene_params(env_cfg)
    del (env_cfg, artifacts_dir, world_base, actuation_plans)
    return {
        "terrain": getattr(spec, "terrain", None),
        "static_shapes": getattr(spec, "static_shapes", None),
        "articulations": getattr(spec, "articulations", None),
        "controls": controls,
        "profiles": profiles,
        "robot_realizations": robot_realizations,
        "backend_options": backend_options,
    }


def build_terrain(cfg: dict, num_envs: int, env_spacing: float, spec_terrain=None):
    """Build the yaml ``terrain:`` preset (native IR wins when declared)."""
    if spec_terrain is not None:
        return None
    if "terrain" in cfg:
        from unigym.backends.isaaclab2.terrains import (
            build_terrain as _build_terrain,
        )

        return _build_terrain(cfg["terrain"], num_envs=num_envs, env_spacing=env_spacing)
    raise ValueError(
        "isaaclab run declares no terrain: add a yaml 'terrain:' "
        "section or pass Run(world=World.terrain...)"
    )


def build_events(section):
    """Build the yaml ``randomization.events`` config."""
    from unigym.backends.isaaclab2.dr_events import build_events_cfg

    return build_events_cfg(section)


__all__ = [
    "build_events",
    "build_terrain",
    "env_kwargs",
]
