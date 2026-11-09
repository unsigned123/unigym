"""IsaacLab run-construction hooks (``BackendInfo`` faces).

The neutral ``unigym.run.build`` path calls these through the registry's
lazy factories; every isaaclab-specific import (kit, terrain, actuators,
DR events) stays inside this module / this package.
"""

from __future__ import annotations


def env_kwargs(
    env_cfg: dict,
    artifacts_dir,
    world_base,
    spec=None,
    *,
    controls=None,
    recipes=None,
    backend_options=None,
) -> dict:
    """Builder kwargs ``IsaaclabEnv`` accepts.

    The isaaclab env takes neither ``backend_params`` nor ``world_base``
    (the world is the cloned scene); it takes the native ``terrain`` IR
    and the ``static_shapes`` declarations from the run spec.
    ``controls`` / ``recipes`` carry the neutral control runtimes and the
    per-articulation RobotRecipes.
    """
    del (env_cfg, artifacts_dir, world_base)
    return {
        "terrain": getattr(spec, "terrain", None),
        "static_shapes": getattr(spec, "static_shapes", None),
        "articulations": getattr(spec, "articulations", None),
        "controls": controls,
        "recipes": recipes,
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


def apply_actuator_override(profile, section) -> None:
    """Apply the yaml top-level ``actuator:`` override to the ArticulationCfg."""
    from unigym.backends.isaaclab2.actuators import build_actuator

    profile.articulation_cfg.actuators = {"legs": build_actuator(section)}


def build_events(section):
    """Build the yaml ``randomization.events`` config."""
    from unigym.backends.isaaclab2.dr_events import build_events_cfg

    return build_events_cfg(section)


__all__ = [
    "apply_actuator_override",
    "build_events",
    "build_terrain",
    "env_kwargs",
]
