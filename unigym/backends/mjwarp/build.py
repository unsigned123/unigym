"""MuJoCo-Warp run-construction hooks (``BackendInfo`` faces).

The neutral ``unigym.run.build`` path calls these through the registry's
lazy factories; the backend-name knowledge lives here, not there.
"""

from __future__ import annotations


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
    """Builder kwargs for ``MjwarpEnv`` (legacy kwargs, byte-identical).

    ``artifact_dir`` is deliberately not passed: the historic mjwarp
    wiring left it at its default ``None`` and this face preserves that
    (the mujoco face passes it; see ``unigym.backends.mujoco.build``).
    ``articulations`` is the neutral run spec's ordered list.
    ``controls`` / ``profiles`` carry the neutral control runtimes and the
    per-articulation :class:`~unigym.robots.base.RobotProfile` objects.
    ``robot_realizations`` is not consumed: this backend builds no backend
    robot object and receives the neutral profile only.
    """
    del artifacts_dir
    del robot_realizations
    return {
        "backend_params": env_cfg.get("params") or {},
        "world_base": world_base,
        "articulations": getattr(spec, "articulations", None),
        "controls": controls,
        "profiles": profiles,
        "actuation_plans": actuation_plans,
        "backend_options": backend_options,
    }


__all__ = ["env_kwargs"]
