"""MuJoCo-Warp run-construction hooks (``BackendInfo`` faces).

The neutral ``unigym.run.build`` path calls these through the registry's
lazy factories; the backend-name knowledge lives here, not there.
"""
from __future__ import annotations


def env_kwargs(env_cfg: dict, artifacts_dir, world_base, spec=None, *,
               controls=None, recipes=None, backend_options=None) -> dict:
    """Builder kwargs for ``MjwarpEnv`` (legacy kwargs, byte-identical).

    ``artifact_dir`` is deliberately not passed: the historic mjwarp
    wiring left it at its default ``None`` and this face preserves that
    (the mujoco face passes it; see ``unigym.backends.mujoco.build``).
    ``articulations`` is the neutral run spec's ordered list.
    ``controls`` / ``recipes`` carry the neutral control runtimes and the
    per-articulation RobotRecipes.
    """
    del artifacts_dir
    return {"backend_params": env_cfg.get("params") or {},
            "world_base": world_base,
            "articulations": getattr(spec, "articulations", None),
            "controls": controls,
            "recipes": recipes,
            "backend_options": backend_options}


__all__ = ["env_kwargs"]
