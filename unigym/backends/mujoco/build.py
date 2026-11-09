"""MuJoCo run-construction hooks (``BackendInfo`` faces).

The neutral ``unigym.run.build`` path calls these through the registry's
lazy factories; the backend-name knowledge lives here, not there.
"""
from __future__ import annotations


def env_kwargs(env_cfg: dict, artifacts_dir, world_base, spec=None, *,
               controls=None, recipes=None, backend_options=None) -> dict:
    """Builder kwargs ``MujocoEnv`` accepts (the mj-family shape + artifact dir).

    ``backend_params`` is the yaml ``env.params`` block and ``world_base``
    is the run-level world declaration.  The per-run scene
    artifact directory is mujoco-only wiring (never an ``env.params`` key).
    ``articulations`` is the neutral run spec's ordered list: the
    env composes entries declaring ``model.mjcf`` into the world base.
    ``controls`` / ``recipes`` carry the neutral control runtimes and the
    per-articulation RobotRecipes.
    """
    return {"backend_params": env_cfg.get("params") or {},
            "world_base": world_base,
            "artifact_dir": artifacts_dir,
            "articulations": getattr(spec, "articulations", None),
            "controls": controls,
            "recipes": recipes,
            "backend_options": backend_options}


def route_content(spec, env_cfg: dict) -> None:
    """Route an ``objects:`` content spec into ``env.params.content``.

    The mujoco scene build consumes it (robocasa provider scene fill).
    """
    env_cfg.setdefault("params", {})["content"] = spec
    provider = (spec.get("provider", "robocasa")
                if isinstance(spec, dict) else spec)
    print(f"[collect] content fill: provider={provider}", flush=True)


__all__ = ["env_kwargs", "route_content"]
