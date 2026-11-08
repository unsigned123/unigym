"""yaml randomization.events list -> Isaac Lab EventTermCfg builder.

The core logic (tupleization / asset_cfg conversion / function resolution) is
written in dependency-injection form so it is unit-testable locally without
isaaclab; ``build_events_cfg`` is the server-side wrapper.
"""
from __future__ import annotations


def _tupleize(v):
    """Recursively convert lists to tuples (the yaml -> EventTermCfg convention)."""
    if isinstance(v, list):
        return tuple(_tupleize(x) for x in v)
    if isinstance(v, dict):
        return {k: _tupleize(x) for k, x in v.items()}
    return v


def convert_params(params: dict, scene_entity_cfg_cls) -> dict:
    """Convert a yaml params dict to EventTermCfg-style params.

    * ``asset_cfg`` sub-dict -> ``SceneEntityCfg(**dict)``
    * every other list value (ranges, distribution params) -> tuple
      (nested dicts like velocity_range keep dict-of-tuples)
    """
    out = {}
    for k, v in params.items():
        if k == "asset_cfg" and isinstance(v, dict):
            out[k] = scene_entity_cfg_cls(**v)
        else:
            out[k] = _tupleize(v)
    return out


def resolve_func(name: str, mdp_module):
    """Resolve dotted attribute path against the mdp module."""
    obj = mdp_module
    for part in name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise ValueError(f"event func '{name}' resolved to non-callable {obj!r}")
    return obj


#: Upstream mdp functions judged "accumulate across resets" and therefore
#: deprecated -> replacement names.  Upstream ``randomize_rigid_body_com``
#: adds the sampled offset onto the *current* CoM (``coms[...] += rand``, see
#: its docstring); under mode=reset that accumulates across episodes into a
#: random walk (sigma ~= com_range * sqrt(n_resets)) and long runs push the
#: policy over.  Measured: the fr_net_recovery 2h collection crashed keep
#: from 0.85 to 0.04 for exactly this reason.  This repo's fixed
#: ``randomize_rigid_body_com_abs`` anchors on the default CoM, each reset an
#: independent bounded sample.
DEPRECATED_EVENT_FUNCS = {
    "randomize_rigid_body_com": "randomize_rigid_body_com_abs",
}


def build_event_terms(events_list: list, *, mdp_module, term_cfg_cls,
                      scene_entity_cfg_cls) -> list:
    """Build [(term_name, term_cfg), ...] from the yaml events list.

    term_cfg_cls / scene_entity_cfg_cls admit fake implementations for local
    unit tests.
    """
    if not isinstance(events_list, list):
        raise ValueError(f"randomization.events must be a list, got {type(events_list)}")
    terms = []
    for i, spec in enumerate(events_list):
        if "func" not in spec or "mode" not in spec:
            raise ValueError(f"event entry #{i} needs 'func' and 'mode', got {spec!r}")
        if spec["mode"] == "interval" and spec.get("interval_range_s") is None:
            raise ValueError(f"event entry #{i} mode='interval' needs interval_range_s")
        if spec["func"] in DEPRECATED_EVENT_FUNCS:
            raise ValueError(
                f"event entry #{i} uses deprecated func '{spec['func']}': it accumulates "
                f"CoM offsets across resets (random walk) and destabilizes long runs. "
                f"Use '{DEPRECATED_EVENT_FUNCS[spec['func']]}' instead"
                f" (dr_events.randomize_rigid_body_com_abs, anchored to default CoM).")
        func = resolve_func(spec["func"], mdp_module)
        params = convert_params(spec.get("params", {}), scene_entity_cfg_cls)
        kwargs = {"func": func, "mode": spec["mode"], "params": params}
        if spec.get("interval_range_s") is not None:
            kwargs["interval_range_s"] = tuple(spec["interval_range_s"])
        if "is_global_time" in spec:
            kwargs["is_global_time"] = bool(spec["is_global_time"])
        term = term_cfg_cls(**kwargs)
        terms.append((f"term_{i:02d}_{spec['func'].replace('.', '_')}", term))
    return terms


def randomize_rigid_body_com_abs(env, env_ids, com_range: dict, asset_cfg):
    """Center-of-mass randomization anchored to the DEFAULT CoM.

    Idempotent per reset.  Drop-in replacement for
    ``isaaclab.envs.mdp.randomize_rigid_body_com``,
    which ADDS the sampled offset to the CURRENT CoM — under ``mode: reset``
    that accumulates into a random walk across episodes (sigma grows as
    sqrt(n_resets) * com_range, e.g. ~17 cm drift after 100 resets with
    +-3 cm range) and progressively destabilizes the policy. This variant
    mirrors what     ``randomize_rigid_body_mass`` does: capture the pristine CoM once
    (cached on the asset as ``_dc_default_com`` — ArticulationData exposes
    no ``default_com``), then anchor every sample to it, so each episode
    gets an independent, bounded CoM displacement.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    lo = torch.tensor([r[0] for r in range_list], device="cpu")
    hi = torch.tensor([r[1] for r in range_list], device="cpu")
    rand = lo + (hi - lo) * torch.rand(len(env_ids), 1, 3, device="cpu")

    # Capture the pristine CoM on the first call (ArticulationData exposes
    # default_mass but no default_com, so we cache it on the asset itself).
    # The first call happens during env init, before any CoM randomization.
    default_com = getattr(asset, "_dc_default_com", None)
    coms = asset.root_physx_view.get_coms().clone()
    if default_com is None:
        default_com = coms.clone()
        asset._dc_default_com = default_com
    # anchor to the default CoM instead of the current (possibly randomized) one
    coms[env_ids[:, None], body_ids, :3] = (
        default_com[env_ids[:, None], body_ids, :3].clone() + rand
    )
    asset.root_physx_view.set_coms(coms, env_ids)


def build_events_cfg(events_list: list) -> dict:
    """Server-side wrapper: yaml list -> {term_name: EventTermCfg} dict.

    This isaaclab version exports no EventManagerCfg; ManagerBasedEnvCfg.events
    is typed ``object`` and the EventManager accepts a
    dict[str, EventTermCfg] config directly.

    Besides the regular isaaclab.envs.mdp functions, the fixed
    ``randomize_rigid_body_com_abs`` is registered too (see its docstring:
    the upstream implementation accumulates across resets).
    """
    import types

    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    mdp = types.SimpleNamespace(**vars(mdp))
    mdp.randomize_rigid_body_com_abs = randomize_rigid_body_com_abs
    return dict(build_event_terms(
        events_list, mdp_module=mdp, term_cfg_cls=EventTermCfg,
        scene_entity_cfg_cls=SceneEntityCfg))
