"""IsaacLab robot realizers: neutral ``RobotProfile`` -> ``ArticulationCfg``.

This package owns every isaaclab-specific robot construction detail (asset
choice, prim paths, spawn conversion and the actuator-plan seam).  The neutral
``unigym.robots`` package never imports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from unigym._typing import JsonValue
from unigym.robots.base import RobotProfile

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from isaaclab.assets import ArticulationCfg

    from unigym.assets.plan import AssetPlan
    from unigym.scenes.articulation import ArticulationSpec

__all__ = ["articulation_cfg", "build_robot_realization"]


def _param_float(params: Mapping[str, JsonValue], key: str, default: float) -> float:
    """Read one optional finite numeric realizer param."""
    value = params.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"isaaclab2 realizer param {key!r} must be numeric, got {value!r}")
    return float(value)


def _param_int(params: Mapping[str, JsonValue], key: str, default: int) -> int:
    """Read one optional numeric realizer param as an int."""
    value = params.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"isaaclab2 realizer param {key!r} must be numeric, got {value!r}")
    return int(value)


def _param_bool(params: Mapping[str, JsonValue], key: str, default: bool) -> bool:
    """Read one optional boolean realizer param."""
    value = params.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"isaaclab2 realizer param {key!r} must be a bool, got {value!r}")
    return value


def _param_mapping(params: Mapping[str, JsonValue], key: str) -> Mapping[str, object] | None:
    """Read one optional mapping realizer param."""
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"isaaclab2 realizer param {key!r} must be a mapping, got {value!r}")
    return value


def articulation_cfg(realizations: Mapping[str, object] | None, name: str) -> ArticulationCfg:
    """The backend realization of one articulation (fail-fast when absent)."""
    from isaaclab.assets import ArticulationCfg

    value = dict(realizations or {}).get(name)
    if not isinstance(value, ArticulationCfg):
        raise ValueError(
            f"isaaclab backend: articulation {name!r} carries no "
            f"ArticulationCfg realization (the backend robot realizer must "
            f"build one)"
        )
    return value


def build_robot_realization(
    profile: RobotProfile,
    *,
    spec: ArticulationSpec,
    params: Mapping[str, JsonValue],
    asset_plan: AssetPlan | None = None,
) -> object:
    """Build the isaaclab realization of one neutral robot profile.

    The realizer is generic: any effective profile with a USD asset and
    run-level ``actuation`` groups builds through the same path.  There is no
    ``profile.name`` dispatch.

    Args:
        profile: the neutral robot profile.
        spec: the run-level articulation declaration (spawn authority,
            effective asset override and actuation groups).
        params: the realizer-only robot-section params (``prim_path``,
            ``spawn_options``, ``collision_group``, ``debug_vis``,
            ``soft_joint_pos_limit_factor``).
        asset_plan: the composition boundary's asset resolver, forwarded to the
            generic builder.

    Raises:
        NotImplementedError: The effective asset (``spec.model`` when set,
            otherwise ``profile.assets``) carries no USD reference.  A
            ``spec.model`` without USD never falls back to ``profile.assets``.
        ValueError: The articulation declares no run-level ``actuation``
            groups.
    """
    from unigym.actuators.ir import resolve_group_joints

    from ..actuators.plan import build_actuator_plan
    from .builder import build_articulation_cfg

    assets = spec.model if spec.model is not None else profile.assets
    if assets is None or not assets.usd:
        source = "spec.model" if spec.model is not None else "profile.assets"
        raise NotImplementedError(
            f"isaaclab2 backend: articulation {spec.name!r} has no effective "
            f"USD asset ({source} carries no `usd` reference; spec.model does "
            f"not fall back to profile.assets) | hint: declare a run-level "
            f"`model: {{usd: <path>}}` or give the neutral profile a USD "
            f"asset; the isaaclab builder spawns USD only"
        )
    if not spec.actuation:
        raise ValueError(
            f"articulation {spec.name!r}: the IsaacLab realizer requires "
            f"explicit run-level `actuation` groups; the actuator source is "
            f"the run-level ActuationGroup set (no hardcoded table)"
        )
    resolved = resolve_group_joints(
        spec.actuation,
        articulation=spec.name,
        joint_sets=profile.joint_sets,
    )
    base_height = profile.base_height
    default_pos = (0.0, 0.0, base_height) if base_height is not None else None
    return build_articulation_cfg(
        profile,
        spec=spec,
        params=params,
        asset_format="usd",
        actuator_plan=lambda _profile, _spec: build_actuator_plan(resolved),
        asset_plan=asset_plan,
        default_pos=default_pos,
        spawn_options=_param_mapping(params, "spawn_options"),
        collision_group=_param_int(params, "collision_group", 0),
        debug_vis=_param_bool(params, "debug_vis", False),
        soft_joint_pos_limit_factor=_param_float(params, "soft_joint_pos_limit_factor", 1.0),
    )
