"""IsaacLab 3 robot realizers: neutral ``RobotProfile`` -> ``ArticulationCfg``.

This package owns every IsaacLab-specific robot construction detail (asset
choice, prim paths, spawn conversion and the actuator-plan seam).  The neutral
``unigym.robots`` package never imports it.

There is no hardcoded backend identity here: callers pass the explicit
canonical backend (``isaaclab3_physx`` / ``isaaclab3_newton``) so the shared
library stays engine-neutral.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from unigym._typing import JsonValue
from unigym.robots.assets import select_backend_asset, select_robot_asset
from unigym.robots.base import RobotProfile

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from isaaclab.assets import ArticulationCfg

    from unigym.assets.plan import AssetPlan
    from unigym.scenes.articulation import ArticulationSpec

__all__ = ["articulation_cfg", "build_robot_realization", "realizer_for"]


def _param_float(params: Mapping[str, JsonValue], key: str, default: float) -> float:
    """Read one optional finite numeric realizer param."""
    value = params.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"IsaacLab 3 realizer param {key!r} must be numeric, got {value!r}")
    return float(value)


def _param_int(params: Mapping[str, JsonValue], key: str, default: int) -> int:
    """Read one optional numeric realizer param as an int."""
    value = params.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"IsaacLab 3 realizer param {key!r} must be numeric, got {value!r}")
    return int(value)


def _param_bool(params: Mapping[str, JsonValue], key: str, default: bool) -> bool:
    """Read one optional boolean realizer param."""
    value = params.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"IsaacLab 3 realizer param {key!r} must be a bool, got {value!r}")
    return value


def _param_mapping(params: Mapping[str, JsonValue], key: str) -> Mapping[str, object] | None:
    """Read one optional mapping realizer param."""
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"IsaacLab 3 realizer param {key!r} must be a mapping, got {value!r}")
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
    backend: str,
) -> object:
    """Build the IsaacLab 3 realization of one neutral robot profile.

    The realizer is generic: any effective profile with a USD or URDF asset
    and run-level ``actuation`` groups builds through the same path.  There is
    no ``profile.name`` dispatch.

    Args:
        profile: the neutral robot profile.
        spec: the run-level articulation declaration (spawn authority,
            effective asset override and actuation groups).
        params: the realizer-only robot-section params (``prim_path``,
            ``spawn_options``, ``collision_group``, ``debug_vis``,
            ``soft_joint_pos_limit_factor``).
        asset_plan: the composition boundary's asset resolver, forwarded to the
            generic builder.
        backend: the explicit canonical backend identity (``isaaclab3_physx``
            or ``isaaclab3_newton``) whose registry info supplies the asset
            formats.

    Raises:
        AssetRefError: The effective asset (``spec.model`` when set, otherwise
            ``profile.assets``) has no USD or URDF reference, or the profile's
            ``asset_backends`` does not allow this backend for the selected
            format.  A ``spec.model`` without a supported format never falls
            back to ``profile.assets``.
        ValueError: The articulation declares no run-level ``actuation``
            groups.
    """
    from unigym.actuators.ir import resolve_group_joints
    from unigym.capabilities.specs import backend_spec

    from ..actuators.plan import build_actuator_plan
    from .builder import build_articulation_cfg

    info = backend_spec(backend)
    if spec.model is None or spec.model == profile.assets:
        asset_format, _ = select_robot_asset(
            profile, backend=info.name, backend_formats=info.asset_formats
        )
    else:
        asset_format, _ = select_backend_asset(
            spec.model,
            asset_backends=profile.asset_backends,
            backend=info.name,
            backend_formats=info.asset_formats,
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
        asset_format=asset_format,
        actuator_plan=lambda _profile, _spec: build_actuator_plan(resolved, backend=info.name),
        asset_plan=asset_plan,
        default_pos=default_pos,
        spawn_options=_param_mapping(params, "spawn_options"),
        collision_group=_param_int(params, "collision_group", 0),
        debug_vis=_param_bool(params, "debug_vis", False),
        soft_joint_pos_limit_factor=_param_float(params, "soft_joint_pos_limit_factor", 1.0),
    )


def realizer_for(backend: str) -> Callable[..., object]:
    """Return the registry ``robot_realizer`` seam closure for one backend.

    The returned callable matches ``realize(profile, *, spec, params,
    asset_plan)`` and forwards the explicit canonical ``backend`` to
    :func:`build_robot_realization`.  Backend packages use it to expose their
    own ``robot_realizer`` factory without hardcoding an identity in shared
    code.
    """

    def realize(
        profile: RobotProfile,
        *,
        spec: ArticulationSpec,
        params: Mapping[str, JsonValue],
        asset_plan: AssetPlan | None = None,
    ) -> object:
        return build_robot_realization(
            profile,
            spec=spec,
            params=params,
            asset_plan=asset_plan,
            backend=backend,
        )

    return realize
