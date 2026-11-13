"""Generic IsaacLab ``ArticulationCfg`` builder from neutral robot data.

This module is the asset-spawn and init-state seam for the generic robot
realizer.  It owns no per-robot dispatch and no G1 template: the neutral
:class:`~unigym.robots.base.RobotProfile` and
:class:`~unigym.scenes.articulation.ArticulationSpec` pair supply the asset
reference, the spawn authority, the default joint pose and the actuator-plan
callback.

The module body imports no IsaacLab symbol.  Callers either pass the cfg
classes through ``classes`` (offline construction) or let the builder import
the real classes lazily inside :func:`build_articulation_cfg`.

``spawn_options`` merges flat spawn-constructor props verbatim; the nested
``rigid_props``/``articulation_props`` mappings are constructed through the
injected :class:`RigidBodyPropertiesCfg` / :class:`ArticulationRootPropertiesCfg`
schema classes.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unigym._typing import JsonValue
from unigym.env.naming import ENV_REGEX_NS
from unigym.robots.base import RobotProfile
from unigym.scenes.articulation import ArticulationSpec

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.assets.plan import AssetPlan

__all__ = ["build_articulation_cfg"]

_CLASS_KEYS = (
    "ArticulationCfg",
    "UsdFileCfg",
    "UrdfFileCfg",
    "RigidBodyPropertiesCfg",
    "ArticulationRootPropertiesCfg",
)
"""The injected cfg-class keys the builder constructs."""

_NESTED_PROP_KEYS = ("rigid_props", "articulation_props")
"""Spawn-option keys whose mapping value is a schema cfg, not a flat prop."""

_SUPPORTED_FORMATS = ("usd", "urdf")
"""The asset formats this backend declares for robot articulations."""

_FIXED_ROOTS = ("fixed", "welded")
"""Root modes whose URDF spawn is base-fixed."""


@dataclass(frozen=True)
class _CfgClasses:
    """The validated cfg classes one builder call constructs."""

    articulation: Callable[..., object]
    init_state: Callable[..., object]
    usd: Callable[..., object]
    urdf: Callable[..., object]
    rigid_props: Callable[..., object]
    articulation_props: Callable[..., object]


def _resolve_classes(classes: Mapping[str, object] | None) -> _CfgClasses:
    """Return the cfg-class table, lazily importing the real classes."""
    if classes is None:
        from isaaclab.assets import ArticulationCfg
        from isaaclab.sim.schemas import (
            ArticulationRootPropertiesCfg,
            RigidBodyPropertiesCfg,
        )
        from isaaclab.sim.spawners import UrdfFileCfg, UsdFileCfg

        table: Mapping[str, object] = {
            "ArticulationCfg": ArticulationCfg,
            "UsdFileCfg": UsdFileCfg,
            "UrdfFileCfg": UrdfFileCfg,
            "RigidBodyPropertiesCfg": RigidBodyPropertiesCfg,
            "ArticulationRootPropertiesCfg": ArticulationRootPropertiesCfg,
        }
    else:
        table = classes

    missing = [key for key in _CLASS_KEYS if key not in table]
    if missing:
        raise ValueError(
            f"isaaclab2 robot builder: classes is missing {missing}; expected keys "
            f"{list(_CLASS_KEYS)}"
        )

    articulation = table["ArticulationCfg"]
    if not callable(articulation):
        raise TypeError(
            f"isaaclab2 robot builder: classes['ArticulationCfg'] must be a class, "
            f"got {articulation!r}"
        )
    init_state = getattr(articulation, "InitialStateCfg", None)
    if not callable(init_state):
        raise TypeError(
            f"isaaclab2 robot builder: classes['ArticulationCfg'].InitialStateCfg "
            f"must be a class, got {init_state!r}"
        )
    usd = table["UsdFileCfg"]
    if not callable(usd):
        raise TypeError(
            f"isaaclab2 robot builder: classes['UsdFileCfg'] must be a class, got {usd!r}"
        )
    urdf = table["UrdfFileCfg"]
    if not callable(urdf):
        raise TypeError(
            f"isaaclab2 robot builder: classes['UrdfFileCfg'] must be a class, got {urdf!r}"
        )
    rigid_props = table["RigidBodyPropertiesCfg"]
    if not callable(rigid_props):
        raise TypeError(
            f"isaaclab2 robot builder: classes['RigidBodyPropertiesCfg'] must be a "
            f"class, got {rigid_props!r}"
        )
    articulation_props = table["ArticulationRootPropertiesCfg"]
    if not callable(articulation_props):
        raise TypeError(
            f"isaaclab2 robot builder: classes['ArticulationRootPropertiesCfg'] must "
            f"be a class, got {articulation_props!r}"
        )
    return _CfgClasses(
        articulation=articulation,
        init_state=init_state,
        usd=usd,
        urdf=urdf,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    )


def _selected_asset_ref(
    profile: RobotProfile,
    spec: ArticulationSpec,
    asset_format: str,
    asset_plan: AssetPlan | None,
) -> str:
    """Resolve the asset reference the builder must spawn."""
    if asset_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"articulation {spec.name!r}: unsupported asset_format {asset_format!r}; "
            f"supported formats are {list(_SUPPORTED_FORMATS)}"
        )
    assets = spec.model if spec.model is not None else profile.assets
    if assets is None:
        raise ValueError(
            f"articulation {spec.name!r}: profile {profile.name!r} declares no asset "
            f"and the articulation declares no model"
        )
    ref = getattr(assets, asset_format)
    if not ref:
        raise ValueError(
            f"articulation {spec.name!r}: asset format {asset_format!r} has no "
            f"reference (assets.{asset_format}={ref!r})"
        )
    if asset_plan is not None:
        return asset_plan.resolve(ref)
    if ref.startswith("pack:") or "${" in ref or not os.path.isabs(ref):
        raise ValueError(
            f"articulation {spec.name!r}: asset reference {ref!r} needs an asset "
            f"plan; relative refs, pack: refs and ${{...}} placeholders resolve at "
            f"the composition boundary"
        )
    return ref


def _merge_spawn_options(
    spec: ArticulationSpec,
    base: dict[str, object],
    spawn_options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Merge the explicit spawn-option escape hatch into one spawn kwargs dict."""
    if spawn_options is None:
        return base
    if not isinstance(spawn_options, Mapping):
        raise TypeError(
            f"articulation {spec.name!r}: spawn_options must be a mapping, "
            f"got {spawn_options!r}"
        )
    duplicates = sorted(str(key) for key in set(spawn_options) & set(base))
    if duplicates:
        raise ValueError(
            f"articulation {spec.name!r}: spawn_options declares duplicate spawn "
            f"keys {duplicates}; those keys are owned by the builder"
        )
    merged = dict(base)
    merged.update(spawn_options)
    return merged


def _spawn_prop_cfg(classes: _CfgClasses, key: str, value: object) -> object:
    """Construct one nested schema prop from its mapping (non-mappings pass)."""
    if not isinstance(value, Mapping):
        return value
    cfg_cls = classes.rigid_props if key == "rigid_props" else classes.articulation_props
    return cfg_cls(**dict(value))


def _convert_spawn_options(
    classes: _CfgClasses,
    options: dict[str, object],
) -> dict[str, object]:
    """Convert nested prop mappings and keep every flat spawn prop verbatim."""
    converted = dict(options)
    for key in _NESTED_PROP_KEYS:
        if key in converted:
            converted[key] = _spawn_prop_cfg(classes, key, converted[key])
    return converted


def _spawn_cfg(
    classes: _CfgClasses,
    spec: ArticulationSpec,
    asset_format: str,
    ref: str,
    spawn_options: Mapping[str, object] | None,
) -> object:
    """Build the USD or URDF file spawn config for one asset reference."""
    if asset_format == "usd":
        base: dict[str, object] = {
            "usd_path": ref,
            "activate_contact_sensors": True,
        }
        merged = _merge_spawn_options(spec, base, spawn_options)
        return classes.usd(**_convert_spawn_options(classes, merged))
    base = {
        "asset_path": ref,
        "fix_base": spec.root in _FIXED_ROOTS,
        "merge_fixed_joints": False,
        "collision_from_visuals": False,
        "make_instanceable": True,
    }
    merged = _merge_spawn_options(spec, base, spawn_options)
    return classes.urdf(**_convert_spawn_options(classes, merged))


def _finite_number(value: object, spec: ArticulationSpec, field: str) -> float:
    """Validate one finite numeric value (bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"articulation {spec.name!r}: {field} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"articulation {spec.name!r}: {field} must be finite, got {value!r}")
    return number


def _finite_pos(value: object, spec: ArticulationSpec, field: str) -> tuple[float, float, float]:
    """Validate one explicit 3-vector spawn position (finite floats only)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise TypeError(f"articulation {spec.name!r}: {field} must be a 3-vector, got {value!r}")
    out = [_finite_number(entry, spec, f"{field}[{index}]") for index, entry in enumerate(value)]
    return (out[0], out[1], out[2])


def _spawn_pose(
    spec: ArticulationSpec,
    default_pos: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], float]:
    """The root position and yaw from ``spec.spawn`` or the explicit default."""
    spawn = dict(spec.spawn or {})
    if spawn:
        pos = _finite_pos(spawn["pos"], spec, "spawn.pos")
        return pos, _finite_number(spawn.get("yaw", 0.0), spec, "spawn.yaw")
    if default_pos is None:
        raise ValueError(
            f"articulation {spec.name!r}: spec declares no spawn and the builder "
            f"received no default_pos"
        )
    return _finite_pos(default_pos, spec, "default_pos"), 0.0


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """The (w, x, y, z) quaternion for one world-z yaw (radians)."""
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def build_articulation_cfg(
    profile: RobotProfile,
    *,
    spec: ArticulationSpec,
    params: Mapping[str, JsonValue],
    asset_format: str,
    actuator_plan: Callable[[RobotProfile, ArticulationSpec], Mapping[str, object]],
    asset_plan: AssetPlan | None = None,
    classes: Mapping[str, object] | None = None,
    default_pos: tuple[float, float, float] | None = None,
    spawn_options: Mapping[str, object] | None = None,
    collision_group: int = 0,
    debug_vis: bool = False,
    soft_joint_pos_limit_factor: float = 1.0,
) -> object:
    """Build one generic IsaacLab ``ArticulationCfg`` from neutral robot data.

    Args:
        profile: the neutral robot profile (default joint pose).
        spec: the run-level articulation declaration (spawn and asset authority).
        params: the robot section's factory params (``prim_path``).
        asset_format: the backend-selected asset format (``usd`` or ``urdf``).
        actuator_plan: the D4 seam producing the actuator-cfg mapping.
        asset_plan: optional composition-boundary asset resolver.
        classes: optional cfg-class table for offline construction; ``None``
            imports the real IsaacLab classes lazily.
        default_pos: explicit root position when ``spec.spawn`` is absent.
        spawn_options: spawn-constructor props merged over the builder's; flat
            keys pass verbatim and ``rigid_props``/``articulation_props``
            mappings become schema cfg objects.
        collision_group: the articulation's collision group.
        debug_vis: whether the articulation prim renders debug visuals.
        soft_joint_pos_limit_factor: the IsaacLab soft joint-limit scale.

    Raises:
        ValueError: A missing/unsupported asset format, a missing asset
            reference, an unresolved reference without an asset plan, a
            missing spawn default, a non-finite default position, a duplicate
            spawn option or a missing class key.
        TypeError: A non-callable actuator plan or cfg class, a non-mapping
            spawn options value or a non-string prim path.
    """
    if not callable(actuator_plan):
        raise TypeError(
            f"articulation {spec.name!r}: actuator_plan must be a callable, "
            f"got {actuator_plan!r}"
        )
    cfg_classes = _resolve_classes(classes)
    ref = _selected_asset_ref(profile, spec, asset_format, asset_plan)
    spawn = _spawn_cfg(cfg_classes, spec, asset_format, ref, spawn_options)
    pos, yaw = _spawn_pose(spec, default_pos)

    prim_path = params.get("prim_path", f"{ENV_REGEX_NS}/Robot")
    if not isinstance(prim_path, str):
        raise TypeError(
            f"articulation {spec.name!r}: param 'prim_path' must be a string, " f"got {prim_path!r}"
        )

    planned = actuator_plan(profile, spec)
    if not isinstance(planned, Mapping):
        raise TypeError(
            f"articulation {spec.name!r}: actuator_plan must return a mapping, " f"got {planned!r}"
        )
    return cfg_classes.articulation(
        prim_path=prim_path,
        spawn=spawn,
        init_state=cfg_classes.init_state(
            pos=pos,
            rot=_yaw_quaternion(yaw),
            joint_pos=dict(profile.default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators=dict(planned),
        collision_group=collision_group,
        debug_vis=debug_vis,
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    )
