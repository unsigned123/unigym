"""IsaacLab prop spawners: cfg-class resolution + spawn/actuator builders.

The isaaclab half of the old neutral ``unigym/props`` package.
Every isaaclab import stays inside :func:`_resolve`; the module body is
import-safe without the kit so unit tests can inject fake cfg classes.
"""

from __future__ import annotations

from typing import Any

from unigym.props.base import PropSpec


def _resolve(classes):
    """Return the isaaclab cfg-class table (lazy import when classes is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils  # noqa: F401  (ensures spawners loaded)
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.sim import schemas
    from isaaclab.sim.converters import UrdfConverterCfg

    # SphereCfg/UrdfFileCfg are re-exported at the spawners top level
    # (spawners/__init__ does ``from .shapes import *``); importing the names
    # directly avoids the shapes-module-vs-package ambiguity.
    from isaaclab.sim.spawners import (
        CapsuleCfg,
        CuboidCfg,
        CylinderCfg,
        SphereCfg,
        UrdfFileCfg,
        UsdFileCfg,
    )
    from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, RigidBodyMaterialCfg

    return {
        "UrdfFileCfg": UrdfFileCfg,
        "UsdFileCfg": UsdFileCfg,
        "RigidObjectCfg": RigidObjectCfg,
        "ArticulationCfg": ArticulationCfg,
        "AssetBaseCfg": AssetBaseCfg,
        "ImplicitActuatorCfg": ImplicitActuatorCfg,
        "UrdfConverterCfg": UrdfConverterCfg,
        "SphereCfg": SphereCfg,
        "CylinderCfg": CylinderCfg,
        "CapsuleCfg": CapsuleCfg,
        "CuboidCfg": CuboidCfg,
        "MassPropertiesCfg": schemas.MassPropertiesCfg,
        "RigidBodyPropertiesCfg": schemas.RigidBodyPropertiesCfg,
        "ArticulationRootPropertiesCfg": schemas.ArticulationRootPropertiesCfg,
        "CollisionPropertiesCfg": schemas.CollisionPropertiesCfg,
        "PreviewSurfaceCfg": PreviewSurfaceCfg,
        "RigidBodyMaterialCfg": RigidBodyMaterialCfg,
    }


def _build_urdf_spawn(spec: PropSpec, c: dict):
    """Build the spawn cfg (UrdfFileCfg) for a rigid_object/articulation prop.

    ``articulation``  is always anchored: ``fix_base=True`` is
    forced and a zero-gain joint drive is installed as the conversion-time
    baseline (per-joint behaviour comes from the declared actuators).
    """
    UrdfFileCfg = c["UrdfFileCfg"]
    spawn_cfg = dict(spec.spawn)
    asset_path = spec.asset if spec.type == "articulation" else spec.urdf_path
    merge_fixed = bool(spawn_cfg.get("merge_fixed_joints", False))
    fix_base = bool(spawn_cfg.get("fix_base", False))
    if spec.type == "articulation":
        fix_base = True  # anchored semantics (no freejoint in the asset)
    kwargs = dict(
        asset_path=asset_path,
        merge_fixed_joints=merge_fixed,
        fix_base=fix_base,
        collision_from_visuals=True,
        make_instanceable=True,
    )
    if spec.type == "articulation":
        JointDriveCfg = c["UrdfConverterCfg"].JointDriveCfg
        PDGainsCfg = JointDriveCfg.PDGainsCfg
        kwargs["joint_drive"] = JointDriveCfg(gains=PDGainsCfg(stiffness=0.0, damping=0.0))
    else:
        kwargs["joint_drive"] = None
    return UrdfFileCfg(**kwargs)


def _build_usd_spawn(spec: PropSpec, c: dict):
    """Build the spawn cfg (UsdFileCfg) for a ``type: usd`` prop .

    Native USD spawn -- the asset loads as authored, so materials/textures
    carried by the USD (or referenced by relative path) render without any
    URDF conversion step.  ``mass`` (spawn.mass) is applied via
    MassPropertiesCfg when declared.

    Visual-copy knobs (round-3 ha_open_door: render a door alongside its
    physics articulation, whose own spawner breaks the leaf materials):
    ``spawn.strip_articulation`` disables the USD-embedded ArticulationRoot,
    ``spawn.strip_collision`` disables collision, ``spawn.kinematic`` forces
    a kinematic gravity-free rigid body, ``spawn.visible`` (default True)
    toggles prim visibility.
    """
    UsdFileCfg = c["UsdFileCfg"]
    spawn_cfg = dict(spec.spawn)
    kwargs = dict(usd_path=spec.usd_path)  # isaaclab 2.3.2: usd_path (UrdfFileCfg uses asset_path)
    mass = spawn_cfg.get("mass")
    if mass is not None:
        kwargs["mass_props"] = c["MassPropertiesCfg"](mass=float(mass))
    rb = dict(spawn_cfg.get("rigid_body") or {})
    if spawn_cfg.get("kinematic"):
        rb["kinematic_enabled"] = True
        rb["disable_gravity"] = True
    # ALWAYS pass rigid_props: SIMPLE-authored USDs carry their rigid/collision
    # APIs on inner prims (e.g. /Meshes), so the reference root has no
    # RigidBodyAPI and RigidObject init fails without this.
    kwargs["rigid_props"] = c["RigidBodyPropertiesCfg"](**rb)
    if spawn_cfg.get("strip_articulation"):
        kwargs["articulation_props"] = c["ArticulationRootPropertiesCfg"](
            articulation_enabled=False
        )
    if spawn_cfg.get("strip_collision"):
        kwargs["collision_props"] = c["CollisionPropertiesCfg"](collision_enabled=False)
    if spawn_cfg.get("visible") is not None:
        kwargs["visible"] = bool(spawn_cfg["visible"])
    return UsdFileCfg(**kwargs)


def _build_actuators(spec: PropSpec, c: dict) -> dict:
    """Build implicit actuator cfgs for an articulated prop from spawn.joint_params."""
    ImplicitActuatorCfg = c["ImplicitActuatorCfg"]
    joint_params = dict(spec.spawn.get("joint_params", {}))
    actuators: dict[str, Any] = {}
    if joint_params:
        for jname, gains in joint_params.items():
            actuators[jname] = ImplicitActuatorCfg(
                joint_names_expr=[jname],
                stiffness=float(gains.get("stiffness", 0.0)),
                damping=float(gains.get("damping", 0.0)),
            )
    else:
        actuators["all"] = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)
    return actuators


__all__ = [
    "_build_actuators",
    "_build_urdf_spawn",
    "_build_usd_spawn",
    "_resolve",
]
