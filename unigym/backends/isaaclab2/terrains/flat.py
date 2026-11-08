"""Flat ground-plane terrain preset.

Migrates the verified ``flat_plane()`` construction (AssetBaseCfg +
GroundPlaneCfg + RigidBodyMaterialCfg) into the pluggable class interface.
"""
from __future__ import annotations

from .base import TerrainPreset


def _resolve(classes):
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    return {
        "AssetBaseCfg": AssetBaseCfg,
        "GroundPlaneCfg": sim_utils.GroundPlaneCfg,
        "RigidBodyMaterialCfg": sim_utils.RigidBodyMaterialCfg,
    }


class FlatTerrain(TerrainPreset):
    """Flat infinite ground plane with configurable friction.

    ``visible=False`` keeps the collision plane but hides the grid render —
    for scenes whose USD stage provides its own floor (e.g. the HA digital
    twin room), where a visible default plane z-fights with / paints over
    the stage floor.
    """

    def __init__(self, friction: float = 1.0, color=(0.4, 0.4, 0.4),
                 visible: bool = True):
        self.friction = friction
        self.color = color
        self.visible = visible

    def build(self, num_envs: int, env_spacing: float, *,
              classes=None, **kwargs) -> dict:
        """Build the flat-ground preset (see TerrainPreset.build).

        Returns:
            ``{"ground": AssetBaseCfg(GroundPlaneCfg)}``.
        """
        c = _resolve(classes)
        return {"ground": c["AssetBaseCfg"](
            prim_path="/World/ground",
            spawn=c["GroundPlaneCfg"](
                color=self.color,
                visible=self.visible,
                physics_material=c["RigidBodyMaterialCfg"](
                    static_friction=self.friction,
                    dynamic_friction=self.friction,
                    restitution=0.0,
                ),
            ),
        )}
