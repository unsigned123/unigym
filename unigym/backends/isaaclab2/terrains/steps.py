"""Pyramid-stairs mesh terrain preset (Isaac Lab TerrainImporterCfg).

Uses Isaac Lab's built-in ``MeshPyramidStairsTerrainCfg`` sub-terrain generator
(function ``pyramid_stairs_terrain``) wrapped in a ``TerrainGeneratorCfg`` +
``TerrainImporterCfg``.  Each parallel environment gets its own pyramid
sub-terrain (``num_cols == num_envs``) so spawn origins never collide; robots
are placed on the central platform top via ``use_terrain_origins=True``.

The sub-terrain size is derived from the requested ``num_steps`` so the user
gets exactly that many stair rings around the platform:
    side = (num_steps - 1) * 2 * step_width + platform_width

Platform-top height = (num_steps + 1) * step_height; IsaaclabEnv's robot
spawn_height (e.g. 0.34 m) is added on top of this origin z by the scene, so
the robot stands on the platform surface.
"""

from __future__ import annotations

from .base import TerrainPreset


def _resolve(classes):
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

    return {
        "TerrainImporterCfg": TerrainImporterCfg,
        "TerrainGeneratorCfg": TerrainGeneratorCfg,
        "MeshPyramidStairsTerrainCfg": terrain_gen.MeshPyramidStairsTerrainCfg,
        "RigidBodyMaterialCfg": sim_utils.RigidBodyMaterialCfg,
        "PreviewSurfaceCfg": sim_utils.PreviewSurfaceCfg,
    }


class StepTerrain(TerrainPreset):
    """Pyramid-stairs terrain with configurable step geometry."""

    def __init__(
        self,
        step_width: float = 0.3,
        step_height: float = 0.08,
        num_steps: int = 8,
        platform_width: float = 1.0,
        border_width: float = 0.5,
        friction: float = 0.8,
        color=(0.5, 0.45, 0.4),
    ):
        self.step_width = step_width
        self.step_height = step_height
        self.num_steps = num_steps
        self.platform_width = platform_width
        self.border_width = border_width
        self.friction = friction
        self.color = color

    def build(self, num_envs: int, env_spacing: float, *, classes=None, **kwargs) -> dict:
        """Build the stair-steps preset (see TerrainPreset.build).

        Returns:
            ``{"terrain": TerrainImporterCfg}`` over a single-cell
            TerrainGeneratorCfg holding the custom steps function.
        """
        c = _resolve(classes)
        # sub-terrain internal border kept at 0 so steps reach the cell edge;
        # the outer flat border is the TerrainGeneratorCfg.border_width.
        sub_border = 0.0
        side = (self.num_steps - 1) * 2 * self.step_width + 2 * sub_border + self.platform_width
        gen_cfg = c["TerrainGeneratorCfg"](
            size=(side, side),
            border_width=self.border_width,
            num_rows=1,
            num_cols=max(1, num_envs),
            curriculum=False,
            use_cache=False,
            sub_terrains={
                "steps": c["MeshPyramidStairsTerrainCfg"](
                    proportion=1.0,
                    step_height_range=(self.step_height, self.step_height),
                    step_width=self.step_width,
                    platform_width=self.platform_width,
                    border_width=sub_border,
                ),
            },
        )
        cfg = c["TerrainImporterCfg"](
            prim_path="/World/ground",
            num_envs=num_envs,
            terrain_type="generator",
            terrain_generator=gen_cfg,
            use_terrain_origins=True,
            max_init_terrain_level=0,
            env_spacing=env_spacing,
            physics_material=c["RigidBodyMaterialCfg"](
                static_friction=self.friction,
                dynamic_friction=self.friction,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=c["PreviewSurfaceCfg"](diffuse_color=self.color),
            debug_vis=False,
        )
        return {"terrain": cfg}
