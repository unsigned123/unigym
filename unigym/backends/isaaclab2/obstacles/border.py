"""Corridor border walls for the obstacle course (solid, immovable).

Dynamically spawned URDF ``kinematic`` obstacles do NOT generate contact in
this cloned multi-env setup, and static (collision-only) instanced geometry
does not either (both verified empirically: walls were walked straight
through).  The variant that keeps a *real* collision with the robot is a
normal **dynamic rigid body**: it collides; to keep it practically immovable
we give it a 2000 kg mass plus very strong linear/angular damping.  A dog
hit transfers micro-motion and per-episode ``RigidObject`` reset restores the
baked spawn pose anyway.

This module builds one ``RigidObjectCfg`` per wall tile from a yaml section
``obstacles.borders``:

    borders:
      num_segments: 4      # wall tiles per side
      segment_length: 8.0  # tile length along +X (m)
      thickness: 0.2       # wall thickness (m)
      height: 1.0          # wall height (m)
      y_center: 2.1        # tile centre lateral offset (m) -> inner faces +-2.0
      x_start: 2.0         # first tile centre x (m)
      x_stride: 8.0        # tile pitch along +X (m)  (butt joints)

Every env gets a left (x, -y_center) and right (x, +y_center) tile per
segment, spanning x in [x_start - L/2, x_start + (n-1)*stride + L/2], i.e. a
4 m-wide corridor (y in [-2, 2]) with solid walls.  Borders are injected as
scene fields (not ``ObstacleManager`` specs), so they do not consume track
slots.
"""
from __future__ import annotations


def build_border_scene_fields(config: dict | None, *, classes=None) -> dict:
    """Build ``{field: RigidObjectCfg}`` for a yaml ``obstacles.borders`` block.

    Walls are spawned as *dynamic* (non-kinematic) rigid cuboids with a very
    large mass (2000 kg) and strong damping: unlike kinematic/static variants
    (which we verified generate no contact in this cloned setup), a normal
    dynamic rigid body does collide with the robot, yet its 2000 kg mass and
    damping make it immovable for all practical purposes.  ``RigidObject``
    reset RESTORES such a body to its baked spawn pose every episode, so no
    reset-time placement is needed.

    Returns an empty dict when ``config`` is falsy.  ``classes`` may inject a
    dict of isaaclab cfg classes for unit tests; when ``None`` the real ones
    are imported lazily.
    """
    if not config:
        return {}
    c = _resolve(classes)
    RigidObjectCfg, CuboidCfg, RigidBodyPropertiesCfg = (
        c["RigidObjectCfg"], c["CuboidCfg"], c["RigidBodyPropertiesCfg"])
    MassPropertiesCfg, CollisionPropertiesCfg = (
        c["MassPropertiesCfg"], c["CollisionPropertiesCfg"])

    n = int(config.get("num_segments", 4))
    length = float(config.get("segment_length", 8.0))
    thickness = float(config.get("thickness", 0.2))
    height = float(config.get("height", 1.0))
    y_center = float(config.get("y_center", 2.1))
    x_start = float(config.get("x_start", 2.0))
    stride = float(config.get("x_stride", length))

    fields = {}
    for i in range(n):
        x = x_start + i * stride
        for tag, y in (("l", -y_center), ("r", y_center)):
            name = f"bw_{tag}{i}"
            fields["border_" + name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Border_{name}",
                spawn=CuboidCfg(
                    size=(length, thickness, height),
                    collision_props=CollisionPropertiesCfg(collision_enabled=True),
                    rigid_props=RigidBodyPropertiesCfg(
                        linear_damping=1000.0,
                        angular_damping=1000.0,
                        max_linear_velocity=2.0,
                        max_angular_velocity=2.0,
                    ),
                    mass_props=MassPropertiesCfg(mass=2000.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x, y, height / 2.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                collision_group=0,
                debug_vis=False,
            )
    return fields


def border_geometry(config: dict | None) -> list[tuple[str, float, float, float]]:
    """Wall segments ``(name, local_x, local_y, world_z_centre)`` for probes.

    ``config`` is the raw yaml ``obstacles.borders`` block.  Used by tests and
    diagnostic scripts to locate walls without touching the scene.
    """
    if not config:
        return []
    n = int(config.get("num_segments", 4))
    length = float(config.get("segment_length", 8.0))
    height = float(config.get("height", 1.0))
    y_center = float(config.get("y_center", 2.1))
    x_start = float(config.get("x_start", 2.0))
    stride = float(config.get("x_stride", length))
    geom = []
    for i in range(n):
        x = x_start + i * stride
        for tag, y in (("l", -y_center), ("r", y_center)):
            geom.append((f"bw_{tag}{i}", x, y, height / 2.0))
    return geom


def _resolve(classes):
    """Isaac Lab cfg classes (lazy import when classes is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas.schemas_cfg import (
        CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg,
    )

    return {
        "RigidObjectCfg": RigidObjectCfg,
        "CuboidCfg": sim_utils.CuboidCfg,
        "RigidBodyPropertiesCfg": RigidBodyPropertiesCfg,
        "MassPropertiesCfg": MassPropertiesCfg,
        "CollisionPropertiesCfg": CollisionPropertiesCfg,
    }
