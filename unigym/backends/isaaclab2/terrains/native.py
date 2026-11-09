"""Native ``Terrain`` IR -> Isaac Lab generator preset (S1b-iv).

The scene layer's :class:`~unigym.scenes.terrain.Terrain` is a pure height
field in metres.  Isaac Lab consumes terrain through
``TerrainImporterCfg(terrain_type="generator")``, whose sub-terrains are
``@height_field_to_mesh`` functions returning an integer height field in
*vertical-scale units*.  This preset bridges the two by resampling the IR
onto the generator's metric grid and wrapping that grid in a custom
sub-terrain (the ``custom:`` route, see :mod:`terrains.custom.gap`).

Design:

* one tile per environment (``num_rows=1``, ``num_cols=max(1, num_envs)``);
* ``size = terrain.size`` (metres);
* ``horizontal_scale = size[0] / (nx - 1)`` so the x samples are the IR's
  own grid pitch at its last index;
* the y grid is resampled to the same pitch (the IR may use a different
  y pitch; Isaac Lab has a single horizontal scale);
* ``vertical_scale`` defaults to 0.001 m.  Isaac Lab's height-field buffer
  is ``int16`` (``height_field_to_mesh`` allocates it), so the requested
  ``vertical_scale=1.0`` would quantize every sub-metre rise to whole
  metres.  Millimetres preserve the IR (0.05 m stairs) with a 1 mm floor;
  heights are returned in metres / ``vertical_scale`` units (rounded);
* all variants are ``kind="heightfield"`` -- stairs/slope are already
  sampled grids, so there is no separate mesh route;
* the returned terrain origin is ``(size_x/2, size_y/2, centre_surface_z)``.
  Isaac Lab's decorator derives that z from a centre window whose negative
  index wraps onto the zeroed one-pixel ring for tiles narrower than 2 m
  (origin z becomes 0); the preset computes the centre height explicitly.
"""

from __future__ import annotations

import numpy as np

from .custom._compat import height_field_to_mesh
from .base import TerrainPreset
from .generator import _make_custom_cfg

__all__ = ["NativeTerrainPreset"]

_DEFAULT_FRICTION = 0.8
_DEFAULT_COLOR = (0.5, 0.45, 0.4)
_DEFAULT_VERTICAL_SCALE = 0.001


def _resolve(classes):
    """Return the cfg-class table (real isaaclab when ``classes`` is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
    from isaaclab.terrains.height_field import HfTerrainBaseCfg

    return {
        "TerrainImporterCfg": TerrainImporterCfg,
        "TerrainGeneratorCfg": TerrainGeneratorCfg,
        "HfTerrainBaseCfg": HfTerrainBaseCfg,
        "RigidBodyMaterialCfg": sim_utils.RigidBodyMaterialCfg,
        "PreviewSurfaceCfg": sim_utils.PreviewSurfaceCfg,
    }


def _grid_axes(terrain):
    """IR sample coordinates (cell centres) along x and y (metres)."""
    nx, ny = terrain.resolution
    sx, sy = terrain.size
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * (sx / nx) - sx / 2.0
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * (sy / ny) - sy / 2.0
    return xs, ys


def _sample(terrain, xs, ys):
    """Bilinear sample of ``terrain.heights`` at ``(xs, ys)`` (metres).

    Coordinates outside the IR cell-centre span clamp to the edge value
    (the same "border holds its value" convention the IR grids use).
    """
    heights = np.asarray(terrain.heights, dtype=np.float64)
    ir_x, ir_y = _grid_axes(terrain)
    along_x = np.empty((heights.shape[0], xs.size), dtype=np.float64)
    for row in range(heights.shape[0]):
        along_x[row] = np.interp(xs, ir_x, heights[row])
    out = np.empty((xs.size, ys.size), dtype=np.float64)
    for col in range(xs.size):
        out[col] = np.interp(ys, ir_y, along_x[:, col])
    return out


def _full_field(terrain, horizontal_scale, vertical_scale):
    """Build the full ``(gx, gy)`` height field in vertical-scale units.

    ``gx``/``gy`` mirror :func:`height_field_to_mesh`'s full buffer shape
    (``int(size / horizontal_scale) + 1``), including the one-pixel ring
    that the decorator always zeroes (``border_pixels = 1`` when
    ``border_width == 0``).
    """
    sx, sy = terrain.size
    gx = int(sx / horizontal_scale) + 1
    gy = int(sy / horizontal_scale) + 1
    xs = np.arange(gx, dtype=np.float64) * horizontal_scale - sx / 2.0
    ys = np.arange(gy, dtype=np.float64) * horizontal_scale - sy / 2.0
    metres = _sample(terrain, xs, ys)
    units = np.rint(metres / vertical_scale)
    if units.size and (units.min() < -32768 or units.max() > 32767):
        raise ValueError(
            f"terrain {terrain.name!r}: heights exceed the int16 height-field "
            f"range at vertical_scale={vertical_scale}: "
            f"[{units.min()}, {units.max()}]"
        )
    full = units.astype(np.int16)
    full[0, :] = 0
    full[-1, :] = 0
    full[:, 0] = 0
    full[:, -1] = 0
    return full


def _centre_height(full, size, horizontal_scale, vertical_scale):
    """Height of the sampled field at the tile centre (0, 0), in metres."""
    sx, sy = size
    gx, gy = full.shape
    xs = np.arange(gx, dtype=np.float64) * horizontal_scale - sx * 0.5
    ys = np.arange(gy, dtype=np.float64) * horizontal_scale - sy * 0.5
    values = full.astype(np.float64)
    along_x = np.array([np.interp(0.0, xs, values[:, col]) for col in range(gy)])
    return float(np.interp(0.0, ys, along_x)) * vertical_scale


def _origin(full, size, horizontal_scale, vertical_scale):
    """Terrain origin: tile centre, with z on the sampled surface.

    Isaac Lab's decorator derives ``origin_z`` from a centre window; for a
    tile narrower than 2 m its ``x1`` index is negative and the slice wraps
    onto the decorator's zeroed border ring, yielding z=0.  The preset
    returns the centre height explicitly instead (the env spawn point).
    """
    sx, sy = size
    origin_z = _centre_height(full, size, horizontal_scale, vertical_scale)
    return (sx * 0.5, sy * 0.5, origin_z)


class NativeTerrainPreset(TerrainPreset):
    """One Isaac Lab terrain tile per env, sampled from a native ``Terrain``."""

    def __init__(
        self,
        terrain,
        friction: float = _DEFAULT_FRICTION,
        vertical_scale: float = _DEFAULT_VERTICAL_SCALE,
        color=_DEFAULT_COLOR,
    ):
        """Store the source IR and the preset parameters.

        Args:
            terrain: the :class:`~unigym.scenes.terrain.Terrain` to map.
            friction: static/dynamic terrain friction (default 0.8).
            vertical_scale: metres per height-field unit (default 0.001;
                Isaac Lab's int16 buffer cannot represent 1.0 m units).
            color: visual diffuse colour.
        """
        if terrain is None:
            raise ValueError("NativeTerrainPreset requires a Terrain")
        self.terrain = terrain
        self.friction = float(friction)
        self.vertical_scale = float(vertical_scale)
        if self.vertical_scale <= 0:
            raise ValueError(f"vertical_scale must be > 0, got {vertical_scale!r}")
        self.color = tuple(color)
        nx, _ = terrain.resolution
        self.horizontal_scale = terrain.size[0] / (nx - 1)

    def height_field(self):
        """Return ``(full_int16_grid, origin_xyz)`` used by the generator."""
        full = _full_field(self.terrain, self.horizontal_scale, self.vertical_scale)
        origin = _origin(full, self.terrain.size, self.horizontal_scale, self.vertical_scale)
        return full, origin

    def build(self, num_envs: int, env_spacing: float, *, classes=None, **kwargs) -> dict:
        """Build the native terrain preset (see ``TerrainPreset.build``).

        Returns:
            ``{"terrain": TerrainImporterCfg}`` over a one-row generator
            whose single ``native`` sub-terrain is the IR height field.
        """
        c = _resolve(classes)
        full, origin_xyz = self.height_field()
        interior = full[1:-1, 1:-1]
        if interior.size == 0:
            raise ValueError(
                f"terrain {self.terrain.name!r}: resolution "
                f"{self.terrain.resolution} is too coarse for the Isaac Lab "
                f"height-field preset (the decorator reserves a one-pixel "
                f"border; need at least 3 samples per axis)"
            )

        def native_height_field(difficulty: float, cfg) -> np.ndarray:
            return interior

        decorated = height_field_to_mesh(native_height_field)
        origin = np.asarray(origin_xyz, dtype=np.float64)

        def native_terrain(difficulty: float, cfg):
            result = decorated(difficulty, cfg)
            if isinstance(result, tuple):
                return result[0], origin.copy()
            return result

        sub_terrains = {
            "native": _make_custom_cfg(native_terrain, 1.0, {}, c),
        }
        gen_cfg = c["TerrainGeneratorCfg"](
            size=tuple(self.terrain.size),
            border_width=0.0,
            num_rows=1,
            num_cols=max(1, int(num_envs)),
            curriculum=False,
            use_cache=False,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            slope_threshold=None,
            sub_terrains=sub_terrains,
        )
        cfg = c["TerrainImporterCfg"](
            prim_path="/World/ground",
            num_envs=int(num_envs),
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
