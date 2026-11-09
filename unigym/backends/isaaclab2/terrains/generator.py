"""Generator terrain preset (Route A + Route C hybrid).

Wraps Isaac Lab's :class:`TerrainGeneratorCfg` + :class:`TerrainImporterCfg`
and exposes every built-in height-field / trimesh sub-terrain generator
through a string registry (Route A).  Sub-terrain entries whose ``type`` begins
with ``custom:`` resolve to an arbitrary ``@height_field_to_mesh``-decorated
Python function (Route C), letting user-defined height-field generators mix
with the built-ins inside one ``TerrainGeneratorCfg``.

yaml sketch::

    terrain:
      name: generator
      params:
        num_rows: 1
        num_cols: 8           # >= num_envs so each env owns a sub-terrain
        size: [8.0, 8.0]
        horizontal_scale: 0.05
        vertical_scale: 0.005
        friction: 0.8
        sub_terrains:
          flat:   {type: mesh_plane, proportion: 0.3}
          rough:  {type: hf_random_uniform, proportion: 0.3,
                   params: {noise_range: [0.02, 0.06], noise_step: 0.02}}
          stairs: {type: hf_pyramid_stairs, proportion: 0.2,
                   params: {step_height_range: [0.05, 0.15], step_width: 0.31}}
          gap:    {type: "custom:unigym.backends.isaaclab2.terrains.custom.gap.gap_terrain",
                   proportion: 0.2, params: {gap_width: 0.6, gap_depth: -1.0}}

The build runs on the server (isaaclab); unit tests inject fake cfg classes
through ``classes=`` exactly like :class:`StepTerrain`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from .base import TerrainPreset


# --------------------------------------------------------------------------- #
# sub-terrain type -> canonical Isaac Lab cfg class name
# --------------------------------------------------------------------------- #
# ``hf_*`` are height-field generators (HfTerrainBaseCfg subclasses); Isaac
# Lab's TerrainGenerator copies size/horizontal_scale/vertical_scale/slope_threshold
# onto them.  ``mesh_*`` are trimesh generators (SubTerrainBaseCfg subclasses).
SUB_TERRAIN_TYPES: dict[str, str] = {
    # height field
    "hf_random_uniform": "HfRandomUniformTerrainCfg",
    "hf_pyramid_sloped": "HfPyramidSlopedTerrainCfg",
    "hf_inverted_pyramid_sloped": "HfInvertedPyramidSlopedTerrainCfg",
    "hf_pyramid_stairs": "HfPyramidStairsTerrainCfg",
    "hf_inverted_pyramid_stairs": "HfInvertedPyramidStairsTerrainCfg",
    "hf_discrete_obstacles": "HfDiscreteObstaclesTerrainCfg",
    "hf_wave": "HfWaveTerrainCfg",
    "hf_stepping_stones": "HfSteppingStonesTerrainCfg",
    # trimesh
    "mesh_plane": "MeshPlaneTerrainCfg",
    "mesh_pyramid_stairs": "MeshPyramidStairsTerrainCfg",
    "mesh_inverted_pyramid_stairs": "MeshInvertedPyramidStairsTerrainCfg",
    "mesh_random_grid": "MeshRandomGridTerrainCfg",
    "mesh_rails": "MeshRailsTerrainCfg",
    "mesh_pit": "MeshPitTerrainCfg",
    "mesh_box": "MeshBoxTerrainCfg",
    "mesh_gap": "MeshGapTerrainCfg",
    "mesh_floating_ring": "MeshFloatingRingTerrainCfg",
    "mesh_star": "MeshStarTerrainCfg",
    "mesh_repeated_pyramids": "MeshRepeatedPyramidsTerrainCfg",
    "mesh_repeated_boxes": "MeshRepeatedBoxesTerrainCfg",
    "mesh_repeated_cylinders": "MeshRepeatedCylindersTerrainCfg",
}


def _resolve(classes):
    """Return the cfg-class table (real isaaclab when classes is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    import isaaclab.terrains as terrain_gen  # noqa: F401  (ensures terrain cfg modules registered)
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
    from isaaclab.terrains.height_field import (
        HfDiscreteObstaclesTerrainCfg,
        HfInvertedPyramidSlopedTerrainCfg,
        HfInvertedPyramidStairsTerrainCfg,
        HfPyramidSlopedTerrainCfg,
        HfPyramidStairsTerrainCfg,
        HfRandomUniformTerrainCfg,
        HfSteppingStonesTerrainCfg,
        HfTerrainBaseCfg,
        HfWaveTerrainCfg,
    )
    from isaaclab.terrains.trimesh import (
        MeshBoxTerrainCfg,
        MeshFloatingRingTerrainCfg,
        MeshGapTerrainCfg,
        MeshInvertedPyramidStairsTerrainCfg,
        MeshPitTerrainCfg,
        MeshPlaneTerrainCfg,
        MeshPyramidStairsTerrainCfg,
        MeshRailsTerrainCfg,
        MeshRandomGridTerrainCfg,
        MeshRepeatedBoxesTerrainCfg,
        MeshRepeatedCylindersTerrainCfg,
        MeshRepeatedPyramidsTerrainCfg,
        MeshStarTerrainCfg,
    )

    return {
        "TerrainImporterCfg": TerrainImporterCfg,
        "TerrainGeneratorCfg": TerrainGeneratorCfg,
        "RigidBodyMaterialCfg": sim_utils.RigidBodyMaterialCfg,
        "PreviewSurfaceCfg": sim_utils.PreviewSurfaceCfg,
        "HfTerrainBaseCfg": HfTerrainBaseCfg,
        # height field
        "HfRandomUniformTerrainCfg": HfRandomUniformTerrainCfg,
        "HfPyramidSlopedTerrainCfg": HfPyramidSlopedTerrainCfg,
        "HfInvertedPyramidSlopedTerrainCfg": HfInvertedPyramidSlopedTerrainCfg,
        "HfPyramidStairsTerrainCfg": HfPyramidStairsTerrainCfg,
        "HfInvertedPyramidStairsTerrainCfg": HfInvertedPyramidStairsTerrainCfg,
        "HfDiscreteObstaclesTerrainCfg": HfDiscreteObstaclesTerrainCfg,
        "HfWaveTerrainCfg": HfWaveTerrainCfg,
        "HfSteppingStonesTerrainCfg": HfSteppingStonesTerrainCfg,
        # trimesh
        "MeshPlaneTerrainCfg": MeshPlaneTerrainCfg,
        "MeshPyramidStairsTerrainCfg": MeshPyramidStairsTerrainCfg,
        "MeshInvertedPyramidStairsTerrainCfg": MeshInvertedPyramidStairsTerrainCfg,
        "MeshRandomGridTerrainCfg": MeshRandomGridTerrainCfg,
        "MeshRailsTerrainCfg": MeshRailsTerrainCfg,
        "MeshPitTerrainCfg": MeshPitTerrainCfg,
        "MeshBoxTerrainCfg": MeshBoxTerrainCfg,
        "MeshGapTerrainCfg": MeshGapTerrainCfg,
        "MeshFloatingRingTerrainCfg": MeshFloatingRingTerrainCfg,
        "MeshStarTerrainCfg": MeshStarTerrainCfg,
        "MeshRepeatedPyramidsTerrainCfg": MeshRepeatedPyramidsTerrainCfg,
        "MeshRepeatedBoxesTerrainCfg": MeshRepeatedBoxesTerrainCfg,
        "MeshRepeatedCylindersTerrainCfg": MeshRepeatedCylindersTerrainCfg,
    }


def _resolve_callable(dotted: str) -> Callable:
    """Import ``module.attr`` and return ``attr`` (a callable)."""
    dotted = dotted.strip()
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"custom: path must be 'module.attr', got '{dotted}'")
    return getattr(importlib.import_module(mod_name), attr)


def _make_custom_cfg(func: Callable, proportion: float, params: dict, classes: dict):
    """Wrap an already-decorated height-field function in a cfg instance.

    Returns an ``HfTerrainBaseCfg`` subclass instance.  ``function`` and
    the custom params are installed as **class attributes** of
    a fresh subclass so they survive the TerrainGenerator's ``cfg.copy()``
    (which is ``dataclasses.replace`` and only round-trips *declared*
    dataclass fields — class attrs are inherited by the replaced instance).
    ``function`` is wrapped in :class:`staticmethod` (so instance access
    returns the raw callable under both the real configclass and the plain
    fake base used in tests) and the custom params are plain class attributes.
    On the server ``@configclass`` is then applied to the subclass — this
    regenerates ``__init__`` so the inherited ``function=MISSING`` field
    default is overridden by our value (otherwise the parent ``__init__``
    writes ``MISSING`` into the instance and shadows the class attr), promotes
    the custom params to *declared* fields (so they survive the
    TerrainGenerator's ``cfg.copy()`` = ``dataclasses.replace``), and stores
    ``function`` as the raw callable in the instance dict.  Locally
    (isaaclab absent, fake base class with no MISSING-overwriting ``__init__``)
    the ``@configclass`` step is skipped and the staticmethod class attr is
    read directly.  ``proportion`` is a declared field on
    ``HfTerrainBaseCfg`` and is passed through the constructor;
    ``size``/``horizontal_scale``/``vertical_scale`` are filled in later by
    the TerrainGenerator.
    """
    hf_base = classes["HfTerrainBaseCfg"]
    namespace = {"function": staticmethod(func)}
    namespace.update(params)  # gap_width / pit_depth / ... as class attrs
    cls = type("CustomTerrainCfg", (hf_base,), namespace)
    try:  # server: regenerate __init__ + promote params to declared fields
        from isaaclab.utils import configclass

        cls = configclass(cls)
    except Exception:  # local: fake base, staticmethod attr read directly
        pass
    return cls(proportion=proportion)


class GeneratorTerrain(TerrainPreset):
    """Route A+C: Isaac Lab TerrainGeneratorCfg terrain mix.

    Built-in sub-terrains plus ``custom:`` height-field functions.
    """

    #: type-string -> canonical cfg-class name (resolved through ``classes``)
    SUB_TERRAIN_REGISTRY = SUB_TERRAIN_TYPES

    def __init__(self, params: dict | None = None, **kwargs):
        # Accept either ``GeneratorTerrain({...})`` (design signature) or the
        # ``build_terrain`` factory which unpacks the params dict as kwargs.
        self.params = dict(params or {})
        self.params.update(kwargs)

    def build(self, num_envs: int, env_spacing: float, *, classes=None, **kwargs) -> dict:
        """Assemble the ``TerrainImporterCfg`` (see TerrainPreset.build).

        Args:
            num_envs: number of parallel environments.
            env_spacing: grid spacing between env origins (m).
            classes: optional injected cfg-class table (testing).
            **kwargs: accepted for ``TerrainPreset.build`` signature
                parity; parameters arrive via ``__init__`` instead.

        Returns:
            ``{"terrain": TerrainImporterCfg}`` with the sub-terrain mix
            resolved from ``params["sub_terrains"]``.
        """
        c = _resolve(classes)
        p = self.params
        num_rows = int(p.get("num_rows", 1))
        num_cols = int(p.get("num_cols", max(1, num_envs)))
        size = tuple(p.get("size", [8.0, 8.0]))
        friction = float(p.get("friction", 0.8))
        use_origins = bool(p.get("use_terrain_origins", True))

        sub_terrains = {}
        for name, spec in p.get("sub_terrains", {}).items():
            spec = dict(spec)
            t = spec["type"]
            proportion = float(spec.get("proportion", 1.0))
            params = dict(spec.get("params", {}))
            if isinstance(t, str) and t.startswith("custom:"):
                func = _resolve_callable(t[len("custom:") :])
                sub_terrains[name] = _make_custom_cfg(func, proportion, params, c)
            else:
                cls_name = self.SUB_TERRAIN_REGISTRY.get(t)
                if cls_name is None:
                    raise ValueError(
                        f"unknown sub-terrain type '{t}' "
                        f"(available: {sorted(self.SUB_TERRAIN_REGISTRY)})"
                    )
                sub_terrains[name] = c[cls_name](proportion=proportion, **params)

        gen_cfg = c["TerrainGeneratorCfg"](
            size=size,
            border_width=float(p.get("border_width", 0.0)),
            num_rows=num_rows,
            num_cols=num_cols,
            curriculum=False,
            use_cache=False,
            horizontal_scale=float(p.get("horizontal_scale", 0.1)),
            vertical_scale=float(p.get("vertical_scale", 0.005)),
            slope_threshold=p.get("slope_threshold", 0.75),
            sub_terrains=sub_terrains,
        )
        cfg = c["TerrainImporterCfg"](
            prim_path="/World/ground",
            num_envs=num_envs,
            terrain_type="generator",
            terrain_generator=gen_cfg,
            use_terrain_origins=use_origins,
            max_init_terrain_level=(
                int(p["max_init_terrain_level"])
                if "max_init_terrain_level" in p
                else (num_rows - 1 if use_origins else None)
            ),
            env_spacing=env_spacing,
            physics_material=c["RigidBodyMaterialCfg"](
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=c["PreviewSurfaceCfg"](
                diffuse_color=tuple(p.get("color", (0.5, 0.45, 0.4)))
            ),
            debug_vis=False,
        )
        return {"terrain": cfg}
