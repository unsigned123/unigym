"""Programmatic world builder: ``unigym.World``.

A :class:`World` is a *declaration* of the world half of a run, not a live
engine object.  It produces only the existing config schema consumed by
:func:`unigym.run.spec.build_spec`:

* a ``scene_preset`` catalog reference for
  :meth:`World.from_preset` / :meth:`World.from_config`, resolved through a
  :class:`~unigym.packs.runtime.PackRuntime`;
* the inline ``scene_preset: {provider: ..., params: ...}`` form for
  :meth:`World.from_provider`;
* a unique runtime scene provider for :meth:`World.inline`;
* the ``objects.types`` content block for dynamic free bodies added with
  ``World.add(..., dynamic=True)``.

The inline form registers a provider in the process-wide scene registry
through :func:`unigym.scenes.providers.register_inline_scene_provider`.  That
registration is **prototype-only**: inline providers never appear in the
public ``registered_scene_providers()`` list or provider-info query, their
origin is ``kind="inline"``, the registry lives for the process lifetime and
is never serialized into a config.  Use :meth:`World.from_provider` or
:meth:`World.from_preset` for reproducible worlds.

The builder adds static instances to the closed scene IR
(:mod:`unigym.scenes.scene_ir`) through the registered payload generators.
A ``dynamic=True`` primitive takes the content/props path instead: it is
declared in ``objects.types``, built by the props manager as a free body
(freejoint + mass/inertia) and therefore survives
:meth:`unigym.Run.to_config`.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import yaml

from unigym._typing import JsonValue, is_json_object
from unigym.scenes.generators import build_payload
from unigym.scenes.providers import SceneProviderContext, register_inline_scene_provider
from unigym.scenes.scene_ir import (
    Instance,
    Material,
    Scene,
    validate_scene,
)
from unigym.scenes.terrain import HeightsInput, Terrain
from unigym.world_primitives import (
    Box,
    Capsule,
    Cylinder,
    Plane,
    Sphere,
    _Primitive,
    _SHAPE_TYPES,
    _validate_dynamic_entry,
)

if TYPE_CHECKING:
    from unigym.packs.runtime import PackRuntime

__all__ = ["World", "Box", "Sphere", "Cylinder", "Capsule", "Plane"]


class WorldFragment(TypedDict, total=False):
    """The run-config fragment a :class:`World` contributes."""

    scene_preset: str | dict[str, JsonValue]
    world_base: dict[str, JsonValue]
    articulations: list[dict[str, JsonValue]]
    controls: list[dict[str, JsonValue]]
    objects: dict[str, JsonValue]


def _mapping_section(
    parent: dict[str, JsonValue],
    key: str,
) -> dict[str, JsonValue]:
    """Return a nested JSON mapping child, creating it when absent."""
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


class _TerrainBuilder:
    """Pre-build terrain face of one inline :class:`World`.

    Every constructor method stores a :class:`~unigym.scenes.terrain.Terrain`
    on the world and returns the WORLD (so ``Run(world=world.terrain.stairs(...))``
    reads naturally).  ``replace_floor`` is a live boolean: True (default)
    makes the composer drop the base model's floor/ground plane so the
    terrain is the ground.
    """

    def __init__(self, world: World):
        self._world = world

    def _check(self) -> None:
        """Refuse bundle worlds and post-build mutation."""
        world = self._world
        if world.built:
            raise RuntimeError(
                "World.terrain: the world is already built; terrain "
                "declarations are pre-build only"
            )
        if world._scene_preset_ref is not None or world._scene_provider is not None:
            raise RuntimeError(
                "World.terrain requires an inline world (World() / "
                "World.inline(scene)); preset/provider worlds own their "
                "ground"
            )
        world._ensure_inline()

    def _set(self, terrain: Terrain | None) -> World:
        """Store *terrain* (None = flat) and return the world."""
        self._world._terrain = terrain
        self._world._terrain_explicit = True
        return self._world

    @property
    def replace_floor(self) -> bool:
        """Whether the base model's floor/ground plane is removed."""
        return self._world._terrain_replace_floor

    @replace_floor.setter
    def replace_floor(self, value: bool) -> None:
        self._world._terrain_replace_floor = bool(value)

    def flat(self) -> World:
        """Remove any declared terrain (the ground falls back to the base)."""
        self._check()
        return self._set(None)

    def heightmap(
        self,
        heights: HeightsInput,
        size: tuple[float, float] | None = None,
        base_z: float = 0.0,
        name: str = "terrain",
        meta: Mapping[str, JsonValue] | None = None,
    ) -> World:
        """Use an explicit ``(ny, nx)`` elevation matrix."""
        self._check()
        return self._set(Terrain.heightmap(heights, size=size, base_z=base_z, name=name, meta=meta))

    def stairs(
        self,
        steps: int,
        rise: float,
        run: float,
        width: float,
        base_z: float = 0.0,
        margin: float | None = None,
        resolution: tuple[int, int] | None = None,
        name: str = "terrain",
    ) -> World:
        """Use a staircase climbing +x (see :meth:`Terrain.stairs`)."""
        self._check()
        return self._set(
            Terrain.stairs(
                steps,
                rise,
                run,
                width,
                base_z=base_z,
                margin=margin,
                resolution=resolution,
                name=name,
            )
        )

    def slope(
        self,
        angle_deg: float,
        size: tuple[float, float] = (4.0, 4.0),
        base_z: float = 0.0,
        resolution: tuple[int, int] | None = None,
        name: str = "terrain",
    ) -> World:
        """Use a linear ramp along +x (see :meth:`Terrain.slope`)."""
        self._check()
        return self._set(
            Terrain.slope(angle_deg, size=size, base_z=base_z, resolution=resolution, name=name)
        )


class World:
    """A declarative world (scene preset / provider / inline objects).

    Construct one through the factories; the object is cheap and mutable
    until a :class:`~unigym.run.api.Run` consumes it.  ``Run(world=...)``
    merges :meth:`fragment` into the run config and marks the world built
    after the env compiles.
    """

    def __init__(self):
        self._scene_preset_ref: str | dict[str, JsonValue] | None = None
        self._scene_provider: str | None = None
        self._scene_provider_params: dict[str, JsonValue] = {}
        self._world_base: dict[str, JsonValue] | None = None
        self._base_scene: Scene | None = None
        self._inline_name: str | None = None
        self._inline_scene_name: str | None = None
        self._inline_material: Material | None = None
        self._objects: list[_Primitive] = []
        self._dynamic: list[_Primitive] = []
        self._built = False
        self._flat_ground = False
        self._terrain: Terrain | None = None
        self._terrain_explicit = False
        self._terrain_replace_floor = True
        self._terrain_builder: _TerrainBuilder | None = None
        self._articulations: list[dict[str, JsonValue]] = []
        self._controls: list[dict[str, JsonValue]] = []

    # ------------------------------------------------------------------ #
    # factories
    # ------------------------------------------------------------------ #
    @classmethod
    def from_preset(
        cls,
        ref: str,
        *,
        runtime: PackRuntime,
    ) -> World:
        """Use a scene preset by its exact ``scene_presets`` catalog id.

        Args:
            ref: The catalog id declared on the runtime's ``scene_presets`` face.
            runtime: The pack runtime owning the catalog; there is no
                fallback scan when it is absent.
        """
        from unigym.scene_presets import load_scene_preset

        load_scene_preset(ref, runtime)
        world = cls()
        world._scene_preset_ref = ref
        return world

    @classmethod
    def base(
        cls,
        *,
        mjcf: str | Path | None = None,
        urdf: str | Path | None = None,
        usd: str | Path | None = None,
    ) -> World:
        """Declare an explicit format-neutral world asset (``world_base``).

        At least one format path is required.  The run layer selects the first
        format the backend supports; there is no format fallback.
        """
        mapping: dict[str, JsonValue] = {}
        for key, value in (("mjcf", mjcf), ("urdf", urdf), ("usd", usd)):
            if value is not None:
                text = str(value)
                if not text:
                    raise ValueError(f"World.base {key}= must be a non-empty path")
                mapping[key] = text
        if not mapping:
            raise ValueError("World.base requires at least one of mjcf=, urdf= or usd=")
        world = cls()
        world._world_base = mapping
        return world

    @classmethod
    def from_config(
        cls,
        source: dict[str, JsonValue] | str | Path,
        *,
        runtime: PackRuntime | None = None,
    ) -> World:
        """Extract the world half of a run config or scene preset catalog id.

        Args:
            source: A parsed run-config mapping, a path to a run yaml, or a
                bare scene preset catalog id (``"acme.tabletop"``).
            runtime: The pack runtime resolving a declared ``scene_preset``;
                required when *source* declares a catalog preset.

        Raises:
            ValueError: A scene preset file is passed directly.  Preset refs
                are exact catalog ids, so a direct document is not a reference.
        """
        if isinstance(source, dict):
            return cls._from_run_cfg(source, runtime)
        path = Path(source)
        if path.is_dir() and (path / "scene.yaml").is_file():
            path = path / "scene.yaml"
        if path.is_file():
            raw_doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            loaded: object = {} if raw_doc is None else raw_doc
            if not is_json_object(loaded):
                raise ValueError(
                    f"world source {str(source)!r} must be a "
                    f"yaml mapping, got {type(loaded).__name__}"
                )
            doc = loaded
            if "run_name" in doc or "env" in doc:
                return cls._from_run_cfg(doc, runtime)
            if path.name == "scene.yaml" or doc.get("schema") == "unigym.scene_preset/1":
                raise ValueError(
                    f"World.from_config: {str(source)!r} is a scene preset "
                    f"file, not a scene reference; scene refs are exact "
                    f"PackRuntime 'scene_presets' catalog ids (e.g. "
                    f"'acme.tabletop') resolved with runtime=..."
                )
            return cls._from_run_cfg(doc, runtime)
        if runtime is None:
            raise ValueError(
                f"World.from_config: scene preset ref {str(source)!r} requires a pack "
                f"runtime | hint: scene preset refs are exact PackRuntime "
                f"'scene_presets' catalog ids"
            )
        return cls.from_preset(str(source), runtime=runtime)

    @classmethod
    def _from_run_cfg(
        cls,
        cfg: dict[str, JsonValue],
        runtime: PackRuntime | None = None,
    ) -> World:
        """Extract a world declaration from a run-config mapping."""
        from unigym.scene_presets import load_scene_preset

        ref = cfg.get("scene_preset")
        if ref is not None:
            if isinstance(ref, str):
                if not ref:
                    raise ValueError(
                        "World.from_config: scene_preset ref must be a non-empty "
                        "pack catalog id string"
                    )
                if runtime is None:
                    raise ValueError(
                        f"World.from_config: scene preset ref {ref!r} requires a pack "
                        f"runtime | hint: scene preset refs are exact PackRuntime "
                        f"'scene_presets' catalog ids"
                    )
                load_scene_preset(ref, runtime)
                world = cls()
                world._scene_preset_ref = ref
                return world
            if not isinstance(ref, dict):
                raise ValueError(
                    f"World.from_config: scene_preset must be a catalog id string "
                    f"or a mapping, got {type(ref).__name__}"
                )
            world = cls()
            world._scene_preset_ref = copy.deepcopy(ref)
            return world
        world_base = cfg.get("world_base")
        if world_base is None:
            raise ValueError(
                "World.from_config: the config declares no world (expected "
                "a top-level 'scene_preset:' reference or 'world_base:')"
            )
        world = cls()
        if not isinstance(world_base, dict):
            raise ValueError(
                "World.from_config: world_base must be a mapping "
                "{mjcf|urdf|usd: <path>}; a bare path is constructor-only"
            )
        world._world_base = copy.deepcopy(world_base)
        return world

    @classmethod
    def from_provider(
        cls,
        name: str,
        *,
        params: Mapping[str, JsonValue] | None = None,
    ) -> World:
        """Use a registered scene provider with its explicit params."""
        world = cls()
        world._scene_provider = str(name)
        world._scene_provider_params = dict(params or {})
        return world

    @classmethod
    def inline(cls, scene: Scene) -> World:
        """Use an in-process :class:`Scene` (prototype-only registry)."""
        validate_scene(scene)
        world = cls()
        world._base_scene = scene
        world._ensure_inline()
        return world

    @property
    def terrain(self) -> _TerrainBuilder:
        """The terrain builder face (``world.terrain.stairs(...)``)."""
        if self._terrain_builder is None:
            self._terrain_builder = _TerrainBuilder(self)
        return self._terrain_builder

    def terrain_value(self) -> Terrain | None:
        """The effective terrain IR this world declares (``None`` = flat).

        An inline world returns its declared (or inherited) ``Terrain``;
        bundle/generator worlds own their ground and return ``None``.  This
        is the one read the run spec uses to carry the IR to a backend.
        """
        if self._scene_preset_ref is not None or self._scene_provider is not None:
            return None
        return self.to_scene().terrain

    def to_scene(self) -> Scene:
        """The inline :class:`Scene` this world currently declares."""
        if self._scene_preset_ref is not None or self._scene_provider is not None:
            raise RuntimeError(
                "World.to_scene requires an inline world (World() / "
                "World.inline(scene)); bundle/generator worlds are owned by "
                "their providers"
            )
        if self._inline_name is None and self._base_scene is None and not self._objects:
            # dynamic-only world: no static scene to serve, and registering
            # a provider would make the (fully serializable) declaration
            # process-local -- return the transient view instead.
            return self._scene_now()
        self._ensure_inline()
        return self._scene_now()

    # ------------------------------------------------------------------ #
    # inline scene registry
    # ------------------------------------------------------------------ #
    def _ensure_inline(self) -> None:
        """Create + register the unique runtime provider once."""
        if self._inline_name is not None:
            return
        token = uuid.uuid4().hex
        self._inline_name = f"_inline_{token}"
        self._inline_scene_name = f"inline_{token[:12]}"
        world = self

        def provider(
            slots: Mapping[str, JsonValue],
            context: SceneProviderContext,
        ) -> Scene:
            """Return this World's current inline scene (runtime provider)."""
            return world._scene_now()

        register_inline_scene_provider(self._inline_name, provider)

    def _material(self) -> Material:
        """The inline world's declared material (one texture declaration)."""
        if self._inline_material is None:
            from unigym.core.paths import repo_root

            texture = Path(repo_root()) / "assets" / "table_wood.png"
            inline_name = self._inline_name
            if inline_name is None:
                raise RuntimeError(
                    "World material requested before the inline scene was " "registered"
                )
            name = f"inline_mat_{inline_name.rsplit('_', 1)[-1]}"
            self._inline_material = Material(name, {"kind": "2d", "texture_file": str(texture)})
        return self._inline_material

    def _scene_now(self) -> Scene:
        """Compose the current inline scene (objects + base declarations)."""
        base = self._base_scene
        name = base.name if base is not None else (self._inline_scene_name or "inline_dynamic")
        instances: list[Instance] = list(base.instances) if base is not None else []
        materials: list[Material] = list(base.materials) if base is not None else []
        if self._objects:
            material = self._material()
            if material.name not in {m.name for m in materials}:
                materials.append(material)
            instances.extend(obj._instance(material.name) for obj in self._objects)
        inherited = not self._terrain_explicit and base is not None
        terrain = base.terrain if base is not None and inherited else self._terrain
        if terrain is not None and not inherited:
            terrain = replace(
                terrain, meta={**terrain.meta, "replace_floor": bool(self._terrain_replace_floor)}
            )
        scene = Scene(
            name=name,
            instances=tuple(instances),
            regions=tuple(base.regions) if base is not None else (),
            slots=tuple(base.slots) if base is not None else (),
            materials=tuple(materials),
            engine_hints=dict(base.engine_hints) if base is not None else {},
            terrain=terrain,
        )
        validate_scene(scene)
        return scene

    # ------------------------------------------------------------------ #
    # object composition
    # ------------------------------------------------------------------ #
    def _used_names(self) -> set[str | None]:
        """Every object/instance id already declared in this world."""
        used = {o.name for o in self._objects}
        used |= {o.name for o in self._dynamic}
        if self._base_scene is not None:
            used |= {inst.id for inst in self._base_scene.instances}
        return used

    def _assign_name(self, obj: _Primitive, used: set[str | None]) -> None:
        """Ensure *obj* carries a unique non-null name (mutates it)."""
        if obj.name is None:
            index = len(self._objects) + len(self._dynamic) + 1
            name = f"{obj.kind}_{index}"
            while name in used:
                index += 1
                name = f"{obj.kind}_{index}"
            obj._name = name
            used.add(name)
        elif obj.name in used:
            raise ValueError(f"World.add: duplicate object name " f"{obj.name!r}")
        else:
            used.add(obj.name)

    def add(
        self,
        obj: _Primitive,
        dynamic: bool | None = None,
    ) -> _Primitive:
        """Add a primitive to this world; returns the object.

        Supported primitives are :class:`Box` / :class:`Sphere` /
        :class:`Cylinder` / :class:`Capsule` / :class:`Plane`.

        ``dynamic=False`` (or ``dynamic=None`` with an object whose
        ``dynamic`` flag is false) adds STATIC scene geometry: fixed to the
        world, no freejoint, no mass/inertia, consumed by the inline scene.

        ``dynamic=True`` (or ``dynamic=None`` with ``obj.dynamic``) adds a
        DYNAMIC free body through the existing content/props path: the
        declaration joins the world's ``objects.types`` block, so the
        props manager builds a freejoint body with mass/inertia and the
        object survives :meth:`Run.to_config`.  Static and dynamic objects
        may be mixed in one world; a plane cannot be dynamic.
        """
        if self._built:
            raise RuntimeError(
                "World.add: the world is already built; object composition " "is pre-build only"
            )
        if not isinstance(obj, _SHAPE_TYPES):
            raise TypeError(
                f"World.add expects Box/Sphere/Cylinder/Capsule/Plane, " f"got {type(obj).__name__}"
            )
        if self._scene_preset_ref is not None or self._scene_provider is not None:
            raise RuntimeError(
                "World.add requires an inline world; from_preset/from_provider "
                "worlds are immutable declarations. Start with World() / "
                "World.inline(scene) to compose primitives"
            )
        is_dynamic = obj.dynamic if dynamic is None else bool(dynamic)
        used = self._used_names()
        if is_dynamic:
            if obj.kind == "plane":
                raise ValueError(
                    f"World.add: plane {obj.name!r} cannot be dynamic -- a "
                    f"plane is infinite static scene geometry; use Box/"
                    f"Sphere/Cylinder/Capsule for free bodies"
                )
            self._assign_name(obj, used)
            entry = obj._dynamic_declaration()
            _validate_dynamic_entry(entry)
            obj._world = self
            self._dynamic.append(obj)
            return obj
        self._ensure_inline()
        self._assign_name(obj, used)
        # Validate the exact payload the composer will build (one gate).
        build_payload(
            obj.kind,
            {
                "name": obj.name,
                **obj._shape_params(),
                "material": obj.material or self._material().name,
                "pos": list(obj.pos),
                "quat": list(obj.quat),
                "layer": "visual",
            },
        )
        obj._world = self
        self._objects.append(obj)
        return obj

    def add_many(
        self,
        objects: Iterable[_Primitive],
        dynamic: bool | None = None,
    ) -> list[_Primitive]:
        """Add several primitives in one batch (static and/or dynamic).

        Args:
            objects: An iterable of :class:`Box` / :class:`Sphere` /
                :class:`Cylinder` / :class:`Capsule` / :class:`Plane`
                instances (each may carry its own ``dynamic`` flag).
            dynamic: Optional override applied to every object in the
                batch (``None`` respects each object's flag).

        Returns:
            The added objects in input order (same objects, now owned by
            this world).
        """
        batch = list(objects)
        for obj in batch:
            if not isinstance(obj, _SHAPE_TYPES):
                raise TypeError(
                    f"World.add_many expects Box/Sphere/Cylinder/Capsule/"
                    f"Plane, got {type(obj).__name__}"
                )
        return [self.add(obj, dynamic=dynamic) for obj in batch]

    def articulation(
        self,
        name: str,
        *,
        model: str | Mapping[str, JsonValue] | None = None,
        control: object | None = None,
        root: str | None = None,
        spawn: Mapping[str, JsonValue] | None = None,
        joints: Mapping[str, JsonValue] | list[Mapping[str, JsonValue]] | None = None,
        base_body: str | None = None,
        foot_bodies: Sequence[str] | None = None,
        body_sets: Mapping[str, Sequence[str]] | None = None,
        contacts: JsonValue = None,
        passive_joints: Sequence[object] | None = None,
        params: Mapping[str, JsonValue] | None = None,
        adapter: str = "manual",
        drive: str = "position_pd",
        target: str = "position",
    ) -> World:
        """Declare one articulation on this world (emits ``articulations[]``).

        The declaration is the v104 neutral shape: an articulation is
        ``root + model + joints`` (no articulation-level control mode).
        ``model`` may be a path string or a mapping (``{"mjcf": <path>}``;
        ``urdf``/``usd`` are accepted by the IR for future backends).  The
        drive is emitted as a top-level ``controls[]`` group named after the
        articulation (action face == articulation name).

        Builder form::

            world = World.base(mjcf=".../base_skin.xml")
            world.articulation(
                "g1",
                model={"mjcf": ".../g1_29dof_with_hand.xml"},
                adapter="manual")

        Args:
            name: Unique articulation name (the neutral ``unigym_name``;
                backend prefixes never appear on this surface).
            model: ``{mjcf|urdf|usd: <path>}`` or a bare MJCF path.
            control: REMOVED (v104); passing it fails with a teaching error.
            root: ``free`` (default) / ``fixed`` / ``welded``.
            spawn: Root spawn declaration (``pos`` + optional ``yaw``).
            joints: Per-joint declarations (mapping or list of mappings).
            base_body: Optional canonical base body id (inherits the profile).
            foot_bodies: Optional canonical foot body ids (inherits/empties).
            body_sets: Optional canonical named body sets.
            contacts: Optional contact sensor inventory (``all`` / ``none`` /
                a group list / a normalized inventory).
            passive_joints: Joints the articulation owns but does not drive.
            params: Optional backend knobs.
            adapter: The adapter name owning this articulation's body group.
            drive: The body control drive (default ``position_pd``).
            target: The explicit target kind (must match ``drive``).

        Returns:
            This world (builder chaining).
        """
        if not isinstance(name, str) or not name:
            raise ValueError("World.articulation: name must be a non-empty " "string")
        if control is not None:
            raise ValueError(
                f"World.articulation {name!r}: the articulation-level "
                f"`control` declaration was removed in v104; declare the "
                f"drive as a top-level `controls:` group "
                f"(name/adapter/articulation/drive/target) via the "
                f"adapter/drive/target arguments"
            )
        entry: dict[str, JsonValue] = {"name": name}
        if model is not None:
            entry["model"] = (
                {"mjcf": str(model)} if isinstance(model, str) else copy.deepcopy(dict(model))
            )
        if root is not None:
            entry["root"] = str(root)
        if spawn is not None:
            entry["spawn"] = copy.deepcopy(dict(spawn))
        if joints is not None:
            entry["joints"] = _joints_json(joints)
        if base_body is not None:
            entry["base_body"] = str(base_body)
        if foot_bodies is not None:
            entry["foot_bodies"] = [str(body) for body in foot_bodies]
        if body_sets is not None:
            entry["body_sets"] = {
                str(name): [str(body) for body in bodies] for name, bodies in body_sets.items()
            }
        if contacts is not None:
            entry["contacts"] = copy.deepcopy(contacts)
        if passive_joints is not None:
            entry["passive_joints"] = [str(j) for j in passive_joints]
        if params is not None:
            entry["params"] = copy.deepcopy(dict(params))
        if any(existing["name"] == name for existing in self._articulations):
            raise ValueError(f"World.articulation: duplicate articulation name {name!r}")
        self._articulations.append(entry)
        group: dict[str, JsonValue] = {
            "name": name,
            "adapter": str(adapter),
            "articulation": name,
            "drive": str(drive),
            "target": str(target),
        }
        if joints is not None:
            declared = _joint_names_from_declaration(joints)
            if declared:
                joints_json: list[JsonValue] = [name for name in declared]
                group["joints"] = joints_json
            else:
                # empty declaration == the profile's body set
                group["joint_set"] = "body"
        else:
            # a programmatic body group selects the profile set explicitly;
            # build resolves it once (no group-name inference)
            group["joint_set"] = "body"
        self._controls.append(group)
        return self

    def flat_ground(self) -> World:
        """Add a ground slab to an inline world (no-op for bundle worlds).

        The common in-repo ``scene_43dof*.xml`` world assets already carry
        a floor plane, so this helper is only meaningful for a base model
        that lacks one.  Bundle/generator worlds own their own ground, so
        the call is a no-op there.
        """
        if self._scene_preset_ref is None and self._scene_provider is None:
            self.add(
                Box((20.0, 20.0, 0.1), name=_unique_name(self, "ground"), pos=(0.0, 0.0, -0.05))
            )
            self._flat_ground = True
        else:
            self._flat_ground = True
        return self

    # ------------------------------------------------------------------ #
    # config fragment / lifecycle
    # ------------------------------------------------------------------ #
    @property
    def built(self) -> bool:
        """True once a consuming :class:`Run` has compiled the world."""
        return self._built

    def static_declarations(self) -> list[dict[str, JsonValue]]:
        """The engine-neutral static primitive declarations (``World.add``).

        Each entry carries the primitive kind, the neutral shape params and
        the world pose, so a backend with a native static-collider face
        (isaaclab: ``AssetBaseCfg`` + a collision-only shape spawner) can
        map the SAME declaration the mujoco inline scene renders.  Empty for
        bundle/generator worlds and for ``World.inline(scene)`` (those are
        provider-owned; see :attr:`unmapped_static_scene`).
        """
        return [
            {
                "name": obj.name,
                "kind": obj.kind,
                "params": dict(obj._shape_params()),
                "pos": list(obj.pos),
                "quat": list(obj.quat),
            }
            for obj in self._objects
        ]

    @property
    def unmapped_static_scene(self) -> bool:
        """True when an inline base scene carries instances no backend maps.

        ``World.add`` primitives are the closed vocabulary a static backend
        can translate (:meth:`static_declarations`); a ``World.inline(scene)``
        base scene can carry arbitrary scene-IR generators, so it has no
        backend-neutral static mapping.  Callers use this to fail fast
        instead of silently dropping the geometry.
        """
        return self._base_scene is not None and bool(self._base_scene.instances)

    def dynamic_declarations(self) -> dict[str, JsonValue]:
        """The ``objects:`` content block for this world's dynamic objects.

        ``dynamic=True`` objects are NOT scene instances: they are dynamic
        free bodies built by the props manager, so they persist in the
        run config (and through :meth:`Run.to_config`) instead of the
        process-local inline scene registry.  Returns an empty mapping
        when no dynamic object was added.
        """
        if not self._dynamic:
            return {}
        return {
            "enabled": True,
            "types": [obj._dynamic_declaration() for obj in self._dynamic],
        }

    def fragment(self) -> WorldFragment:
        """The run-config fragment this world contributes.

        Returns either ``{"scene_preset": "<catalog id>"}``, the inline
        ``{"scene_preset": {"provider": ..., "params": ...}}`` form, or
        ``{"world_base": {mjcf|urdf|usd: ...}}``, plus an ``objects:`` block
        when the world declared dynamic free bodies and an
        ``articulations:`` block when :meth:`articulation` was called.
        ``world_base`` is a TOP-LEVEL run config key; it never rides
        ``env.params``.
        """
        fragment = WorldFragment()
        if self._scene_preset_ref is not None:
            fragment["scene_preset"] = copy.deepcopy(self._scene_preset_ref)
        else:
            if self._inline_name is not None:
                fragment["scene_preset"] = {"provider": self._inline_name, "params": {}}
            elif self._scene_provider is not None:
                fragment["scene_preset"] = {
                    "provider": self._scene_provider,
                    "params": copy.deepcopy(self._scene_provider_params),
                }
            if self._world_base is not None:
                fragment["world_base"] = copy.deepcopy(self._world_base)
        if self._articulations:
            fragment["articulations"] = [copy.deepcopy(entry) for entry in self._articulations]
        if self._controls:
            fragment["controls"] = [copy.deepcopy(entry) for entry in self._controls]
        dynamic = self.dynamic_declarations()
        if dynamic:
            fragment["objects"] = dynamic
        return fragment

    def apply(self, cfg: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Merge this world's fragment into a run config (in place)."""
        fragment = self.fragment()
        if "scene_preset" in fragment:
            cfg["scene_preset"] = copy.deepcopy(fragment["scene_preset"])
        if "world_base" in fragment:
            cfg["world_base"] = copy.deepcopy(fragment["world_base"])
        if "articulations" in fragment:
            _merge_articulations(cfg, fragment["articulations"])
        if "controls" in fragment:
            _merge_controls(cfg, fragment["controls"])
        dynamic = fragment.get("objects")
        if dynamic:
            # Preserve the config's own spelling (objects: surface wins,
            # legacy props: stays legacy) and append the world's types.
            key = "objects" if "objects" in cfg or "props" not in cfg else "props"
            block = _mapping_section(cfg, key)
            block["enabled"] = True
            raw_types = block.get("types")
            types_list: list[JsonValue] = raw_types if isinstance(raw_types, list) else []
            block["types"] = types_list
            dynamic_types = dynamic.get("types")
            if isinstance(dynamic_types, list):
                types_list.extend(copy.deepcopy(dynamic_types))
        return cfg

    def _mark_built(self) -> None:
        """Seal the world after the env compiles (mutations now fail)."""
        self._built = True


def _joints_json(
    joints: Mapping[str, JsonValue] | list[Mapping[str, JsonValue]],
) -> JsonValue:
    """A deep-copied JSON view of a ``joints:`` declaration."""
    if isinstance(joints, Mapping):
        return copy.deepcopy(dict(joints))
    return [copy.deepcopy(dict(entry)) for entry in joints]


def _joint_names_from_declaration(
    joints: Mapping[str, JsonValue] | list[Mapping[str, JsonValue]],
) -> list[str]:
    """Joint names from a ``joints:`` declaration (mapping or list form)."""
    if isinstance(joints, Mapping):
        return [str(name) for name in joints]
    names: list[str] = []
    for entry in joints:
        if "name" in entry:
            names.append(str(entry["name"]))
    return names


def _merge_controls(
    cfg: dict[str, JsonValue],
    declared: list[dict[str, JsonValue]],
) -> None:
    """Merge world-declared control groups into the config list by name."""
    existing = cfg.get("controls")
    if not existing:
        initial: list[JsonValue] = [copy.deepcopy(entry) for entry in declared]
        cfg["controls"] = initial
        return
    existing_list: list[JsonValue] = existing if isinstance(existing, list) else []
    merged: list[JsonValue] = [copy.deepcopy(entry) for entry in existing_list]
    index: dict[str, int] = {}
    for position, entry in enumerate(merged):
        if isinstance(entry, dict):
            name_value = entry.get("name")
            if isinstance(name_value, str):
                index[name_value] = position
    for entry in declared:
        name_value = entry["name"]
        assert isinstance(name_value, str)
        position = index.get(name_value)
        if position is None:
            index[name_value] = len(merged)
            merged.append(copy.deepcopy(entry))
        else:
            merged[position] = copy.deepcopy(entry)
    cfg["controls"] = merged


def _merge_articulations(
    cfg: dict[str, JsonValue],
    declared: list[dict[str, JsonValue]],
) -> None:
    """Merge world-declared articulations into the config list by name.

    A world entry with the same ``name`` replaces the existing declaration
    in place (so the programmatic ``Run(robot="g1")`` stub is upgraded to
    the world's asset-bearing entry); new names append.  Mixing two
    policy articulations still fails later in ``normalize_articulations``.
    """
    existing = cfg.get("articulations")
    if not existing:
        initial: list[JsonValue] = [copy.deepcopy(entry) for entry in declared]
        cfg["articulations"] = initial
        return
    existing_list: list[JsonValue] = existing if isinstance(existing, list) else []
    merged: list[JsonValue] = [copy.deepcopy(entry) for entry in existing_list]
    index: dict[str, int] = {}
    for position, entry in enumerate(merged):
        if isinstance(entry, dict):
            name_value = entry.get("name")
            if isinstance(name_value, str):
                index[name_value] = position
    for entry in declared:
        name_value = entry["name"]
        assert isinstance(name_value, str)
        position = index.get(name_value)
        if position is None:
            index[name_value] = len(merged)
            merged.append(copy.deepcopy(entry))
        else:
            merged[position] = copy.deepcopy(entry)
    cfg["articulations"] = merged


def _unique_name(world: World, stem: str) -> str:
    """A neutral, unused object name for a helper primitive."""
    used = {o.name for o in world._objects}
    used |= {o.name for o in world._dynamic}
    if world._base_scene is not None:
        used |= {inst.id for inst in world._base_scene.instances}
    index = len(world._objects) + len(world._dynamic) + 1
    name = f"{stem}_{index}"
    while name in used:
        index += 1
        name = f"{stem}_{index}"
    return name
