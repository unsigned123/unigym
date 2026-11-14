"""Scene definitions: the neutral IR vocabulary + the scene provider registry.

:mod:`.scene_ir` is the closed neutral vocabulary (``Pose`` / ``Instance``
/ ``Region`` / ``Slot`` / ``Material`` / ``Scene``; standard library
only), :mod:`.primitives` is the built-in generators' neutral payload
(``BoxShape`` / ``SphereShape`` / ``CylinderShape`` / ``CapsuleShape`` /
``PlaneShape`` / ``ShapePayload`` / ``ContactSpec``), :mod:`.generators`
is the closed built-in generator registry (``room_shell`` / ``box`` /
``sphere`` / ``cylinder`` / ``capsule`` / ``plane`` -> payload),
:mod:`.pipeline`
is the two-protocol bridge (pipeline / content filler) with the four
binding checks and the build scaffold, and :mod:`.providers` is the
closed scene-provider registry.  The composer (the L1 mechanism that
turns a scene into engine elements) lives on the backend side
(:mod:`unigym.backends.shared.mjfamily.scene.scene_render`) -- this package never
imports backends (the import-linter contract pins the direction).

The production registry stays empty here: content fillers ship with
the robocasa pack's code face and self-register when that
pack is imported, so this package is importable with no pack mounted.

Import lightness: the neutral vocabulary (``scene_ir`` / ``primitives`` /
``generators`` / ``providers`` / ``RegionDefinition``) is eager because it is
standard-library-only.  The engine-facing names -- ``articulation`` (which
pulls the actuator chain and therefore numpy), ``pipeline`` and ``terrain``
(which imports numpy) -- re-export lazily through PEP 562, so importing a
light sibling (``unigym.pack_api`` / ``unigym.packs`` /
``unigym.scene_presets``) never drags numpy/torch/mujoco into the process.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .generators import (
    build_payload,
    generator_schema,
    register_generator,
    registered_generators,
)
from .primitives import (
    SHAPE_CLASSES,
    BoxShape,
    CapsuleShape,
    ContactSpec,
    CylinderShape,
    PlaneShape,
    ShapePayload,
    SphereShape,
)
from .providers import (
    ProviderInfo,
    ProviderOrigin,
    SceneProvider,
    SceneProviderContext,
    build_scene_from_provider,
    register_scene_provider,
    registered_scene_providers,
    restore_scene_providers,
    scene_provider_origin,
    scene_provider_providers,
    snapshot_scene_providers,
)
from .scene_ir import (
    INSTANCE_ROLES,
    Instance,
    Material,
    Pose,
    Region,
    Scene,
    Slot,
    validate_scene,
)
from unigym.core.region import RegionDefinition

if TYPE_CHECKING:
    # The PEP 562 re-exports below; typed here so pyright sees the names
    # without importing their heavy modules at runtime.
    from .articulation import (
        ArticulationError,
        ArticulationSpec,
        AssetRef,
        ContactSensorGroup,
        ContactSensorInventory,
        JointSpec,
        ROOT_MODES,
        normalize_articulation,
        normalize_articulations,
        normalize_contact_sensor_inventory,
        validate_contact_sensor_inventory,
    )
    from .pipeline import (
        FILLER_WHENS,
        BuildContext,
        CompiledScene,
        Draw,
        PayloadSet,
        Resolution,
        Scaffold,
        SceneBindingError,
        SceneContentFiller,
        ScenePipeline,
        register_scene_content_filler,
        registered_scene_content_fillers,
        restore_scene_content_fillers,
        scene_content_filler,
        snapshot_scene_content_fillers,
        validate_binding,
    )
    from .terrain import Terrain

#: Heavy public name -> (module suffix, attribute).  Imported on first access.
_LAZY = {
    "ArticulationError": ("articulation", "ArticulationError"),
    "ArticulationSpec": ("articulation", "ArticulationSpec"),
    "AssetRef": ("articulation", "AssetRef"),
    "ContactSensorGroup": ("articulation", "ContactSensorGroup"),
    "ContactSensorInventory": ("articulation", "ContactSensorInventory"),
    "JointSpec": ("articulation", "JointSpec"),
    "ROOT_MODES": ("articulation", "ROOT_MODES"),
    "normalize_articulation": ("articulation", "normalize_articulation"),
    "normalize_articulations": ("articulation", "normalize_articulations"),
    "normalize_contact_sensor_inventory": (
        "articulation",
        "normalize_contact_sensor_inventory",
    ),
    "validate_contact_sensor_inventory": (
        "articulation",
        "validate_contact_sensor_inventory",
    ),
    "FILLER_WHENS": ("pipeline", "FILLER_WHENS"),
    "BuildContext": ("pipeline", "BuildContext"),
    "CompiledScene": ("pipeline", "CompiledScene"),
    "Draw": ("pipeline", "Draw"),
    "PayloadSet": ("pipeline", "PayloadSet"),
    "Resolution": ("pipeline", "Resolution"),
    "Scaffold": ("pipeline", "Scaffold"),
    "SceneBindingError": ("pipeline", "SceneBindingError"),
    "SceneContentFiller": ("pipeline", "SceneContentFiller"),
    "ScenePipeline": ("pipeline", "ScenePipeline"),
    "register_scene_content_filler": ("pipeline", "register_scene_content_filler"),
    "registered_scene_content_fillers": (
        "pipeline",
        "registered_scene_content_fillers",
    ),
    "restore_scene_content_fillers": ("pipeline", "restore_scene_content_fillers"),
    "scene_content_filler": ("pipeline", "scene_content_filler"),
    "snapshot_scene_content_fillers": ("pipeline", "snapshot_scene_content_fillers"),
    "validate_binding": ("pipeline", "validate_binding"),
    "Terrain": ("terrain", "Terrain"),
}


def __getattr__(name: str):
    """Resolve one lazy heavy export (PEP 562), caching it in globals.

    Args:
        name: The requested attribute.

    Returns:
        The lazily imported object.

    Raises:
        AttributeError: The name is not part of this package's surface.
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    value = getattr(import_module(f".{module_name}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the eager globals plus every lazy export."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ArticulationError",
    "ArticulationSpec",
    "AssetRef",
    "JointSpec",
    "ROOT_MODES",
    "BoxShape",
    "BuildContext",
    "CapsuleShape",
    "CylinderShape",
    "CompiledScene",
    "ContactSensorGroup",
    "ContactSensorInventory",
    "ContactSpec",
    "Draw",
    "FILLER_WHENS",
    "INSTANCE_ROLES",
    "Instance",
    "Material",
    "PayloadSet",
    "PlaneShape",
    "Pose",
    "ProviderInfo",
    "ProviderOrigin",
    "Region",
    "RegionDefinition",
    "Resolution",
    "SHAPE_CLASSES",
    "Scaffold",
    "Scene",
    "SceneBindingError",
    "SceneContentFiller",
    "ScenePipeline",
    "SceneProvider",
    "SceneProviderContext",
    "ShapePayload",
    "Slot",
    "SphereShape",
    "Terrain",
    "build_payload",
    "build_scene_from_provider",
    "normalize_articulation",
    "normalize_articulations",
    "normalize_contact_sensor_inventory",
    "validate_contact_sensor_inventory",
    "generator_schema",
    "register_generator",
    "register_scene_content_filler",
    "register_scene_provider",
    "registered_generators",
    "registered_scene_content_fillers",
    "registered_scene_providers",
    "restore_scene_content_fillers",
    "restore_scene_providers",
    "scene_content_filler",
    "scene_provider_origin",
    "scene_provider_providers",
    "snapshot_scene_content_fillers",
    "snapshot_scene_providers",
    "validate_binding",
    "validate_scene",
]
