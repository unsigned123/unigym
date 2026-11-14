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
"""

from __future__ import annotations

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
from .generators import (
    build_payload,
    generator_schema,
    register_generator,
    registered_generators,
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
from .terrain import Terrain
from unigym.core.region import RegionDefinition

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
