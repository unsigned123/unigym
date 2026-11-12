"""Scene definitions: the neutral IR vocabulary + the scene registry.

:mod:`.scene_ir` is the closed neutral vocabulary (``Pose`` / ``Instance``
/ ``Region`` / ``Slot`` / ``Material`` / ``Scene``; standard library
only), :mod:`.primitives` is the built-in generators' neutral payload
(``BoxShape`` / ``SphereShape`` / ``CylinderShape`` / ``CapsuleShape`` /
``PlaneShape`` / ``ShapePayload`` / ``ContactSpec``), :mod:`.generators`
is the closed built-in generator registry (``room_shell`` / ``box`` /
``sphere`` / ``cylinder`` / ``capsule`` / ``plane`` -> payload),
:mod:`.pipeline`
is the two-protocol bridge (pipeline / content provider) with the four
binding checks and the build scaffold, and :mod:`.registry` is the
closed scene-provider registry.  The composer (the L1 mechanism that
turns a scene into engine elements) lives on the backend side
(:mod:`unigym.backends.shared.mjfamily.scene.scene_render`) -- this package never
imports backends (the import-linter contract pins the direction).

The production registry stays empty here: content providers ship with
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
    PROVIDER_WHENS,
    BuildContext,
    CompiledScene,
    ContentProvider,
    Draw,
    PayloadSet,
    Resolution,
    Scaffold,
    SceneBindingError,
    ScenePipeline,
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
from .registry import (
    SceneContext,
    build_scene_definition,
    register_scene,
    registered_scenes,
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
    "ContentProvider",
    "Draw",
    "INSTANCE_ROLES",
    "Instance",
    "Material",
    "PROVIDER_WHENS",
    "PayloadSet",
    "PlaneShape",
    "Pose",
    "Region",
    "RegionDefinition",
    "Resolution",
    "SHAPE_CLASSES",
    "Scaffold",
    "Scene",
    "SceneBindingError",
    "SceneContext",
    "ScenePipeline",
    "ShapePayload",
    "Slot",
    "SphereShape",
    "Terrain",
    "build_payload",
    "build_scene_definition",
    "normalize_articulation",
    "normalize_articulations",
    "normalize_contact_sensor_inventory",
    "validate_contact_sensor_inventory",
    "generator_schema",
    "register_generator",
    "register_scene",
    "registered_generators",
    "registered_scenes",
    "validate_binding",
    "validate_scene",
]
