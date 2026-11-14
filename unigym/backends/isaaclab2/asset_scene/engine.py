"""IsaacLab 2 scene engine: composed file-backed asset fields.

Composes the neutral scene into IsaacLab scene fields, keeps the ``World.add``
primitive passthrough byte-compatible, and supports static USD, dynamic USD and
jointless-URDF ``RigidObjectCfg`` assets.  A static URDF, an articulating URDF
and an articulation-authored USD are refused with teaching errors.

``compose``, ``describe`` and ``to_xml`` are Kit-free: they do not import
``isaaclab``, ``pxr`` or ``omni``.  Only ``attach`` builds cfg objects (with
lazily resolved classes) and inspects a USD stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from unigym.assets.asset_ref import AssetRef
from unigym.assets.plan import AssetPlan, resolve_asset_instance
from unigym.backends.isaaclab2.static_shapes import build_static_payload_fields
from unigym.backends.registry import backend_info
from unigym.scenes.artifacts import ASSET_SCENE_DIGEST_VERSION, asset_instance_rows
from unigym.scenes.engine import register_scene_engine
from unigym.scenes.generators import build_payload, generator_schema, registered_generators
from unigym.scenes.pipeline import validate_asset_scene
from unigym.scenes.scene_ir import AssetInstance, Scene

from .fields import build_field
from .inspect import (
    UsdAuthoring,
    inspect_usd_authoring as _inspect_usd_authoring,
    urdf_joint_names,
)
from .names import asset_field, asset_prim_path

#: ``_IsaaclabSceneCfg`` fields an asset-scene field may never shadow.  The
#: list is local because importing ``env`` would import IsaacLab at module
#: import time; it mirrors the closed scene config declaration.
_RESERVED_SCENE_FIELDS = frozenset(
    {"num_envs", "env_spacing", "ground", "terrain", "light", "dome_light", "robot"}
)


@dataclass(frozen=True)
class AssetFieldSpec:
    """One composed file-backed asset field.

    Attributes:
        field: The scene config field name (``asset_<scene>_<id>``).
        prim_path: The env-relative prim path.
        cfg_kind: The cfg form (``static_usd`` / ``dynamic_usd`` /
            ``dynamic_urdf``).
        kind: The neutral asset role (``static_rigid`` / ``dynamic_rigid``).
        format: The selected asset format (``usd`` / ``urdf``).
        asset: The effective neutral asset reference.
        resolved_path: The absolute asset path (artifacts only).
        pos: The world position.
        rot: The world rotation ``(w, x, y, z)``.
        collision: The collision toggle.
        source: The declaration provenance.
    """

    field: str
    prim_path: str
    cfg_kind: str
    kind: str
    format: str
    asset: AssetRef
    resolved_path: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    collision: bool
    source: str


@dataclass(frozen=True)
class ComposedAssetScene:
    """The IsaacLab composed product: asset fields, neutral rows and scene."""

    scene_name: str
    fields: tuple[AssetFieldSpec, ...]
    rows: tuple[Mapping[str, object], ...]
    scene: Scene


def _asset_cfg_kind(entry: AssetInstance, asset_format: str, resolved_path: str) -> str:
    """Select the cfg form for one asset instance, or refuse it.

    The Kit-free jointless guard runs here for a dynamic URDF: any declared
    ``<joint>`` converts to an articulation, which has no passive
    ``RigidObjectCfg`` form.

    Args:
        entry: The declared asset instance (kind/id for the teaching errors).
        asset_format: The resolved asset format (``usd`` / ``urdf``).
        resolved_path: The resolved absolute asset path (URDF joint check).

    Returns:
        ``static_usd``, ``dynamic_usd`` or ``dynamic_urdf``.

    Raises:
        ValueError: The kind/format pair has no field form, a static URDF is
            declared, or a dynamic URDF declares joints (or cannot be read).
    """
    if entry.kind == "static_rigid":
        if asset_format == "usd":
            return "static_usd"
        if asset_format == "urdf":
            raise ValueError(
                f"static_rigid URDF {resolved_path!r} is not supported on "
                f"IsaacLab; provide a static USD or use kind: "
                f"dynamic_rigid with a jointless URDF (URDF conversion "
                f"yields an articulation/anchored root)"
            )
    elif entry.kind == "dynamic_rigid":
        if asset_format == "usd":
            return "dynamic_usd"
        if asset_format == "urdf":
            joints = urdf_joint_names(resolved_path)
            if joints:
                raise ValueError(
                    f"dynamic_rigid URDF {resolved_path!r} declares joints "
                    f"{list(joints)}; only a jointless URDF maps to "
                    f"RigidObjectCfg; articulation-in-scene is unsupported"
                )
            return "dynamic_urdf"
    raise ValueError(
        f"asset instance {entry.id!r}: kind {entry.kind!r} with format "
        f"{asset_format!r} has no IsaacLab asset-scene field form "
        f"(supported: static_rigid/dynamic_rigid with usd/urdf; URDF must be "
        f"jointless for dynamic_rigid)"
    )


class AssetSceneEngine:
    """Compose, attach, serialize and describe one neutral scene for IsaacLab."""

    name = "isaaclab2"

    def __init__(
        self,
        *,
        classes: Mapping[str, type] | None = None,
        inspect_usd_authoring: Callable[[str], UsdAuthoring] | None = None,
    ) -> None:
        """Store the offline cfg-class table and USD inspector seams.

        Args:
            classes: Optional cfg-class table for tests; ``None`` resolves
                the real IsaacLab classes lazily at ``attach``.
            inspect_usd_authoring: Optional USD inspector for tests;
                ``None`` uses the module implementation.
        """
        self._classes = classes
        self._inspect_usd_authoring = (
            _inspect_usd_authoring if inspect_usd_authoring is None else inspect_usd_authoring
        )

    def compose(self, scene: Scene, asset_plan: AssetPlan | None) -> ComposedAssetScene:
        """Compose one neutral scene into IsaacLab asset fields and rows.

        Args:
            scene: The neutral scene to compose.
            asset_plan: The boundary that resolves file-backed asset paths;
                ``None`` is accepted for a primitive-only scene.  A scene
                with asset instances names the missing boundary.

        Returns:
            The composed asset scene.

        Raises:
            SceneBindingError: The scene fails ``validate_asset_scene``.
            AssetRefError: No asset format is compatible with IsaacLab.
            ValueError: A scene semantic (terrain, engine hints, a
                non-primitive generator, a static URDF, or an articulating
                URDF) is unsupported.
        """
        validate_asset_scene(scene)
        if scene.terrain is not None:
            raise ValueError(
                f"scene {scene.name!r}: terrain is not supported on an asset "
                f"scene -- the run terrain owns the ground; remove scene.terrain"
            )
        if scene.engine_hints:
            raise ValueError(
                f"scene {scene.name!r}: engine_hints are not supported on an "
                f"asset scene -- they name the MuJoCo-only hint vocabulary; "
                f"remove scene.engine_hints"
            )

        generators = set(registered_generators())
        for instance in scene.instances:
            if instance.generator not in generators:
                raise ValueError(
                    f"scene instance {instance.id!r}: generator "
                    f"{instance.generator!r} has no IsaacLab file-backed "
                    f"realization; use AssetInstance with .usd/.urdf, or run "
                    f"this scene on backend: mujoco (generated payloads are "
                    f"mujoco-only)"
                )

        if scene.asset_instances and asset_plan is None:
            raise ValueError(
                "scene composition requires an asset plan: build a pack "
                "runtime at the composition boundary and pass its AssetPlan"
            )
        formats = tuple(backend_info("isaaclab2").asset_formats)
        selected: dict[str, str] = {}
        fields: list[AssetFieldSpec] = []
        for entry in sorted(scene.asset_instances, key=lambda item: item.id):
            asset_format, resolved_path = resolve_asset_instance(entry.asset, asset_plan, formats)
            cfg_kind = _asset_cfg_kind(entry, asset_format, resolved_path)
            selected[entry.id] = asset_format
            fields.append(
                AssetFieldSpec(
                    field=asset_field(scene.name, entry.id),
                    prim_path=asset_prim_path(scene.name, entry.id),
                    cfg_kind=cfg_kind,
                    kind=entry.kind,
                    format=asset_format,
                    asset=entry.asset,
                    resolved_path=resolved_path,
                    pos=entry.pose.xyz,
                    rot=entry.pose.quat_wxyz,
                    collision=entry.collision,
                    source=entry.source,
                )
            )
        rows = asset_instance_rows(scene, selected)
        return ComposedAssetScene(
            scene_name=scene.name,
            fields=tuple(fields),
            rows=tuple(rows),
            scene=scene,
        )

    def attach(
        self,
        base: object,
        composed: ComposedAssetScene,
        *,
        frame: str,
    ) -> Mapping[str, object]:
        """Assign the composed fields onto an ``InteractiveSceneCfg``-like base.

        ``frame`` is a namespace label only: IsaacLab has no transform frame
        at this seam, and the MVP composes one scene per run.  The fixed
        ``{ENV_REGEX_NS}/Scene_<scene>/<id>`` prim paths stay authoritative.

        Args:
            base: The scene config object fields are assigned to.
            composed: The composed asset scene.
            frame: The caller's namespace label (documented no-op).

        Returns:
            An empty region table (asset scenes declare no regions).

        Raises:
            ValueError: The base is not a settable scene config, a field
                collides, a static USD authors rigid/articulation schemas, or
                a dynamic USD authors an articulation root.
        """
        del frame
        if base is None or not callable(getattr(base, "__setattr__", None)):
            raise ValueError(
                "asset-scene attach requires an InteractiveSceneCfg-like "
                f"settable base, got {type(base).__name__}"
            )
        if not isinstance(composed, ComposedAssetScene):
            raise ValueError(
                f"asset-scene attach expects a ComposedAssetScene, got "
                f"{type(composed).__name__}"
            )
        pending: list[tuple[str, object]] = []
        seen: set[str] = set()
        for field, cfg in self._primitive_fields(composed).items():
            self._require_available(base, field, seen)
            pending.append((field, cfg))
        for spec in composed.fields:
            self._require_available(base, spec.field, seen)
            if spec.cfg_kind == "static_usd":
                authored = self._inspect_usd_authoring(spec.resolved_path)
                self._require_authored_static(spec, authored)
            elif spec.cfg_kind == "dynamic_usd":
                authored = self._inspect_usd_authoring(spec.resolved_path)
                self._require_no_articulation(spec, authored)
            # dynamic_urdf: the URDF converter spawns inside
            # ``InteractiveScene.__init__`` (after this attach returns), so the
            # converted articulation root does not exist yet.  There is no
            # honest pre-convert check here beyond the Kit-free jointless guard
            # in ``compose``; the post-convert articulation refusal is verified
            # only in the real IsaacLab dog lane.
            pending.append((spec.field, build_field(spec, self._classes)))
        for field, cfg in pending:
            setattr(base, field, cfg)
        return {}

    def to_xml(self, composed: ComposedAssetScene) -> str:
        """Return the composed asset fields as canonical JSON text.

        IsaacLab has no scene XML; the historical protocol method name is
        kept.  Absolute resolved paths may appear here (artifact/A-B face)
        but never in :meth:`describe`.

        Args:
            composed: The composed asset scene.

        Returns:
            Canonical JSON text (sorted keys, two-space indent).
        """
        if not isinstance(composed, ComposedAssetScene):
            raise ValueError(
                f"asset-scene to_xml expects a ComposedAssetScene, got "
                f"{type(composed).__name__}"
            )
        payload: dict[str, object] = {
            "scene_name": composed.scene_name,
            "fields": [
                {
                    "field": spec.field,
                    "prim_path": spec.prim_path,
                    "cfg_kind": spec.cfg_kind,
                    "kind": spec.kind,
                    "format": spec.format,
                    "asset": spec.asset.to_dict(),
                    "resolved_path": spec.resolved_path,
                    "pose": {"pos": list(spec.pos), "rot": list(spec.rot)},
                    "collision": spec.collision,
                    "source": spec.source,
                }
                for spec in composed.fields
            ],
        }
        return json.dumps(payload, sort_keys=True, indent=2)

    def describe(self, composed: ComposedAssetScene) -> Mapping[str, object]:
        """Return the neutral description of the composed asset scene.

        Args:
            composed: The composed asset scene.

        Returns:
            The asset rows plus the declared ``ext.scene`` field/prim table;
            no resolved paths (authored refs are carried verbatim) and no
            cfg-class names.
        """
        if not isinstance(composed, ComposedAssetScene):
            raise ValueError(
                f"asset-scene describe expects a ComposedAssetScene, got "
                f"{type(composed).__name__}"
            )
        fields = [
            {
                "field": spec.field,
                "prim_path": spec.prim_path,
                "kind": spec.kind,
                "format": spec.format,
            }
            for spec in sorted(composed.fields, key=lambda item: item.field)
        ]
        return {
            "asset_scene": {
                "digest_version": ASSET_SCENE_DIGEST_VERSION,
                "instances": [dict(row) for row in composed.rows],
            },
            "ext": {"scene": {"fields": fields}},
            "digest_version": ASSET_SCENE_DIGEST_VERSION,
        }

    def _primitive_fields(self, composed: ComposedAssetScene) -> dict:
        """Build the ``World.add`` primitive passthrough fields."""
        fields: dict = {}
        for instance in composed.scene.instances:
            schema = generator_schema(instance.generator)
            params = dict(instance.params)
            params.setdefault("name", instance.id)
            if "pos" in schema:
                params.setdefault("pos", tuple(instance.pose.xyz))
            if "quat" in schema:
                params.setdefault("quat", tuple(instance.pose.quat_wxyz))
            payload = build_payload(instance.generator, params)
            built = build_static_payload_fields(
                payload,
                instance_id=instance.id,
                classes=self._classes,
            )
            for field, cfg in built.items():
                if field in fields:
                    raise ValueError(
                        f"scene field {field!r} is declared twice -- instance "
                        f"ids and generated shape names must be unique"
                    )
                fields[field] = cfg
        return fields

    @staticmethod
    def _require_available(base: object, field: str, seen: set[str]) -> None:
        """Reject a field that shadows a reserved/existing/duplicate entry."""
        if field in seen:
            raise ValueError(
                f"scene field {field!r} is declared twice -- instance ids and "
                f"generated shape names must be unique"
            )
        if field in _RESERVED_SCENE_FIELDS:
            raise ValueError(
                f"scene field {field!r} collides with the reserved "
                f"InteractiveSceneCfg field {field!r} -- choose a different "
                f"instance id"
            )
        if hasattr(base, field):
            raise ValueError(
                f"scene field {field!r} collides with an existing scene entry "
                f"-- choose a different instance id"
            )
        seen.add(field)

    @staticmethod
    def _require_authored_static(spec: AssetFieldSpec, authored: UsdAuthoring) -> None:
        """Reject a static USD that authors rigid-body/articulation schemas."""
        authored_schemas: list[str] = []
        if authored.rigid_body_paths:
            authored_schemas.append(f"RigidBodyAPI at {list(authored.rigid_body_paths)}")
        if authored.articulation_root_paths:
            authored_schemas.append(
                f"ArticulationRootAPI at {list(authored.articulation_root_paths)}"
            )
        if authored_schemas:
            raise ValueError(
                f"asset field {spec.field!r}: static USD {spec.resolved_path!r} "
                f"authors {' and '.join(authored_schemas)}; a static USD must "
                f"not carry authored rigid/articulation schemas -- provide a "
                f"truly static USD or declare kind: dynamic_rigid"
            )

    @staticmethod
    def _require_no_articulation(spec: AssetFieldSpec, authored: UsdAuthoring) -> None:
        """Reject a dynamic USD that authors an articulation root.

        A ``RigidBodyAPI`` is expected on a passive rigid-body USD and is
        allowed; an ``ArticulationRootAPI`` is an articulation-in-scene and is
        unsupported.
        """
        if authored.articulation_root_paths:
            raise ValueError(
                f"asset field {spec.field!r}: dynamic USD {spec.resolved_path!r} "
                f"authors ArticulationRootAPI at "
                f"{list(authored.articulation_root_paths)}; "
                f"articulation-in-scene is unsupported -- provide a passive "
                f"rigid-body USD"
            )


register_scene_engine("isaaclab2", AssetSceneEngine)

__all__ = ["AssetFieldSpec", "AssetSceneEngine", "ComposedAssetScene"]
