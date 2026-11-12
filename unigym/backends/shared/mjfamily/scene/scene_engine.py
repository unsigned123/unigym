"""The MuJoCo scene-engine implementation (the concrete backend seam).

The neutral core owns the four-call protocol
(:class:`unigym.scenes.engine.SceneEngine`); this module supplies the
concrete mapping: compose through the existing scene composer, attach an
already composed spec at a named frame, serialize the composed spec, and
describe it through the neutral ``ModelIR`` used by conformance.

Importing this module self-registers the engine under ``mujoco`` so the
backend registry and the neutral registry agree without the core naming
a backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..model import MujocoModelProvider
from unigym.scenes.engine import register_scene_engine
from unigym.scenes.scene_ir import Scene

from .scene_render import ComposedScene, attach_composed, compose_scene

if TYPE_CHECKING:
    from unigym.assets.plan import AssetPlan

NAME = "mujoco"

DIGEST_VERSION = "unigym.modelir/2"
"""The canonical version tag of the model-IR conformance digest."""

DIGEST_VERSION_KEY = "digest_version"
"""The top-level digest field carrying :data:`DIGEST_VERSION`."""

DIGEST_MJ_KEYS = ("geom_contact", "opt", "textures")
"""The whitelisted ``ext.mj`` slice: coverage the XML describe had.

The base ``ModelIR`` stays backend-neutral; these three MuJoCo-only
facts live under the namespaced ``ext.mj`` key so cross-engine consumers
can ignore them.
"""


class MujocoSceneEngine:
    """Compose/attach/serialize/describe one scene for the MuJoCo backend."""

    name = NAME

    def compose(self, scene: Scene, asset_plan: AssetPlan) -> ComposedScene:
        """Build the composed scene product (pre-compile spec + regions)."""
        return compose_scene(scene, asset_plan)

    def attach(self, base: object, composed: ComposedScene, *, frame: str) -> Mapping[str, object]:
        """Join an already composed product into *base* at *frame*."""
        return attach_composed(base, composed, frame)

    def to_xml(self, composed: ComposedScene) -> str:
        """Serialize the composed product's spec."""
        return composed.spec.to_xml()

    def describe(self, composed: ComposedScene) -> Mapping[str, object]:
        """Return the canonical declared ``ModelIR`` mapping of *composed*.

        This is the conformance face: it reads the neutral model IR built
        directly from the composed ``MjSpec`` -- no XML serialization and
        no XML parsing on this path.  The IR carries declared values only
        (compiled mass/inertia/addresses stay ``None``), floats are
        ``float.hex()`` strings and mapping keys are sorted by
        :meth:`ModelIR.to_dict`.  The version tag names the digest
        contract, not the MuJoCo version.

        The mapping is ``{"base": ..., "ext": {"mj": ...},
        "digest_version": ...}``.  ``base`` is exactly the previous N1
        shape (no MuJoCo-only fields); ``ext.mj`` carries the whitelisted
        slice that restores the old XML describe's coverage (options,
        textures, per-geom contact parameters).
        """
        declared = MujocoModelProvider().build(composed.spec)
        full = declared.to_dict(include_ext=True, include_meta=False)
        base = {key: value for key, value in full.items() if key != "ext"}
        mj = full.get("ext", {}).get("mj", {})
        ext = {"mj": {key: mj[key] for key in DIGEST_MJ_KEYS if key in mj}}
        data = {"base": base, "ext": ext}
        data[DIGEST_VERSION_KEY] = DIGEST_VERSION
        return dict(sorted(data.items()))


def resolve_scene_engine(name: str = NAME) -> MujocoSceneEngine:
    """Resolve a backend's scene engine through the backend registry.

    Args:
        name: The canonical backend name (default: this backend).

    Returns:
        A fresh engine instance from the backend record's factory.

    Raises:
        RuntimeError: The backend record declares no scene engine.
    """
    from unigym.backends.registry import backend_info

    factory = backend_info(name).scene_engine
    if factory is None:
        raise RuntimeError(
            f"backend {name!r}: the backend registry declares no scene engine "
            f"-- register one before composing a scene"
        )
    engine = factory()
    if not isinstance(engine, MujocoSceneEngine):
        raise RuntimeError(
            f"backend {name!r}: the registered scene engine is not a "
            f"MujocoSceneEngine ({type(engine).__name__})"
        )
    return engine


def compose_and_attach(
    base: object, scene: Scene, asset_plan: AssetPlan, frame: str | None = None
) -> Mapping[str, object]:
    """Compose *scene* through the engine and attach it at one frame.

    *asset_plan* is the composition boundary; every scene asset resolves
    through it before the engine builds elements.  The historical frame
    spelling is preserved when *frame* is omitted, so every existing caller
    keeps its exact composed bytes.
    """
    engine = resolve_scene_engine()
    composed = engine.compose(scene, asset_plan)
    return engine.attach(base, composed, frame=frame or f"Scene_{scene.name}_attach")


register_scene_engine(NAME, MujocoSceneEngine)

__all__ = [
    "NAME",
    "MujocoSceneEngine",
    "compose_and_attach",
    "resolve_scene_engine",
]
