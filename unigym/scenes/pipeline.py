"""The scene pipeline seam: two protocols, four checks, one scaffold.

Standard library plus the sibling neutral modules (``.scene_ir`` and
``.generators``) only -- no engine import, no content import.  This is the
whole bridge design section 4 asks for:

* :class:`ScenePipeline` builds one :class:`~unigym.scenes.scene_ir.Scene`
  from a :class:`BuildContext` (data, never a handle);
* :class:`SceneContentFiller` fills a built scene and declares ``when``
  (``"build"`` joins the compiled product, ``"reset"`` re-fills per
  episode; the core supports both and never picks a side);
* :func:`validate_binding` is the FOUR core checks -- generator
  registered, referenced slot declared, pose frame legal, source present;
* :class:`Scaffold` is the one order: build -> validate -> fill ->
  validate, returning the scene plus the three artifact views
  (:class:`Draw`, :class:`Resolution`, :class:`PayloadSet`).

Everything else (how to draw, how to match, how to place) belongs to the
engine side.  Every failure names the instance id, the rule it broke and
the fix; every artifact view is sorted by id and JSON-ready.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, TypeGuard, overload

from unigym._typing import JsonValue

from .generators import registered_generators
from .scene_ir import Instance, Scene, validate_scene

#: legal content-filler timings (declared by the engine, never chosen here).
FILLER_WHENS = ("build", "reset")

#: the neutral parameter keys through which an instance's params cite a slot.
SLOT_REFERENCE_KEYS = ("slot", "slot_id")

#: a content-filler factory (called by an engine with its own keyword facts).
_SceneContentFillerFactory = Callable[..., "SceneContentFiller"]

#: the closed content-filler registry (name -> factory callable).
_SCENE_CONTENT_FILLERS: dict[str, _SceneContentFillerFactory] = {}
_CONTENT_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class SceneBindingError(ValueError):
    """One pipeline/content binding rule failed (fail closed, teaching)."""


def _empty_mounts() -> dict[str, str | os.PathLike[str]]:
    """An empty mount table (typed dataclass default)."""
    return {}


def _empty_facts() -> dict[str, JsonValue]:
    """An empty compile-facts table (typed dataclass default)."""
    return {}


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Narrow a runtime value to a mapping (leaf shape trusted)."""
    return isinstance(value, Mapping)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Narrow a runtime value to a list/tuple (leaf shape trusted)."""
    return isinstance(value, (list, tuple))


def _is_set(value: object) -> TypeGuard[set[object] | frozenset[object]]:
    """Narrow a runtime value to a set/frozenset (leaf shape trusted)."""
    return isinstance(value, (set, frozenset))


def _is_pathlike(value: object) -> TypeGuard[os.PathLike[str]]:
    """Narrow a runtime value to a filesystem path."""
    return isinstance(value, os.PathLike)


@overload
def _jsonify(
    value: Mapping[str, JsonValue],
    path: str = "params",
) -> dict[str, JsonValue]: ...


@overload
def _jsonify(value: object, path: str = "params") -> JsonValue: ...


def _jsonify(value: object, path: str = "params") -> JsonValue:
    """Convert *value* into a JSON-ready structure (sorted keys, no sets)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if _is_pathlike(value):
        return os.fspath(value)
    if _is_mapping(value):
        result: dict[str, JsonValue] = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise TypeError(f"{path}: artifact views need string mapping keys, got {key!r}")
            result[key] = _jsonify(value[key], f"{path}.{key}")
        return result
    if _is_sequence(value):
        return [_jsonify(entry, f"{path}[{index}]") for index, entry in enumerate(value)]
    if _is_set(value):
        return sorted((_jsonify(entry, f"{path}[]") for entry in value), key=repr)
    raise TypeError(
        f"{path}: {type(value).__name__} is not JSON-ready -- recipe values "
        f"must be plain data (mapping, sequence, set, string, number or None)"
    )


@dataclass(frozen=True)
class BuildContext:
    """What a pipeline needs to build one scene (data, never a handle)."""

    recipe: Mapping[str, JsonValue]
    mounts: Mapping[str, str | os.PathLike[str]] = field(default_factory=_empty_mounts)
    seed: int | None = None
    recipe_path: str = ""

    def __post_init__(self):
        """Validate and copy the context fields."""
        if not isinstance(self.recipe, Mapping):
            raise SceneBindingError(
                f"build context: recipe must be a mapping, got " f"{type(self.recipe).__name__}"
            )
        if not isinstance(self.mounts, Mapping):
            raise SceneBindingError(
                f"build context: mounts must be a mapping of mount name -> root "
                f"path, got {type(self.mounts).__name__}"
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise SceneBindingError(
                f"build context: seed must be an int or None, got {self.seed!r}"
            )
        if not isinstance(self.recipe_path, str):
            raise SceneBindingError(
                f"build context: recipe_path must be a string, got "
                f"{type(self.recipe_path).__name__}"
            )
        object.__setattr__(self, "recipe", dict(self.recipe))
        object.__setattr__(self, "mounts", dict(self.mounts))


class ScenePipeline(Protocol):
    """Turn a :class:`BuildContext` into one neutral scene (engine code)."""

    name: str

    def build(self, context: BuildContext) -> Scene:
        """Return the built scene."""
        ...


class SceneContentFiller(Protocol):
    """Fill a built scene with instances (content code)."""

    name: str
    when: str
    last_record: object
    """Opaque fill record the filler exposes after :meth:`fill`."""

    def fill(
        self,
        scene: Scene,
        spec: Mapping[str, JsonValue],
        rng: random.Random,
    ) -> tuple[Instance, ...]:
        """Return the instances this filler adds to *scene*."""
        ...


def register_scene_content_filler(
    name: str,
    factory: _SceneContentFillerFactory,
) -> _SceneContentFillerFactory:
    """Register one content-filler factory under a snake_case name.

    The core owns the PROTOCOL, never the content: a pack declares its
    filler through this closed registry.  A factory is called as
    ``factory(catalog=..., generator_exists=...)`` and returns an object
    satisfying :class:`SceneContentFiller`; an optional ``catalog_ref``
    attribute names the pack reference the engine loads the catalog from.

    Args:
        name: The filler name (lowercase snake_case, unique).
        factory: The callable that builds a filler instance.

    Returns:
        The registered factory.

    Raises:
        SceneBindingError: The name is illegal or already registered, or
            the factory is not callable.
    """
    if not isinstance(name, str) or _CONTENT_NAME_RE.fullmatch(name) is None:
        raise SceneBindingError(
            f"content filler name {name!r} must be lowercase snake_case "
            f"(path-like spellings are rejected)"
        )
    if not callable(factory):
        raise SceneBindingError(
            f"content filler {name!r} factory must be callable, got " f"{type(factory).__name__}"
        )
    if name in _SCENE_CONTENT_FILLERS:
        raise SceneBindingError(
            f"content filler {name!r} is registered twice -- give each " f"filler a unique name"
        )
    _SCENE_CONTENT_FILLERS[name] = factory
    return factory


def scene_content_filler(name: str) -> _SceneContentFillerFactory:
    """Return the registered factory for *name* (fail closed, teaching).

    Args:
        name: The filler name declared by the content spec.

    Returns:
        The factory registered under *name*.

    Raises:
        SceneBindingError: The name is not registered; the error lists the
            legal set.
    """
    if not isinstance(name, str) or not name:
        raise SceneBindingError(f"content filler name must be a non-empty string, got {name!r}")
    try:
        return _SCENE_CONTENT_FILLERS[name]
    except KeyError:
        raise SceneBindingError(
            f"content filler {name!r} is not registered (registered: "
            f"{sorted(_SCENE_CONTENT_FILLERS)}) -- mount the pack that owns it "
            f"and import its content face"
        ) from None


def registered_scene_content_fillers() -> list[str]:
    """Return the registered content-filler names, sorted."""
    return sorted(_SCENE_CONTENT_FILLERS)


def snapshot_scene_content_fillers() -> dict[str, _SceneContentFillerFactory]:
    """Return a copy of the content-filler registry (isolation/restore face)."""
    return dict(_SCENE_CONTENT_FILLERS)


def restore_scene_content_fillers(
    snapshot: Mapping[str, _SceneContentFillerFactory],
) -> None:
    """Restore the content-filler registry to exactly *snapshot*."""
    if not isinstance(snapshot, Mapping):
        raise TypeError(
            "restore_scene_content_fillers expects a mapping, " f"got {type(snapshot).__name__}"
        )
    _SCENE_CONTENT_FILLERS.clear()
    _SCENE_CONTENT_FILLERS.update(snapshot)


@dataclass(frozen=True)
class Draw:
    """What the pipeline decided to draw (the inputs of a resolution)."""

    pipeline: str
    seed: int | None
    recipe_path: str
    params: Mapping[str, JsonValue]

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the JSON-ready artifact view of this draw."""
        return {
            "pipeline": self.pipeline,
            "seed": self.seed,
            "recipe_path": self.recipe_path,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class Resolution:
    """What came out: the id-level view of the built scene."""

    instances: tuple[Mapping[str, JsonValue], ...] = ()
    regions: tuple[Mapping[str, JsonValue], ...] = ()
    slots: tuple[Mapping[str, JsonValue], ...] = ()
    materials: tuple[Mapping[str, JsonValue], ...] = ()

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the JSON-ready artifact view of this resolution."""
        return {
            "instances": [dict(entry) for entry in self.instances],
            "regions": [dict(entry) for entry in self.regions],
            "slots": [dict(entry) for entry in self.slots],
            "materials": [dict(entry) for entry in self.materials],
        }


@dataclass(frozen=True)
class PayloadSet:
    """The instance -> payload view (kind only; the engine resolves files)."""

    entries: tuple[Mapping[str, JsonValue], ...] = ()

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the JSON-ready artifact view of this payload set."""
        return {"entries": [dict(entry) for entry in self.entries]}


@dataclass(frozen=True)
class CompiledScene:
    """Engine-side facts recorded after compile (filled by the backend)."""

    facts: Mapping[str, JsonValue] = field(default_factory=_empty_facts)

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the JSON-ready artifact view of the compile facts."""
        return {"facts": dict(self.facts)}


def _slot_references(instance: Instance) -> list[str]:
    """Return the slot ids one instance's params cite (invalid shapes fail)."""
    references: list[str] = []
    for key in SLOT_REFERENCE_KEYS:
        value = instance.params.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            references.append(value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(entry, str) and entry for entry in value
        ):
            references.extend(entry for entry in value if isinstance(entry, str))
        else:
            raise SceneBindingError(
                f"instance {instance.id!r}: params[{key!r}] must name a slot id "
                f"or a list of slot ids, got {value!r}"
            )
    return references


def validate_binding(
    scene: Scene,
    generator_exists: Callable[[str], bool],
    slot_ids: Iterable[str] | None = None,
) -> None:
    """The four core checks (section 4), fail closed and teaching.

    Args:
        scene: The scene whose instances are checked.
        generator_exists: Predicate returning whether a generator name
            resolves in the caller's registry (the core never imports the
            asset registry beyond the built-in fallback).
        slot_ids: The legal slot-id set; ``None`` uses the scene's own
            slots.  Pass a wider set only when the caller owns ids the
            scene does not declare yet (the check is then authoritative).

    Raises:
        SceneBindingError: A generator is unknown, a referenced slot is
            undeclared, a pose frame names no declared slot/region frame,
            or an instance carries no provenance.
    """
    validate_scene(scene)
    if not callable(generator_exists):
        raise SceneBindingError(
            "validate_binding: generator_exists must be callable, got "
            f"{type(generator_exists).__name__} -- pass the registry predicate"
        )
    legal_slots = (
        frozenset(slot.id for slot in scene.slots) if slot_ids is None else frozenset(slot_ids)
    )
    legal_frames = {"world"}
    for slot in scene.slots:
        legal_frames.update((slot.id, slot.frame))
    for region in scene.regions:
        legal_frames.update((region.name, region.frame))
    declared = sorted(legal_slots)
    frames = sorted(legal_frames)
    for instance in scene.instances:
        if not generator_exists(instance.generator):
            raise SceneBindingError(
                f"instance {instance.id!r}: generator {instance.generator!r} is "
                f"not registered -- register it in the generator registry or fix "
                f"the declaration"
            )
        for reference in _slot_references(instance):
            if reference not in legal_slots:
                raise SceneBindingError(
                    f"instance {instance.id!r}: slot {reference!r} is not declared "
                    f"in scene {scene.name!r} (declared: {declared}) -- declare "
                    f"the slot or fix the reference"
                )
        if instance.pose.frame not in legal_frames:
            raise SceneBindingError(
                f"instance {instance.id!r}: pose frame {instance.pose.frame!r} is "
                f"neither 'world' nor a declared slot/region frame "
                f"(legal: {frames}) -- declare the frame or use 'world'"
            )
        if not instance.source:
            raise SceneBindingError(
                f"instance {instance.id!r}: source must be a non-empty provenance "
                f"string -- every hop must carry its source"
            )


class Scaffold:
    """Run one pipeline through the fixed order, returning the artifact views."""

    def __init__(
        self,
        pipeline: ScenePipeline,
        fillers: Iterable[SceneContentFiller] = (),
        generator_exists: Callable[[str], bool] | None = None,
    ):
        """Validate the pipeline and fillers; the closed timing set is checked."""
        name = getattr(pipeline, "name", None)
        if not isinstance(name, str) or not name:
            raise SceneBindingError(
                "scaffold: pipeline.name must be a non-empty string -- name the "
                "pipeline so the draw stays traceable"
            )
        if not callable(getattr(pipeline, "build", None)):
            raise SceneBindingError(
                f"scaffold: pipeline {name!r} must expose a callable build(context)"
            )
        self._pipeline: ScenePipeline = pipeline
        self._generator_exists: Callable[[str], bool] = (
            registered_generators().__contains__ if generator_exists is None else generator_exists
        )
        if not callable(self._generator_exists):
            raise SceneBindingError(
                "scaffold: generator_exists must be callable -- pass a registry "
                "predicate (a wider one may include the prop-template registry)"
            )
        self._build_fillers: list[SceneContentFiller] = []
        self._reset_fillers: list[SceneContentFiller] = []
        seen: set[str] = set()
        for filler in fillers:
            filler_name = getattr(filler, "name", None)
            when = getattr(filler, "when", None)
            if not isinstance(filler_name, str) or not filler_name:
                raise SceneBindingError(
                    "scaffold: every filler needs a non-empty name -- the fill "
                    "record must name who added each instance"
                )
            if when not in FILLER_WHENS:
                raise SceneBindingError(
                    f"filler {filler_name!r}: when {when!r} must be one of "
                    f"{FILLER_WHENS} -- declare whether the fill joins the build "
                    f"or re-runs per reset"
                )
            if not callable(getattr(filler, "fill", None)):
                raise SceneBindingError(
                    f"filler {filler_name!r}: fill must be callable as " f"fill(scene, spec, rng)"
                )
            if filler_name in seen:
                raise SceneBindingError(
                    f"filler {filler_name!r} is registered twice -- give each "
                    f"filler a unique name"
                )
            seen.add(filler_name)
            if when == "build":
                self._build_fillers.append(filler)
            else:
                self._reset_fillers.append(filler)

    def build(self, context: BuildContext) -> tuple[Scene, Draw, Resolution, PayloadSet]:
        """Build the scene, validate, fill the build fillers, validate again.

        Args:
            context: The :class:`BuildContext` handed to the pipeline.

        Returns:
            The built scene, its draw inputs, its id-level resolution and
            its instance -> payload view.  Reset-timed fillers are NOT
            run here; they are reserved for the engine's reset phase.

        Raises:
            SceneBindingError: Any build-stage rule fails.
        """
        if not isinstance(context, BuildContext):
            raise SceneBindingError(
                f"scaffold: build expected a BuildContext, got " f"{type(context).__name__}"
            )
        scene = self._pipeline.build(context)
        validate_scene(scene)
        validate_binding(scene, self._generator_exists)
        for filler in self._build_fillers:
            scene = self._fill(scene, filler, context)
        validate_scene(scene)
        validate_binding(scene, self._generator_exists)
        draw = Draw(
            pipeline=self._pipeline.name,
            seed=context.seed,
            recipe_path=context.recipe_path,
            params=_jsonify(context.recipe, "recipe"),
        )
        return scene, draw, _resolution(scene), _payload_set(scene)

    @staticmethod
    def _fill(scene: Scene, filler: SceneContentFiller, context: BuildContext) -> Scene:
        """Run one build-timed filler and merge its instances (additive)."""
        seed = None if context.seed is None else f"{context.seed}:{filler.name}:build"
        rng = random.Random(seed)
        added = filler.fill(scene, context.recipe, rng)
        if isinstance(added, (str, bytes)) or not isinstance(added, (list, tuple)):
            raise SceneBindingError(
                f"filler {filler.name!r}: fill must return a tuple of "
                f"Instance, got {type(added).__name__}"
            )
        existing = {entry.id for entry in scene.instances}
        for entry in added:
            if not isinstance(entry, Instance):
                raise SceneBindingError(
                    f"filler {filler.name!r}: fill returned "
                    f"{type(entry).__name__}, expected Instance"
                )
            if entry.id in existing:
                raise SceneBindingError(
                    f"filler {filler.name!r}: duplicate instance id "
                    f"{entry.id!r} -- the scene already declares that id; give "
                    f"each instance a unique id"
                )
            existing.add(entry.id)
        return replace(scene, instances=scene.instances + tuple(added))


def _resolution(scene: Scene) -> Resolution:
    """Return the sorted id-level artifact view of one scene."""
    return Resolution(
        instances=tuple(
            {"id": entry.id, "generator": entry.generator, "source": entry.source}
            for entry in sorted(scene.instances, key=lambda item: item.id)
        ),
        regions=tuple(
            {"name": entry.name, "kind": "region", "source": entry.source}
            for entry in sorted(scene.regions, key=lambda item: item.name)
        ),
        slots=tuple(
            {"id": entry.id, "kind": "slot", "source": entry.source}
            for entry in sorted(scene.slots, key=lambda item: item.id)
        ),
        materials=tuple(
            {"name": entry.name, "kind": "material"}
            for entry in sorted(scene.materials, key=lambda item: item.name)
        ),
    )


def _payload_set(scene: Scene) -> PayloadSet:
    """Return the sorted instance -> payload view (kind only, no files)."""
    return PayloadSet(
        entries=tuple(
            {
                "id": entry.id,
                "generator": entry.generator,
                "source": entry.source,
                "kind": entry.role,
            }
            for entry in sorted(scene.instances, key=lambda item: item.id)
        )
    )


__all__ = [
    "FILLER_WHENS",
    "SLOT_REFERENCE_KEYS",
    "BuildContext",
    "CompiledScene",
    "SceneContentFiller",
    "Draw",
    "PayloadSet",
    "Resolution",
    "Scaffold",
    "SceneBindingError",
    "ScenePipeline",
    "scene_content_filler",
    "register_scene_content_filler",
    "registered_scene_content_fillers",
    "restore_scene_content_fillers",
    "snapshot_scene_content_fillers",
    "validate_binding",
]
