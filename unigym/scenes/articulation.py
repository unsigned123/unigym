"""Neutral articulation IR (authoring model).

An articulation is ``root + model + joints + spawn``.  There is no
articulation-level control mode: *how* a joint is driven (and which adapter
owns it) is declared explicitly in the run-level ``controls[]`` block
(:mod:`unigym.runtime.controls`), and joints that are not driven at all are
listed in :attr:`ArticulationSpec.passive_joints`.  The IR is pure data and
imports only the standard library -- no engine and no backend name is
allowed here (attribute neutrality).

v104 removed ``ControlSpec`` / ``JointSpec.mode`` / ``policy_articulation``
/ ``policy_section``: the drive declaration lives in ``controls[]`` and the
robot recipe only supplies the joint order / q0 / body parts.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from unigym._typing import JsonValue

__all__ = [
    "ARTICULATION_PREFIX",
    "ArticulationError",
    "ArticulationSpec",
    "AssetRef",
    "JointSpec",
    "ROOT_MODES",
    "articulation_key",
    "normalize_articulation",
    "normalize_articulations",
]

ARTICULATION_PREFIX = "articulation"
"""The observation namespace prefix for per-articulation keys (v101)."""

ROOT_MODES = ("free", "fixed", "welded")
"""Root semantics: free joint / fixed to world / welded to a host body."""


def articulation_key(name: str, key: str) -> str:
    """Namespace one observation key under an articulation.

    Args:
        name: articulation name (e.g. ``"g1"`` / ``"door"``).
        key: bare observation key (e.g. ``"joint/pos"``).

    Returns:
        ``"articulation/<name>/<key>"``.
    """
    if not name:
        raise ArticulationError("articulation name must be non-empty")
    if not key:
        raise ArticulationError("articulation observation key must be "
                                "non-empty")
    return f"{ARTICULATION_PREFIX}/{name}/{key}"


class ArticulationError(ValueError):
    """A malformed articulation declaration (fail-fast, no alias)."""


def _finite(value: JsonValue, what: str) -> float:
    if not isinstance(value, (int, float, str)):
        raise ArticulationError(f"{what} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ArticulationError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise ArticulationError(f"{what} must be finite, got {value!r}")
    return number


def _optional_float(value: JsonValue, what: str) -> float | None:
    if value is None:
        return None
    return _finite(value, what)


def _optional_str(value: JsonValue, what: str) -> str | None:
    if value is None:
        return None
    return str(value)


def _range_value(value: JsonValue, what: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ArticulationError(f"{what} must be [lo, hi], got {value!r}")
    return (_finite(value[0], f"{what}[0]"),
            _finite(value[1], f"{what}[1]"))


def _empty_json_map() -> dict[str, JsonValue]:
    """Typed empty default for mapping fields."""
    return {}


def _unknown_keys(
    mapping: Mapping[str, JsonValue], allowed: tuple[str, ...], what: str,
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ArticulationError(
            f"{what} declares unknown keys {unknown}; allowed keys are "
            f"{list(allowed)}")


@dataclass(frozen=True)
class AssetRef:
    """One articulation asset, per backend family (may leave all None)."""

    mjcf: str | None = None
    urdf: str | None = None
    usd: str | None = None

    def __post_init__(self):
        for key in ("mjcf", "urdf", "usd"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, str(value))

    def to_dict(self) -> dict[str, str]:
        """Non-empty asset fields (stable key order)."""
        out: dict[str, str] = {}
        for key in ("mjcf", "urdf", "usd"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_mapping(
        cls, value: AssetRef | Mapping[str, JsonValue],
        what: str = "model",
    ) -> AssetRef:
        """Normalize an ``AssetRef`` or ``{mjcf|urdf|usd: <path>}`` mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ArticulationError(
                f"{what} must be a mapping {{mjcf|urdf|usd: <path>}}, "
                f"got {value!r}")
        _unknown_keys(value, ("mjcf", "urdf", "usd"), what)
        return cls(mjcf=_optional_str(value.get("mjcf"), f"{what}.mjcf"),
                   urdf=_optional_str(value.get("urdf"), f"{what}.urdf"),
                   usd=_optional_str(value.get("usd"), f"{what}.usd"))


@dataclass(frozen=True)
class JointSpec:
    """One declared joint of an articulation (authoring IR).

    A joint no longer carries a capability ``mode``: every declared joint is
    actuated unless it is named in :attr:`ArticulationSpec.passive_joints`,
    and the drive (``position_pd`` / ``position_servo`` / ...) is declared by
    the owning group in the run-level ``controls[]`` block.
    """

    name: str
    init: float = 0.0
    range: tuple[float, float] | None = None
    effort_limit: float | None = None
    damping: float | None = None

    def __post_init__(self):
        if not self.name:
            raise ArticulationError("joint name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        for key in ("effort_limit", "damping"):
            object.__setattr__(self, key, _optional_float(
                getattr(self, key), f"joint {self.name!r} {key}"))
        rng = self.range
        if rng is not None:
            if len(rng) != 2:
                raise ArticulationError(
                    f"joint {self.name!r} range must be [lo, hi], got {rng!r}")
            rng = (_finite(rng[0], f"joint {self.name!r} range[0]"),
                   _finite(rng[1], f"joint {self.name!r} range[1]"))
            if rng[0] > rng[1]:
                raise ArticulationError(
                    f"joint {self.name!r} range {list(rng)} has lo > hi")
            object.__setattr__(self, "range", rng)
        init = _finite(self.init, f"joint {self.name!r} init")
        if rng is not None and not (rng[0] <= init <= rng[1]):
            raise ArticulationError(
                f"joint {self.name!r} init {init!r} is outside the declared "
                f"range {list(rng)}")
        object.__setattr__(self, "init", init)

    @classmethod
    def from_mapping(
        cls, name: str, params: Mapping[str, JsonValue] | None = None,
    ) -> JointSpec:
        """Normalize one joint declaration (``{init, range, ...}``)."""
        params = params or {}
        if "mode" in params:
            raise ArticulationError(
                f"joint {name!r} declares the removed v104 key 'mode'; joints "
                f"are actuated unless listed in the articulation's "
                f"`passive_joints`, and the drive belongs to a top-level "
                f"`controls[]` group (drive: position_pd | position_servo | "
                f"torque | velocity | effort | current)")
        if "stiffness" in params or "latch" in params:
            raise ArticulationError(
                f"joint {name!r} declares a removed v104 key "
                f"({sorted(set(params) & {'stiffness', 'latch'})}); declare "
                f"gains on the owning top-level `controls[]` group instead")
        _unknown_keys(params, ("init", "range", "effort_limit", "damping"),
                      f"joint {name!r}")
        return cls(
            name=name,
            init=_finite(params.get("init", 0.0), f"joint {name!r} init"),
            range=_range_value(params.get("range"), f"joint {name!r} range"),
            effort_limit=_optional_float(
                params.get("effort_limit"),
                f"joint {name!r} effort_limit"),
            damping=_optional_float(
                params.get("damping"), f"joint {name!r} damping"),
        )


@dataclass(frozen=True)
class ArticulationSpec:
    """One articulation: root + model + joints + spawn (neutral IR).

    ``joints`` may be omitted for recipe-derived articulations (the robot
    recipe supplies the joint order); when it is declared, every joint is
    actuated unless listed in ``passive_joints``.  Control coverage is
    validated at the run level against ``controls[]``
    (:func:`unigym.runtime.controls.validate_controls`).
    """

    name: str
    model: AssetRef | None = None
    root: str = "free"
    spawn: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)
    joints: tuple[JointSpec, ...] = ()
    body_parts: Mapping[str, JsonValue] = field(
        default_factory=_empty_json_map)
    passive_joints: tuple[str, ...] = ()
    params: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)

    def __post_init__(self):
        if not self.name:
            raise ArticulationError("articulation name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        if self.root not in ROOT_MODES:
            raise ArticulationError(
                f"articulation {self.name!r} root must be one of "
                f"{ROOT_MODES}, got {self.root!r}")
        if self.model is not None:
            object.__setattr__(self, "model", AssetRef.from_mapping(
                self.model, f"articulation {self.name!r} model"))
        if not isinstance(self.spawn, Mapping):
            raise ArticulationError(
                f"articulation {self.name!r} spawn must be a mapping")
        object.__setattr__(self, "spawn", dict(self.spawn))
        object.__setattr__(self, "joints", _joints_tuple(self.name,
                                                         self.joints))
        if not isinstance(self.body_parts, Mapping):
            raise ArticulationError(
                f"articulation {self.name!r} body_parts must be a mapping")
        object.__setattr__(self, "body_parts", dict(self.body_parts))
        passive = tuple(str(name) for name in (self.passive_joints or ()))
        if len(set(passive)) != len(passive):
            raise ArticulationError(
                f"articulation {self.name!r} passive_joints contains "
                f"duplicates {passive!r}")
        declared = {joint.name for joint in self.joints}
        if declared:
            unknown = sorted(set(passive) - declared)
            if unknown:
                raise ArticulationError(
                    f"articulation {self.name!r} passive_joints {unknown} are "
                    f"not declared joints")
        object.__setattr__(self, "passive_joints", passive)
        if not isinstance(self.params, Mapping):
            raise ArticulationError(
                f"articulation {self.name!r} params must be a mapping")
        object.__setattr__(self, "params", dict(self.params))


def _joints_tuple(
    name: str, joints: JsonValue | Sequence[JointSpec],
) -> tuple[JointSpec, ...]:
    if not joints:
        return ()
    if isinstance(joints, Mapping):
        entries: list[JointSpec] = []
        for jname, params in joints.items():
            if not isinstance(params, Mapping):
                raise ArticulationError(
                    f"articulation {name!r} joint {jname!r} must be a "
                    f"mapping, got {params!r}")
            entries.append(JointSpec.from_mapping(str(jname), params))
        return tuple(entries)
    if not isinstance(joints, (list, tuple)):
        raise ArticulationError(
            f"articulation {name!r} joints must be a list, got {joints!r}")
    out: list[JointSpec] = []
    for entry in joints:
        if isinstance(entry, JointSpec):
            out.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise ArticulationError(
                f"articulation {name!r} joints entries must be mappings, "
                f"got {entry!r}")
        entry_map = dict(entry)
        jname = entry_map.pop("name", None)
        if jname is None:
            raise ArticulationError(
                f"articulation {name!r} joint entry is missing 'name'")
        out.append(JointSpec.from_mapping(str(jname), entry_map))
    return tuple(out)


def _articulation_name(entry: Mapping[str, JsonValue]) -> str:
    name = entry["name"]
    if not name:
        raise ArticulationError("articulation name must be non-empty")
    return str(name)


def _root_value(value: JsonValue, name: str) -> str:
    if not isinstance(value, str) or value not in ROOT_MODES:
        raise ArticulationError(
            f"articulation {name!r} root must be one of "
            f"{ROOT_MODES}, got {value!r}")
    return value


def _mapping_field(value: JsonValue, what: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ArticulationError(f"{what} must be a mapping")
    return value


def _str_tuple(value: JsonValue, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ArticulationError(
            f"{what} must be a list of names, got {value!r}")
    return tuple(str(item) for item in value)


def _model_field(value: JsonValue, name: str) -> AssetRef | None:
    if value is None:
        return None
    if not isinstance(value, (AssetRef, Mapping)):
        raise ArticulationError(
            f"articulation {name!r} model must be a mapping "
            f"{{mjcf|urdf|usd: <path>}}, got {value!r}")
    return AssetRef.from_mapping(value, f"articulation {name!r} model")


def normalize_articulation(entry: Mapping[str, JsonValue]) -> ArticulationSpec:
    """Normalize one ``articulations[]`` entry into an :class:`ArticulationSpec`.

    The config surface is a mapping with the IR fields
    (``name/model/root/spawn/joints/body_parts/passive_joints/params``);
    unknown keys fail fast (there is no shorthand and no alias).  The
    articulation-level ``control:`` block is rejected with a teaching error
    (drives are declared in the run-level ``controls[]``).
    """
    if not isinstance(entry, Mapping):
        raise ArticulationError(
            f"articulations entries must be mappings, got {entry!r}")
    if "control" in entry:
        raise ArticulationError(
            f"articulation {entry.get('name')!r} declares the removed v104 "
            f"key `control:`; articulation-level control modes "
            f"(policy/named/passive) no longer exist -- declare the drive "
            f"explicitly in the run-level `controls:` list "
            f"(name/adapter/articulation/drive/target) and list undriven "
            f"joints in `passive_joints`")
    _unknown_keys(entry, ("name", "model", "root", "spawn", "joints",
                          "body_parts", "passive_joints", "params"),
                  "articulation")
    if "name" not in entry:
        raise ArticulationError("articulation entry is missing 'name'")
    name = _articulation_name(entry)
    return ArticulationSpec(
        name=name,
        model=_model_field(entry.get("model"), name),
        root=_root_value(entry.get("root", "free"), name),
        spawn=_mapping_field(entry.get("spawn", {}),
                             f"articulation {name!r} spawn"),
        joints=_joints_tuple(name, entry.get("joints")),
        body_parts=_mapping_field(entry.get("body_parts", {}),
                                  f"articulation {name!r} body_parts"),
        passive_joints=_str_tuple(entry.get("passive_joints"),
                                  f"articulation {name!r} passive_joints"),
        params=_mapping_field(entry.get("params", {}),
                              f"articulation {name!r} params"),
    )


def normalize_articulations(entries: JsonValue) -> tuple[ArticulationSpec, ...]:
    """Normalize the run's ``articulations:`` list.

    The list is declaration data only: every articulation is ``root + model
    + joints + spawn``.  Which joints are driven is decided by the run-level
    ``controls[]`` block, not by a per-articulation mode.
    """
    if entries is None:
        raise ArticulationError(
            "run config declares no articulations; the v104 shape is "
            "`articulations: [{name: <robot>, params: ..., body_parts: ...}]` "
            "plus a top-level `controls:` list (the old top-level `robot:` "
            "block was removed -- no alias)")
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ArticulationError(
            "articulations must be a non-empty list of mappings")
    specs_list: list[ArticulationSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArticulationError(
                f"articulations entries must be mappings, got {entry!r}")
        specs_list.append(normalize_articulation(entry))
    specs = tuple(specs_list)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ArticulationError(
            f"articulations names must be unique, got {names!r}")
    return specs
