"""Neutral articulation IR (v101 A2, v104 authoring model).

An articulation is ``root + model + joints + spawn``.  There is no
articulation-level control mode: *how* a joint is driven (and which adapter
owns it) is declared explicitly in the run-level ``controls[]`` block
(:mod:`unigym.runtime.controls`), and joints that are not driven at all are
listed in :attr:`ArticulationSpec.passive_joints`.  The IR is pure data and
imports only the standard library -- no engine and no backend name is
allowed here (v100/v101 attribute neutrality).

v104 removed ``ControlSpec`` / ``JointSpec.mode`` / ``policy_articulation``
/ ``policy_section``: the drive declaration lives in ``controls[]`` and the
robot recipe only supplies the joint order / q0 / body parts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

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


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ArticulationError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise ArticulationError(f"{what} must be finite, got {value!r}")
    return number


def _optional_float(value: Any, what: str) -> float | None:
    if value is None:
        return None
    return _finite(value, what)


def _unknown_keys(mapping: Mapping, allowed: tuple[str, ...], what: str) -> None:
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

    def to_dict(self) -> dict:
        """Non-empty asset fields (stable key order)."""
        return {key: getattr(self, key) for key in ("mjcf", "urdf", "usd")
                if getattr(self, key) is not None}

    @classmethod
    def from_mapping(cls, value: Any, what: str = "model") -> AssetRef:
        """Normalize an ``AssetRef`` or ``{mjcf|urdf|usd: <path>}`` mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ArticulationError(
                f"{what} must be a mapping {{mjcf|urdf|usd: <path>}}, "
                f"got {value!r}")
        _unknown_keys(value, ("mjcf", "urdf", "usd"), what)
        return cls(mjcf=value.get("mjcf"), urdf=value.get("urdf"),
                   usd=value.get("usd"))


@dataclass(frozen=True)
class JointSpec:
    """One declared joint of an articulation (v104 authoring IR).

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
    def from_mapping(cls, name: str, params: Mapping | None = None
                     ) -> JointSpec:
        """Normalize one joint declaration (``{init, range, ...}``)."""
        params = dict(params or {})
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
        return cls(name=name, **params)


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
    spawn: Mapping = field(default_factory=dict)
    joints: tuple[JointSpec, ...] = ()
    body_parts: Mapping = field(default_factory=dict)
    passive_joints: tuple[str, ...] = ()
    params: Mapping = field(default_factory=dict)

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


def _joints_tuple(name: str, joints: Any) -> tuple[JointSpec, ...]:
    if not joints:
        return ()
    if isinstance(joints, Mapping):
        entries = []
        for jname, params in joints.items():
            entries.append(JointSpec.from_mapping(str(jname), params))
        return tuple(entries)
    out = []
    for entry in joints:
        if isinstance(entry, JointSpec):
            out.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise ArticulationError(
                f"articulation {name!r} joints entries must be mappings, "
                f"got {entry!r}")
        entry = dict(entry)
        jname = entry.pop("name", None)
        if jname is None:
            raise ArticulationError(
                f"articulation {name!r} joint entry is missing 'name'")
        out.append(JointSpec.from_mapping(str(jname), entry))
    return tuple(out)


def normalize_articulation(entry: Mapping) -> ArticulationSpec:
    """Normalize one ``articulations[]`` entry into an :class:`ArticulationSpec`.

    The config surface is a mapping with the IR fields
    (``name/model/root/spawn/joints/body_parts/passive_joints/params``);
    unknown keys fail fast (there is no shorthand and no alias).  The v104
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
    return ArticulationSpec(**dict(entry))


def normalize_articulations(entries: Any) -> tuple[ArticulationSpec, ...]:
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
    specs = tuple(normalize_articulation(entry) for entry in entries)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ArticulationError(
            f"articulations names must be unique, got {names!r}")
    return specs
