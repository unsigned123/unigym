"""Compose a world base + declared articulation assets into one MjSpec (v101 A3).

The historical G1 runs fused the robot into the world asset
(``scene_43dof_*.xml`` carries ``<include file="g1_29dof_with_hand.xml"/>``).
v101 A3 splits that asset into a :class:`world_base` scene and an ordered
``articulations[]`` list.  This module is the ONE composition engine that
turns the split declaration back into a single compiled model.

Composition mechanisms:

* **single articulation, no prefix** -- the base scene declares a
  ``<!-- articulations -->`` marker at the exact position the historical
  ``<include .../>`` lived.  The marker is replaced by the declared
  ``<include file="..."/>`` line in a sibling temp MJCF (same directory, so
  ``meshdir`` and relative asset paths resolve exactly as before) and loaded
  through ``MjSpec.from_file``.  This keeps the historical fused model
  byte-identical.  Without a marker the same result comes from
  ``MjSpec.attach(prefix="")``.
* **multiple assets / ``auto_prefix: true``** -- each asset is attached at a
  world-body frame.  When the backend option ``compose.auto_prefix`` is on,
  every asset-bearing articulation receives the short ``<name>_`` prefix;
  otherwise the assets attach with their own names and a body/joint/actuator
  collision fails fast.  The returned :class:`MjComposition` carries the
  backend-private :class:`MjNameMap`; neutral runtime surfaces never see a
  backend name.

No articulation declares ``model.mjcf`` -> :func:`compose_articulation_spec`
returns ``None`` and the caller keeps the historical fused load path.
"""
from __future__ import annotations

import logging
import math
import os
import uuid
import warnings
from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape

from unigym.errors import ConfigError

from ..name_map import MjComposition, MjNameMap

__all__ = [
    "ARTICULATION_MARKER",
    "articulation_assets",
    "compose_articulation_spec",
]

log = logging.getLogger(__name__)

ARTICULATION_MARKER = "<!-- articulations -->"
"""The base-scene marker the composer replaces with the include line(s)."""


def _field(entry, key, default=None):
    """Read *key* from an ``ArticulationSpec`` or a raw mapping."""
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


def articulation_assets(articulations: Sequence | None) -> tuple:
    """Ordered ``(name, mjcf_path)`` for articulations declaring ``model.mjcf``.

    Articulations without a declared MJCF asset (the A2 declaration-only
    shape, still used by the fused configs) are skipped: when the result is
    empty the caller keeps the historical fused load path untouched.
    """
    assets = []
    for entry in articulations or ():
        model = _field(entry, "model")
        mjcf = (_field(model, "mjcf") if model is not None else None)
        if mjcf:
            assets.append((str(_field(entry, "name") or ""), os.fspath(mjcf)))
    return tuple(assets)


def _prefix_for(name: str, auto_prefix: bool) -> str | None:
    """Backend-owned prefix policy (short ``<name>_`` when enabled)."""
    return f"{name}_" if auto_prefix and name else None


def _child_names(spec) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """``(bodies, joints, actuators)`` declared by one child MjSpec."""
    def names(table):
        return tuple(str(element.name) for element in table if element.name)

    return names(spec.bodies), names(spec.joints), names(spec.actuators)


def compose_articulation_spec(world_base_mjcf, articulations: Sequence | None,
                              *, auto_prefix: bool = False):
    """Return the composed :class:`MjComposition` or ``None`` when fused.

    Args:
        world_base_mjcf: The run-level ``world_base.mjcf`` path.
        articulations: The normalized ``ArticulationSpec`` list (or raw
            ``articulations[]`` mappings).
        auto_prefix: Backend option ``compose.auto_prefix`` (default false);
            when true every asset-bearing articulation is attached with the
            short ``<name>_`` prefix and a build warning is emitted.

    Returns:
        ``MjComposition`` whose ``spec`` is a pre-compile ``mujoco.MjSpec``
        plus the backend-private name map, or ``None`` when no articulation
        declares ``model.mjcf``.

    Raises:
        ConfigError: The base or a declared asset is missing, or the attached
            body/joint/actuator names collide.
    """
    assets = articulation_assets(articulations)
    if not assets:
        return None

    import mujoco

    base = os.fspath(world_base_mjcf)
    if not os.path.isfile(base):
        raise ConfigError(f"articulation composition: world_base MJCF not "
                          f"found: {base}")
    for _, path in assets:
        if not os.path.isfile(path):
            raise ConfigError(
                f"articulation composition: declared articulation MJCF not "
                f"found: {path}")

    if auto_prefix:
        warnings.warn(
            "auto_prefix: composing "
            f"{len(assets)} articulation asset(s) with backend-owned "
            f"prefixes {[f'{name}_' for name, _ in assets]}; neutral names "
            "stay unigym_name and the mapping remains backend-private",
            RuntimeWarning, stacklevel=2)
        log.warning("auto_prefix enabled for articulations %s",
                    [name for name, _ in assets])

    if len(assets) == 1 and not auto_prefix:
        spec = _compose_marker_include(mujoco, base, assets[0][1])
        if spec is not None:
            name_map = _build_name_map(mujoco, base, assets,
                                       prefixes={assets[0][0]: None},
                                       articulations=articulations)
            _assert_unique_names(spec)
            return MjComposition(spec=spec, name_map=name_map,
                                 applied_prefixes={assets[0][0]: None})
    return _compose_attach(mujoco, base, assets, auto_prefix=auto_prefix,
                           articulations=articulations)


def _compose_marker_include(mujoco, base: str, robot: str):
    """Replace the base marker with an include line in a sibling temp MJCF.

    Returns ``None`` when the base carries no marker or the sibling write is
    not possible (the caller then falls back to the attach route).
    """
    with open(base, encoding="utf-8") as handle:
        text = handle.read()
    if ARTICULATION_MARKER not in text:
        return None
    include = _include_target(base, robot)
    composed = text.replace(
        ARTICULATION_MARKER, f'<include file="{escape(include)}"/>', 1)
    temp = os.path.join(os.path.dirname(os.path.abspath(base)),
                        f".unigym-articulation-{uuid.uuid4().hex}.xml")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(composed)
        return mujoco.MjSpec.from_file(temp)
    except OSError:
        return None
    finally:
        try:
            os.unlink(temp)
        except OSError:
            pass


def _include_target(base: str, robot: str) -> str:
    """Same-directory relative include when possible, else the absolute path."""
    base_dir = os.path.dirname(os.path.abspath(base))
    robot_abs = os.path.abspath(robot)
    if os.path.dirname(robot_abs) == base_dir:
        return os.path.basename(robot_abs)
    return robot_abs


def _spawn_map(articulations) -> dict:
    """``{name: spawn}`` for the declared articulations (prefixed placement)."""
    out = {}
    for entry in articulations or ():
        name = str(_field(entry, "name") or "")
        if name:
            out[name] = dict(_field(entry, "spawn") or {})
    return out


def _frame_pose(spawn: dict):
    """Resolve a spawn mapping to ``(pos-or-None, quat-or-None)``.

    ``pos`` must be a finite 3-vector when present; ``yaw`` is radians and
    contributes the ``(w, 0, 0, z)`` root quaternion.  A mapping without
    placement keys leaves the frame at the origin (the historical default).
    """
    pos = spawn.get("pos") if spawn else None
    yaw = spawn.get("yaw") if spawn else None
    frame_pos = None
    frame_quat = None
    if pos is not None:
        values = [float(v) for v in pos]
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ConfigError(
                f"articulation spawn.pos must be a finite 3-vector, got "
                f"{pos!r}")
        frame_pos = values
    if yaw is not None:
        angle = float(yaw)
        if not math.isfinite(angle):
            raise ConfigError(
                f"articulation spawn.yaw must be finite radians, got {yaw!r}")
        frame_quat = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    return frame_pos, frame_quat


def _build_name_map(mujoco, base, assets, *, prefixes, articulations):
    """Collect each child asset's entity names into an :class:`MjNameMap`."""
    joints = {}
    bodies = {}
    actuators = {}
    for name, path in assets:
        child = mujoco.MjSpec.from_file(path)
        body_names, joint_names, actuator_names = _child_names(child)
        bodies[name] = body_names
        joints[name] = joint_names
        actuators[name] = actuator_names
    return MjNameMap(prefixes=prefixes, joints=joints, bodies=bodies,
                     actuators=actuators)


def _compose_attach(mujoco, base: str, assets: tuple, *, auto_prefix: bool,
                    articulations=None):
    """Attach each asset into the base spec (short ``<name>_`` when enabled)."""
    spec = mujoco.MjSpec.from_file(base)
    spawns = _spawn_map(articulations)
    prefixes: dict[str, str | None] = {}
    if not auto_prefix and len(assets) > 1:
        _assert_no_cross_asset_collision(mujoco, assets)
    place = auto_prefix or len(assets) > 1
    for name, path in assets:
        prefix = _prefix_for(name, auto_prefix)
        prefixes[name] = prefix
        child = mujoco.MjSpec.from_file(path)
        frame_pos = frame_quat = None
        if place:
            frame_pos, frame_quat = _frame_pose(spawns.get(name) or {})
        frame = spec.worldbody.add_frame(
            name=f"{prefix}attach" if prefix else None,
            pos=frame_pos, quat=frame_quat)
        spec.attach(child, prefix=prefix or "", suffix="", frame=frame)
    name_map = _build_name_map(mujoco, base, assets, prefixes=prefixes,
                               articulations=articulations)
    if auto_prefix:
        _rebind_base_cameras(spec, assets[0][0], name_map)
    _assert_unique_names(spec)
    return MjComposition(spec=spec, name_map=name_map,
                         applied_prefixes=prefixes)


def _rebind_base_cameras(spec, first_name: str, name_map: MjNameMap) -> None:
    """Point base-level cameras at the first articulation's prefixed body.

    The base scene owns world-level cameras (e.g. ``tracking`` targets the
    robot ``pelvis``); after prefixing the original name no longer exists.
    The policy robot (first declared articulation) is the natural bind.
    """
    for camera in spec.cameras:
        target = camera.targetbody
        if not target or spec.body(target) is not None:
            continue
        rebound = name_map.backend_body(first_name, target)
        if spec.body(rebound) is not None:
            camera.targetbody = rebound


def _assert_no_cross_asset_collision(mujoco, assets) -> None:
    """Fail fast with the duplicate names before ``MjSpec.attach`` renames."""
    for kind, attr in (("body", "bodies"), ("joint", "joints"),
                       ("actuator", "actuators")):
        names = []
        for _, path in assets:
            child = mujoco.MjSpec.from_file(path)
            table = getattr(child, attr)
            names.extend(str(element.name) for element in table
                         if element.name)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigError(
                f"articulation composition: duplicate {kind} names without "
                f"auto_prefix: {duplicates} | hint: enable "
                f"env.backend_options.<backend>.compose.auto_prefix or give "
                f"each asset distinct names")


def _assert_unique_names(spec) -> None:
    """Fail fast when the shared body/joint/actuator namespace collides."""
    for kind, elements in (("body", spec.bodies), ("joint", spec.joints),
                           ("actuator", spec.actuators)):
        names = [element.name for element in elements if element.name]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigError(
                f"articulation composition: duplicate {kind} names after "
                f"attachment: {duplicates} | hint: give each articulation a "
                f"unique `name` or enable env.backend_options."
                f"<backend>.compose.auto_prefix")
