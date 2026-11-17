"""Passive-joint authoring sugar for the programmatic builders.

A profile declares its full actuated vocabulary (``joint_sets``) and
every control group selects an explicit subset.  The programmatic builders
(``Run(robot=...)`` / ``World``) are authoring sugar, so the remaining
profile joints are written into ``passive_joints`` instead of inventing a
runtime action group.  An explicit caller declaration always wins.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from unigym._typing import JsonValue, is_json_object

__all__ = ["declare_passive_joints", "declare_world_passive_joints"]


def _driven_joints(action_spec: object, art_name: object) -> set[str]:
    """The joint names any action group drives for one articulation."""
    driven: set[str] = set()
    groups = getattr(action_spec, "groups", None)
    if groups is not None:
        entries: Sequence[object] = tuple(groups)
    elif isinstance(action_spec, Sequence) and not isinstance(action_spec, (str, bytes)):
        entries = tuple(action_spec)
    else:
        return driven
    for group in entries:
        if isinstance(group, dict):
            articulation = group.get("articulation")
            joints = group.get("joints")
        else:
            articulation = getattr(group, "articulation", None)
            joints = getattr(group, "joints", None)
        if articulation not in (None, art_name):
            continue
        if isinstance(joints, (list, tuple)):
            driven.update(str(joint) for joint in joints)
    return driven


def declare_passive_joints(
    entry: dict[str, JsonValue],
    action_spec: object,
    *,
    profile: object | None = None,
) -> None:
    """Declare profile joints no emitted action group drives as passive.

    Args:
        entry: One ``articulations[]`` mapping (mutated in place).
        action_spec: The emitted top-level ``action_spec`` groups (a
            sequence of mappings or a complete ``OrderedActionSpec``).
        profile: An optional complete ``RobotProfile``; when absent the
            entry's registry declaration is resolved instead.
    """
    if entry.get("passive_joints"):
        return
    art_name = entry.get("name")
    if profile is None:
        section = copy.deepcopy(entry)
        raw_params = section.get("params")
        params: dict[str, JsonValue] = dict(raw_params) if isinstance(raw_params, dict) else {}
        selector = params.pop("profile", None)
        if selector is not None:
            # `params.profile` selects the profile for a renamed
            # articulation (e.g. g1b); it never reaches the profile factory.
            section["params"] = params
            section["name"] = str(selector)
        try:
            from unigym.robots import build_robot_profile

            profile = build_robot_profile(section)
        except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            return
    sets = dict(profile.joint_sets)
    actuated: list[str] = []
    for joints in sets.values():
        for joint in joints:
            if joint not in actuated:
                actuated.append(joint)
    driven = _driven_joints(action_spec, art_name)
    passive: list[JsonValue] = []
    for joint in actuated:
        if joint not in driven:
            passive.append(joint)
    if passive:
        entry["passive_joints"] = passive


def declare_world_passive_joints(cfg: dict[str, JsonValue]) -> None:
    """Declare profile joints of every world articulation no action group drives."""
    raw_action_spec = cfg.get("action_spec")
    action_spec: list[JsonValue] = (
        list(raw_action_spec) if isinstance(raw_action_spec, list) else []
    )
    raw_entries = cfg.get("articulations")
    entries: list[JsonValue] = list(raw_entries) if isinstance(raw_entries, list) else []
    for entry in entries:
        if is_json_object(entry):
            declare_passive_joints(entry, action_spec)
