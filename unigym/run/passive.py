"""Passive-joint authoring sugar for the programmatic builders.

A profile declares its full actuated vocabulary (``joint_sets``) and
every control group selects an explicit subset.  The programmatic builders
(``Run(robot=...)`` / ``World``) are authoring sugar, so the remaining
profile joints are written into ``passive_joints`` instead of inventing a
runtime action face.  An explicit caller declaration always wins.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from unigym._typing import JsonValue, is_json_object

__all__ = ["declare_passive_joints", "declare_world_passive_joints"]


def declare_passive_joints(
    entry: dict[str, JsonValue],
    controls: Sequence[JsonValue],
) -> None:
    """Declare profile joints no emitted control group drives as passive.

    Args:
        entry: One ``articulations[]`` mapping (mutated in place).
        controls: The emitted top-level ``controls[]`` groups.
    """
    if entry.get("passive_joints"):
        return
    art_name = entry.get("name")
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
    driven: set[str] = set()
    for group in controls or ():
        if not isinstance(group, dict):
            continue
        if group.get("articulation") not in (None, art_name):
            continue
        joints = group.get("joints")
        if joints:
            if isinstance(joints, (list, tuple)):
                driven.update(str(joint) for joint in joints)
        else:
            joint_set = group.get("joint_set")
            if isinstance(joint_set, str):
                driven.update(sets.get(joint_set, ()))
    passive: list[JsonValue] = []
    for joint in actuated:
        if joint not in driven:
            passive.append(joint)
    if passive:
        entry["passive_joints"] = passive


def declare_world_passive_joints(cfg: dict[str, JsonValue]) -> None:
    """Declare profile joints of every world articulation no control drives."""
    raw_controls = cfg.get("controls")
    controls: list[JsonValue] = list(raw_controls) if isinstance(raw_controls, list) else []
    raw_entries = cfg.get("articulations")
    entries: list[JsonValue] = list(raw_entries) if isinstance(raw_entries, list) else []
    for entry in entries:
        if is_json_object(entry):
            declare_passive_joints(entry, controls)
