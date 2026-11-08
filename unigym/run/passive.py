"""Passive-joint authoring sugar for the programmatic builders (v105).

A v105 recipe declares its full actuated vocabulary (``joint_sets``) and
every control group selects an explicit subset.  The programmatic builders
(``Run(robot=...)`` / ``World``) are authoring sugar, so the remaining
recipe joints are written into ``passive_joints`` instead of inventing a
runtime action face.  An explicit caller declaration always wins.
"""
from __future__ import annotations

import copy

__all__ = ["declare_passive_joints", "declare_world_passive_joints"]


def declare_passive_joints(entry: dict, controls: list) -> None:
    """Declare recipe joints no emitted control group drives as passive.

    Args:
        entry: One ``articulations[]`` mapping (mutated in place).
        controls: The emitted top-level ``controls[]`` groups.
    """
    if entry.get("passive_joints"):
        return
    art_name = entry.get("name")
    section = copy.deepcopy(entry)
    params = dict(section.get("params") or {})
    selector = params.pop("profile", None)
    if selector is not None:
        # v105: `params.profile` selects the recipe for a renamed
        # articulation (e.g. g1b); it never reaches the recipe factory.
        section["params"] = params
        section["name"] = str(selector)
    try:
        from unigym.robots import build_robot_recipe

        recipe = build_robot_recipe(section, with_isaaclab=False)
    except (AttributeError, KeyError, NotImplementedError, TypeError,
            ValueError):
        return
    sets = dict(getattr(recipe, "joint_sets", None) or {})
    actuated: list = []
    for joints in sets.values():
        for joint in joints:
            if joint not in actuated:
                actuated.append(joint)
    driven = set()
    for group in controls or ():
        if not isinstance(group, dict):
            continue
        if group.get("articulation") not in (None, art_name):
            continue
        joints = group.get("joints")
        if joints:
            driven.update(joints)
        elif group.get("joint_set"):
            driven.update(sets.get(group["joint_set"], ()))
    passive = [joint for joint in actuated if joint not in driven]
    if passive:
        entry["passive_joints"] = passive


def declare_world_passive_joints(cfg: dict) -> None:
    """Declare recipe joints of every world articulation no control drives."""
    controls = list(cfg.get("controls") or ())
    for entry in cfg.get("articulations") or ():
        if isinstance(entry, dict):
            declare_passive_joints(entry, controls)
