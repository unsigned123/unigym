"""Robot articulation recipe (backend-agnostic; isaaclab typing-only).

``articulation_cfg`` is ``None`` for backends that do not consume an Isaac
Lab ``ArticulationCfg`` (e.g. the mujoco backend, which loads its own MJ XML
model); the isaaclab backend requires it.  The concrete type is only known
where isaaclab is importable, so it is kept typing-only (``TYPE_CHECKING``)
and typed as ``Optional[Any]`` at runtime — profiles built with
``with_isaaclab=False`` carry every non-asset field (joint sets, q0, feet,
body parts) unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class RobotRecipe:
    """The authoring recipe of one robot articulation (v105).

    The recipe supplies the backend/authoring facts (named joint sets, reset
    pose, feet, body parts, articulation asset); it is NOT a policy and
    carries no control mode -- drives are declared in the run-level
    ``controls[]``.

    Attributes:
        articulation_cfg: ready-to-use Isaac Lab ArticulationCfg (spawn pose,
            actuators, prim_path all configured).  ``None`` when the robot
            profile was built without isaaclab (mujoco backend: the simulator
            model comes from the run yaml's top-level ``world_base`` instead).
        joint_sets: the recipe's named joint vocabulary
            (``{set_name: (joint names in action order)}``).  ``body`` is the
            default body set; extra sets (``hand`` / ``right_hand`` / ...)
            are selected explicitly by ``ControlGroup.joint_set``.  This is
            the ONLY authoring source of joint names (v105).
        default_joint_pos: joint name -> default angle (rad), i.e. the policy
            standby pose (must match the policy's q0 by name).
        foot_bodies: foot body names in policy order FL, FR, RL, RR.
        body_parts: optional dict of body-part names (from yaml) for
            cross-robot configurability.  Keys:
              feet   -> list[str] (policy foot order)
              base   -> str (trunk link name)
              joints -> list[str] (full policy joint order; count varies by robot)
              hips   -> list[str] (hip joint names in policy order)
            When ``None``, the adapter falls back to its hardcoded defaults.

    Compatibility properties (read-only; V105.2 removes the authoring
    semantics):

        joint_order: the ``body`` set as a list.
        full_joint_names: the union of every declared set (or the explicit
            ``_full_joint_names`` document-order override).  Published by the
            ObsCatalog joint keys.
        extra_action_terms: optional extra NamedAction faces this profile
            exposes beyond the primary body face (``_extra_action_terms``) --
            ``{entry_key: [joint names in term input order]}``.  This is an
            isaaclab action-surface capability gate (e.g. ``actuate_hands``),
            NOT a joint-resolution source; control groups must name their
            ``joint_set`` explicitly.
    """

    articulation_cfg: Optional[Any]  # ArticulationCfg | None (see docstring)
    joint_sets: Mapping[str, tuple[str, ...]]
    default_joint_pos: Dict[str, float]
    foot_bodies: List[str]
    body_parts: Optional[Dict] = None
    # Compatibility overrides for the read-only derived properties; V105.2
    # deletes them.  They never participate in ControlGroup resolution.
    _full_joint_names: Optional[List[str]] = None
    _extra_action_terms: Optional[Dict[str, List[str]]] = None

    def __post_init__(self):
        sets: Dict[str, tuple[str, ...]] = {}
        for name, joints in dict(self.joint_sets or {}).items():
            names = tuple(str(joint) for joint in (joints or ()))
            if not names:
                raise ValueError(
                    f"RobotRecipe joint set {name!r} is empty")
            if len(set(names)) != len(names):
                raise ValueError(
                    f"RobotRecipe joint set {name!r} contains duplicate names")
            sets[str(name)] = names
        if not sets:
            raise ValueError("RobotRecipe must declare at least one joint set")
        object.__setattr__(self, "joint_sets", sets)

    # ------------------------------------------------------------------ #
    # read-only compatibility properties (derived; V105.2 removes them)
    # ------------------------------------------------------------------ #
    @property
    def joint_order(self) -> List[str]:
        """The ``body`` joint set (Isaac Lab / policy order)."""
        return list(self.joint_sets.get("body", ()))

    @property
    def full_joint_names(self) -> Optional[List[str]]:
        """Every declared joint in set-declaration order (or explicit order)."""
        if self._full_joint_names is not None:
            return list(self._full_joint_names)
        union: List[str] = []
        for joints in self.joint_sets.values():
            for joint in joints:
                if joint not in union:
                    union.append(joint)
        return union

    @property
    def extra_action_terms(self) -> Optional[Dict[str, List[str]]]:
        """Extra NamedAction faces (compatibility capability gate)."""
        if not self._extra_action_terms:
            return None
        return {str(name): list(joints)
                for name, joints in self._extra_action_terms.items()}

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)
